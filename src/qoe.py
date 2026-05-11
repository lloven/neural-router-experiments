"""QoE-optimised heterogeneous backend assignment (Phase 1d).

Implements the Quality-of-Experience routing strategy: calibrate F1, cost,
and latency per (cluster, backend) pair, then assign each cluster to the
backend that maximises a weighted QoE score.

Three assignment strategies:
  1. homogeneous  -- one backend for all clusters
  2. round-robin  -- cycle backends across clusters
  3. qoe-optimised -- assign via argmax QoE per cluster

Three weight presets:
  - accuracy_first: prioritise F1 (w_acc=0.7, w_cost=0.15, w_lat=0.15)
  - balanced:       equal trade-off (w_acc=0.34, w_cost=0.33, w_lat=0.33)
  - cost_first:     minimise cost   (w_acc=0.15, w_cost=0.7, w_lat=0.15)

Key classes:
  QoEAssigner       -- orchestrates calibration and assignment
  CalibrationResult -- stores per-(cluster, backend) metrics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .router import Cluster

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Weight presets
# ---------------------------------------------------------------------------

QOE_WEIGHT_PRESETS: dict[str, dict[str, float]] = {
    "accuracy_first": {"accuracy": 0.7, "cost": 0.15, "latency": 0.15},
    "balanced": {"accuracy": 0.34, "cost": 0.33, "latency": 0.33},
    "cost_first": {"accuracy": 0.15, "cost": 0.7, "latency": 0.15},
}

DEFAULT_WEIGHTS = QOE_WEIGHT_PRESETS["balanced"]


# ---------------------------------------------------------------------------
# C9: Perturbation hooks for the QoE adaptive loop
# ---------------------------------------------------------------------------

VALID_PERTURBATION_KINDS = (
    "none",
    "topic_restricted_cal",
    "latency_injection",
    "failure_injection",
)


@dataclass
class PerturbationSpec:
    """Perturbation hook specification for the QoE adaptive loop (C9).

    Three perturbation kinds:
      - ``topic_restricted_cal``: the calibration sample is drawn from a
        topic-restricted slice (``topic_mask``); evaluation runs on the
        full corpus. Tests whether QoE-optimised generalises from a
        narrow calibration set.
      - ``latency_injection``: at event index ``injection_event_index``,
        an extra ``injected_latency_s`` is added to every subsequent
        LLM call (treatment-verifiable via wall-clock).
      - ``failure_injection``: at event index ``injection_event_index``,
        the backend named ``backend_to_fail`` returns empty matches
        (F1=0). Target one of N backends, not all — this is a partial
        degradation experiment, not a total outage.

    Every perturbation must be applied verifiably; ``is_noop()`` is
    the smoke-test contract that a default-constructed spec changes
    nothing.

    ``validate()`` raises on invalid configuration (missing required
    fields per kind); never log-and-continue.
    """
    kind: str = "none"
    topic_mask: Optional[Sequence[int]] = None
    injection_event_index: Optional[int] = None
    injected_latency_s: Optional[float] = None
    backend_to_fail: Optional[str] = None

    def is_noop(self) -> bool:
        return self.kind == "none"

    def validate(self) -> "PerturbationSpec":
        """Raise ValueError if the spec is malformed for its kind.

        Perturbation hooks must not silently swallow errors. Returns
        self so callers can chain: ``spec = PerturbationSpec(...).validate()``.
        """
        if self.kind not in VALID_PERTURBATION_KINDS:
            raise ValueError(
                f"unknown perturbation kind {self.kind!r}; "
                f"expected one of {VALID_PERTURBATION_KINDS}"
            )
        if self.kind == "topic_restricted_cal":
            if self.topic_mask is None:
                raise ValueError(
                    "topic_mask is required for kind='topic_restricted_cal'"
                )
        elif self.kind == "latency_injection":
            if self.injection_event_index is None:
                raise ValueError(
                    "injection_event_index is required for kind='latency_injection'"
                )
            if self.injected_latency_s is None:
                raise ValueError(
                    "injected_latency_s is required for kind='latency_injection'"
                )
        elif self.kind == "failure_injection":
            if self.injection_event_index is None:
                raise ValueError(
                    "injection_event_index is required for kind='failure_injection'"
                )
            if self.backend_to_fail is None:
                raise ValueError(
                    "backend_to_fail is required for kind='failure_injection' (target one backend)"
                )
        return self


def apply_latency_injection(
    perturbation: Optional[PerturbationSpec], event_idx: int
) -> float:
    """C9 / Task 1.3: latency-injection hook.

    Returns the additional latency (seconds) to inject for the event at
    ``event_idx``. Zero for any non-latency_injection perturbation (or
    None), and zero for events with ``event_idx < injection_event_index``.

    Callers wrap their LLM calls with::

        extra = apply_latency_injection(self.perturbation, event_idx)
        if extra > 0: time.sleep(extra)

    The per-event return value is the treatment-verifiable signal —
    a smoke test that compares pre-/post-injection wall-clock latencies
    asserts post >= pre + injected_latency_s for at least one cell.
    """
    if perturbation is None or perturbation.kind != "latency_injection":
        return 0.0
    if perturbation.injection_event_index is None or perturbation.injected_latency_s is None:
        return 0.0
    if event_idx >= perturbation.injection_event_index:
        return float(perturbation.injected_latency_s)
    return 0.0


def total_latency_offset_for_queue(
    perturbation: Optional[PerturbationSpec],
    event_indices: Sequence[int],
) -> float:
    """C9 / Task 1.6: aggregate latency offset for a cluster's event queue.

    Returns the total seconds to sleep before the cluster's LLM call,
    summing per-event injected latency for all events with idx >=
    ``injection_event_index``. The cluster-level wiring sleeps once with
    this aggregate (treatment-verifiable via wall-clock difference
    against a no-perturbation baseline).

    Zero for any non-latency_injection perturbation (or None), and zero
    when the queue contains no post-injection events.
    """
    if perturbation is None or perturbation.kind != "latency_injection":
        return 0.0
    per_event = float(perturbation.injected_latency_s or 0.0)
    cutoff = perturbation.injection_event_index
    if cutoff is None or per_event <= 0.0:
        return 0.0
    return per_event * sum(1 for idx in event_indices if idx >= cutoff)


def partition_queue_by_failure(
    perturbation: Optional[PerturbationSpec],
    backend_name: str,
    event_indices: Sequence[int],
) -> tuple[list[int], list[int]]:
    """C9 / Task 1.7: split a cluster's queue into (healthy, failed).

    For ``failure_injection`` on the targeted backend, events with
    ``idx >= injection_event_index`` go into ``failed`` (LLM call skipped,
    empty match emitted); the rest go into ``healthy`` (LLM call as
    normal). A non-targeted backend's queue is entirely healthy
    (one-of-N partial degradation, not a total outage).
    Returns (healthy, failed) with original ordering preserved within
    each list.
    """
    indices = list(event_indices)
    if perturbation is None or perturbation.kind != "failure_injection":
        return indices, []
    cutoff = perturbation.injection_event_index
    target = perturbation.backend_to_fail
    if cutoff is None or target is None or backend_name != target:
        return indices, []
    healthy = [idx for idx in indices if idx < cutoff]
    failed = [idx for idx in indices if idx >= cutoff]
    return healthy, failed


def is_backend_failed(
    perturbation: Optional[PerturbationSpec],
    backend_name: str,
    event_idx: int,
) -> bool:
    """C9 / Task 1.4: backend-failure-injection predicate.

    Returns True iff the active perturbation is ``failure_injection`` and
    ``event_idx >= injection_event_index`` and ``backend_name`` matches
    ``perturbation.backend_to_fail``. The QoE matching path treats a
    failed-backend cell as if the LLM returned an empty match
    (F1=0 contribution); the assigner can then re-route across the
    remaining healthy backends (one-of-N partial degradation).
    """
    if perturbation is None or perturbation.kind != "failure_injection":
        return False
    if (
        perturbation.injection_event_index is None
        or perturbation.backend_to_fail is None
    ):
        return False
    if backend_name != perturbation.backend_to_fail:
        return False
    return event_idx >= perturbation.injection_event_index


# ---------------------------------------------------------------------------
# QoE score computation
# ---------------------------------------------------------------------------

def compute_qoe_score(
    f1: float,
    cost: float,
    latency: float,
    weights: dict[str, float],
    cost_max: float = 20.0,
    latency_max: float = 30.0,
) -> float:
    """Compute a normalised QoE score in [0, 1].

    QoE = w_acc * f1 + w_cost * (1 - cost/cost_max) + w_lat * (1 - lat/lat_max)

    All components are clipped to [0, 1] before weighting.

    Args:
        f1: F1 score (already in [0, 1]).
        cost: Monetary cost (normalised against cost_max).
        latency: Latency in seconds (normalised against latency_max).
        weights: Dict with keys "accuracy", "cost", "latency".
        cost_max: Reference maximum cost for normalisation.
        latency_max: Reference maximum latency for normalisation.

    Returns:
        Weighted QoE score in [0, 1].
    """
    acc_score = max(0.0, min(1.0, f1))
    cost_score = max(0.0, min(1.0, 1.0 - cost / max(cost_max, 1e-9)))
    lat_score = max(0.0, min(1.0, 1.0 - latency / max(latency_max, 1e-9)))

    qoe = (
        weights.get("accuracy", 0.34) * acc_score
        + weights.get("cost", 0.33) * cost_score
        + weights.get("latency", 0.33) * lat_score
    )
    return max(0.0, min(1.0, qoe))


# ---------------------------------------------------------------------------
# Calibration result storage
# ---------------------------------------------------------------------------

class CalibrationResult:
    """Stores per-(cluster_id, backend) calibration metrics."""

    def __init__(self):
        self._data: dict[tuple[int, str], dict[str, float]] = {}

    def add(
        self,
        cluster_id: int,
        backend: str,
        f1: float,
        cost: float,
        latency: float,
    ) -> None:
        self._data[(cluster_id, backend)] = {
            "f1": f1,
            "cost": cost,
            "latency": latency,
        }

    def get(self, cluster_id: int, backend: str) -> dict[str, float]:
        return self._data[(cluster_id, backend)]

    def cluster_ids(self) -> set[int]:
        return {cid for cid, _ in self._data}

    def backends_for_cluster(self, cluster_id: int) -> list[str]:
        return [b for cid, b in self._data if cid == cluster_id]

    def items(self):
        return self._data.items()


# ---------------------------------------------------------------------------
# Assignment strategies
# ---------------------------------------------------------------------------

def assign_homogeneous(
    clusters: list[Cluster],
    backend: str,
) -> dict[int, str]:
    """Assign the same backend to all clusters.

    Args:
        clusters: List of Cluster objects.
        backend: Backend model ID to assign.

    Returns:
        Dict mapping cluster_id to backend.
    """
    return {c.id: backend for c in clusters}


def assign_round_robin(
    clusters: list[Cluster],
    backends: list[str],
) -> dict[int, str]:
    """Cycle backends across clusters in order.

    Args:
        clusters: List of Cluster objects.
        backends: List of backend model IDs to cycle through.

    Returns:
        Dict mapping cluster_id to backend.
    """
    return {
        c.id: backends[i % len(backends)]
        for i, c in enumerate(clusters)
    }


def _minmax(values: list[float], invert: bool = False) -> list[float]:
    """Scale values to [0, 1]. Constant inputs map to all 0.5 (no preference).

    `invert=True` flips the scale (used for cost and latency, where lower
    is better).
    """
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.5 for _ in values]
    scaled = [(v - lo) / (hi - lo) for v in values]
    return [1.0 - s for s in scaled] if invert else scaled


def assign_qoe_optimised(
    calibration: CalibrationResult,
    weights: dict[str, float],
) -> dict[int, str]:
    """Assign each cluster to the backend with the highest QoE score.

    Per-cluster min-max normalisation: F1, cost, and latency are each
    scaled to [0, 1] across the candidate backends *for that cluster*
    before weighting. This replaces the earlier hardcoded
    `cost_max=20.0` / `latency_max=30.0` constants that saturated the
    cost and latency terms for self-hosted-model deployments
    (cancelled job 6620413).

    Args:
        calibration: Calibration results from the calibration phase.
        weights: QoE weight config (accuracy, cost, latency).

    Returns:
        Dict mapping cluster_id to the best backend.
    """
    w_acc = weights.get("accuracy", 0.34)
    w_cost = weights.get("cost", 0.33)
    w_lat = weights.get("latency", 0.33)

    assignment: dict[int, str] = {}
    for cluster_id in calibration.cluster_ids():
        backends = calibration.backends_for_cluster(cluster_id)
        if not backends:
            continue

        f1s = [calibration.get(cluster_id, b)["f1"] for b in backends]
        costs = [calibration.get(cluster_id, b)["cost"] for b in backends]
        lats = [calibration.get(cluster_id, b)["latency"] for b in backends]

        f1_n = _minmax(f1s)
        cost_n = _minmax(costs, invert=True)
        lat_n = _minmax(lats, invert=True)

        scores = [
            w_acc * f1_n[i] + w_cost * cost_n[i] + w_lat * lat_n[i]
            for i in range(len(backends))
        ]
        best_idx = max(range(len(backends)), key=lambda i: scores[i])
        assignment[cluster_id] = backends[best_idx]
    return assignment


# ---------------------------------------------------------------------------
# QoEAssigner orchestrator
# ---------------------------------------------------------------------------

class QoEAssigner:
    """Orchestrates calibration and QoE-based backend assignment.

    Lifecycle:
      1. Create with clusters, backends, and weight config.
      2. Call calibrate() with a sample of events to measure per-(cluster, backend) metrics.
      3. Call assign() to get the optimal per-cluster assignment.

    Args:
        clusters: Subscription clusters from the offline phase.
        backends: List of LLM backend model IDs.
        weights: QoE weight config. Defaults to balanced.
        calibration_fraction: Fraction of events to use for calibration (default 0.1).
    """

    def __init__(
        self,
        clusters: list[Cluster],
        backends: list[str],
        weights: dict[str, float] | None = None,
        calibration_fraction: float = 0.1,
        perturbation: PerturbationSpec | None = None,
        seed: int = 42,
    ):
        self.clusters = clusters
        self.backends = backends
        self.weights = weights or dict(DEFAULT_WEIGHTS)
        self.calibration_fraction = calibration_fraction
        # Thread the seed through every randomness source — calibration
        # sample selection (_sample_calibration_events) and inner RouterConfig
        # (calibrate). Without this, --seeds 42,123,... draws identical
        # samples and the 5-seed CI95 measures k-means + LLM jitter only.
        self.seed = seed
        # C9: validate perturbation eagerly; never log-and-continue.
        self.perturbation = (
            perturbation.validate() if perturbation is not None else None
        )
        self.calibration: CalibrationResult | None = None

    def _sample_calibration_events(self, events: list) -> list:
        """Draw the calibration sample, applying any active perturbation.

        Default behaviour: deterministic random sample of size
        ``len(events) * self.calibration_fraction`` (rng seed 42 for
        reproducibility). The sample preserves event-list order to keep
        downstream code stable.

        With ``perturbation.kind == 'topic_restricted_cal'``: events are
        first filtered to those whose ground_truth intersects the
        ``perturbation.topic_mask``, then the fraction is applied to the
        filtered pool. This is the treatment whose application is
        verifiable from the sample's topic distribution.

        An empty calibration pool (e.g., topic_mask matches no events)
        raises ValueError rather than returning an empty sample.
        """
        import numpy as np

        pool = events
        if self.perturbation is not None and self.perturbation.kind == "topic_restricted_cal":
            mask = set(self.perturbation.topic_mask or [])
            pool = [e for e in events if set(e.ground_truth) & mask]
            if not pool:
                raise ValueError(
                    f"topic_restricted_cal: empty calibration pool "
                    f"(topic_mask={list(mask)!r} matched no events)"
                )

        n_sample = max(1, int(len(pool) * self.calibration_fraction))
        # Use self.seed (threaded from outer per-seed loop), not a
        # hardcoded 42. Pre-fix this line silently disabled the 5-seed
        # variance source the manuscript implies.
        rng = np.random.RandomState(self.seed)
        if n_sample >= len(pool):
            # Sampling more than available — return all (preserve order)
            return list(pool)
        sample_indices = rng.choice(len(pool), n_sample, replace=False)
        return [pool[i] for i in sorted(sample_indices)]

    def calibrate(
        self,
        events: list,
        llm_clients: dict[str, Any],
        embedding_model: Any,
        dataset: Any,
    ) -> CalibrationResult:
        """Run calibration: evaluate each (cluster, backend) on a sample.

        Args:
            events: Full event list (a sample is drawn from this).
            llm_clients: Dict mapping backend name to LLMClient instance.
            embedding_model: Shared embedding model.
            dataset: Dataset for evaluation.

        Returns:
            CalibrationResult with per-(cluster, backend) metrics.
        """
        import time
        import numpy as np

        from .router import NeuralRouter, RouterConfig, ABLATION_CONFIGS
        from .evaluation import evaluate_matches

        cal = CalibrationResult()
        # C9: route through the perturbation-aware helper so every code path
        # that draws a calibration sample respects the active perturbation.
        sample_events = self._sample_calibration_events(events)

        for cluster in self.clusters:
            for backend_name in self.backends:
                llm = llm_clients[backend_name]
                if hasattr(llm, 'reset_stats'):
                    llm.reset_stats()

                # Create a minimal router with A3 config for this backend.
                # Thread self.seed (not hardcoded 42) so calibration's
                # k-means partition varies across seeds.
                config = RouterConfig(
                    **{
                        **ABLATION_CONFIGS["A3"].__dict__,
                        "llm_model": llm.model,
                        "seed": self.seed,
                        "k": 1,  # single cluster (we're testing per-cluster)
                    }
                )

                # Build a temporary single-cluster dataset
                from .data import Dataset as DS
                temp_dataset = DS(
                    name=dataset.name,
                    short_name=dataset.short_name,
                    events=sample_events,
                    subscriptions=cluster.subscriptions,
                    metadata=dataset.metadata,
                )

                router = NeuralRouter(config=config, llm_client=llm, embedding_model=embedding_model)

                t0 = time.time()
                router.optimize_subscriptions(cluster.subscriptions)
                matches = router.match_events(sample_events)
                latency = time.time() - t0

                result = evaluate_matches(
                    matches=matches,
                    dataset=temp_dataset,
                    config_name=f"cal_{cluster.id}_{backend_name}",
                    seed=42,
                    router_stats=router.stats,
                )

                cal.add(
                    cluster_id=cluster.id,
                    backend=backend_name,
                    f1=result.f1.mean,
                    cost=result.cost_per_1k,
                    latency=latency,
                )

                logger.info(
                    f"  Calibration: cluster={cluster.id}, backend={backend_name}, "
                    f"F1={result.f1.mean:.4f}, cost={result.cost_per_1k:.4f}, lat={latency:.1f}s"
                )

        self.calibration = cal
        return cal

    def assign(
        self,
        strategy: str = "qoe_optimised",
        homogeneous_backend: str | None = None,
    ) -> dict[int, str]:
        """Get per-cluster backend assignment using the specified strategy.

        Args:
            strategy: "homogeneous", "round_robin", or "qoe_optimised".
            homogeneous_backend: Required for "homogeneous" strategy.

        Returns:
            Dict mapping cluster_id to backend name.
        """
        if strategy == "homogeneous":
            if homogeneous_backend is None:
                raise ValueError("homogeneous strategy requires homogeneous_backend")
            return assign_homogeneous(self.clusters, homogeneous_backend)
        elif strategy == "round_robin":
            return assign_round_robin(self.clusters, self.backends)
        elif strategy == "qoe_optimised":
            if self.calibration is None:
                raise RuntimeError("Call calibrate() before qoe_optimised assignment")
            return assign_qoe_optimised(self.calibration, self.weights)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
