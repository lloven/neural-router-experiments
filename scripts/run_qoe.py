#!/usr/bin/env python3
"""QoE heterogeneous backend assignment experiment (Phase 1d).

Evaluates three routing strategies for assigning LLM backends to
subscription clusters:

  1. homogeneous   -- one backend for all clusters (baseline)
  2. round_robin   -- cycle backends across clusters
  3. qoe_optimised -- calibrate F1/cost/latency per (cluster, backend),
                      then assign via argmax QoE

Three weight presets:
  - accuracy_first: F1-dominated
  - balanced:       equal trade-off
  - cost_first:     cost-dominated

Usage:
    python scripts/run_qoe.py --dataset D1
    python scripts/run_qoe.py --dataset D1 --strategies homogeneous,qoe_optimised
    python scripts/run_qoe.py --dataset D1 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.data import load_dataset_by_name, Dataset
from src.router import NeuralRouter, RouterConfig, ABLATION_CONFIGS
from src.embeddings import EmbeddingModel
from src.evaluation import evaluate_matches
from src.llm import LLMClient, DryRunLLMClient
from src.qoe import (
    QoEAssigner,
    QOE_WEIGHT_PRESETS,
    assign_homogeneous,
    assign_round_robin,
    PerturbationSpec,
)
from src.config import load_profile, output_dir_for_mode, get_model_overrides

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_STRATEGIES = ["homogeneous", "round_robin", "qoe_optimised"]
DEFAULT_WEIGHT_PRESETS = ["accuracy_first", "balanced", "cost_first"]
DEFAULT_SEEDS = [42, 123, 456, 789, 0]


def _append_row(row: dict, path: Path):
    """Append a single result row to a CSV, creating it with header if needed.

    L51: fail-fast on schema drift. If the CSV exists with a header that
    doesn't match the row's keys, raise rather than appending mixed-width
    rows that downstream pandas.read_csv cannot parse.
    """
    write_header = not path.exists() or path.stat().st_size == 0
    if not write_header:
        with open(path) as fh:
            existing_header = fh.readline().rstrip("\n").split(",")
        if existing_header != list(row.keys()):
            raise RuntimeError(
                f"_append_row schema mismatch at {path}: "
                f"existing header has {len(existing_header)} columns "
                f"({existing_header}), new row has {len(row)} columns "
                f"({list(row.keys())}). Wipe the output dir before re-running, "
                "or rotate the path."
            )
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", header=write_header, index=False)


def _write_progress(output_dir: Path, payload: dict) -> None:
    """L63: per-batch progress writer. Atomic write to .progress.json so
    a mid-run reader sees a consistent snapshot.

    Writes through a tmp file + rename so partial writes are never visible.
    """
    import json, os, tempfile, time
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {**payload, "ts": time.time()}
    p = output_dir / ".progress.json"
    fd, tmp = tempfile.mkstemp(prefix=".progress_", suffix=".tmp", dir=str(output_dir))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        os.replace(tmp, p)
    except Exception:
        try: os.unlink(tmp)
        except Exception: pass


def _run_with_assignment(
    dataset: Dataset,
    clusters: list,
    assignment: dict[int, str],
    llm_clients: dict[str, LLMClient],
    embedder: EmbeddingModel,
    seed: int,
    perturbation: PerturbationSpec | None = None,
) -> dict:
    """Run matching with per-cluster backend assignment and evaluate.

    Args:
        dataset: Full dataset.
        clusters: Pre-optimised cluster list.
        assignment: Dict mapping cluster_id to backend name.
        llm_clients: Dict mapping backend name to LLMClient.
        embedder: Embedding model.
        seed: Random seed.

    Returns:
        Dict with aggregated metrics.
    """
    from src.router import MatchResult, RouterStats

    all_matches: dict[str, MatchResult] = {}
    total_stats = RouterStats()
    total_stats.num_events = len(dataset.events)
    total_stats.num_subscriptions = dataset.num_subscriptions

    # Embed events once
    event_texts = [e.text for e in dataset.events]
    event_embeddings = embedder.encode(event_texts)

    # Assign events to clusters (cosine pre-filter with tau=0.3)
    from sklearn.metrics.pairwise import cosine_similarity
    centroids = np.array([c.centroid for c in clusters])
    similarities = cosine_similarity(event_embeddings, centroids)

    for cluster in clusters:
        cluster.queue = []

    # C9: track per-cluster global event indices in parallel with cluster.queue
    # so the perturbation hooks (latency injection, failure injection) can be
    # applied at evaluation time. The index is the position in dataset.events.
    cluster_queue_indices: dict = {c.id: [] for c in clusters}

    for i, event in enumerate(dataset.events):
        assigned = False
        for j, cluster in enumerate(clusters):
            if similarities[i, j] >= 0.3:
                cluster.queue.append(event)
                cluster_queue_indices[cluster.id].append(i)
                assigned = True
        if not assigned:
            nearest = int(np.argmax(similarities[i]))
            clusters[nearest].queue.append(event)
            cluster_queue_indices[clusters[nearest].id].append(i)

    # Match each cluster using its assigned backend
    from src.qoe import (
        total_latency_offset_for_queue,
        partition_queue_by_failure,
    )
    from src.router import MatchResult as _MR

    t0 = time.time()
    for cluster in clusters:
        if not cluster.queue:
            continue

        backend_name = assignment.get(cluster.id, list(llm_clients.keys())[0])
        llm = llm_clients[backend_name]

        # C9 / Task 1.7: failure injection — split the queue into healthy
        # (LLM call as normal) and failed (skip the call, emit empty match
        # for treatment-verifiable F1=0 contribution).
        queue_indices = cluster_queue_indices.get(cluster.id, [])
        if perturbation is not None and queue_indices:
            healthy_indices, failed_indices = partition_queue_by_failure(
                perturbation, backend_name, queue_indices,
            )
            if failed_indices:
                # Emit empty matches for the failed events — never reach the LLM.
                failed_set = set(failed_indices)
                for idx, event in zip(queue_indices, cluster.queue):
                    if idx in failed_set:
                        all_matches.setdefault(
                            event.id,
                            _MR(event_id=event.id, matched_subscription_ids=[]),
                        )
                # Restrict the queue to healthy events for the LLM call.
                healthy_set = set(healthy_indices)
                cluster.queue = [
                    e for idx, e in zip(queue_indices, cluster.queue)
                    if idx in healthy_set
                ]
                cluster_queue_indices[cluster.id] = list(healthy_indices)
                queue_indices = list(healthy_indices)
                if not cluster.queue:
                    # Whole queue failed; skip remaining LLM setup for this cluster.
                    continue

        # C9 / Task 1.6: latency injection — sleep once per cluster with the
        # aggregate per-event offset for events at-or-after injection_event_index.
        if perturbation is not None and queue_indices:
            extra = total_latency_offset_for_queue(perturbation, queue_indices)
            if extra > 0.0:
                time.sleep(extra)

        config = RouterConfig(
            **{
                **ABLATION_CONFIGS["A3"].__dict__,
                "llm_model": llm.model,
                "seed": seed,
                "k": 1,
            }
        )

        router = NeuralRouter(config=config, llm_client=llm, embedding_model=embedder)
        # Set up the cluster with subscriptions (bypass optimize_subscriptions)
        from src.router import Cluster as ClusterType
        temp_cluster = ClusterType(
            id=0,
            subscriptions=cluster.subscriptions,
            compressed=cluster.compressed,
            centroid=cluster.centroid,
            llm=llm.model,
            queue=cluster.queue,
        )
        router.clusters = [temp_cluster]
        router.stats.num_subscriptions = len(cluster.active_subscriptions)

        # Run matching on this cluster
        cluster_results = router._run_sync_matching()
        for event_id, result in cluster_results.items():
            if event_id in all_matches:
                all_matches[event_id].matched_subscription_ids.extend(
                    result.matched_subscription_ids
                )
            else:
                all_matches[event_id] = result

        total_stats.llm_invocations += router.stats.llm_invocations
        total_stats.total_prompt_tokens += router.stats.total_prompt_tokens
        total_stats.total_response_tokens += router.stats.total_response_tokens

    total_stats.total_time_s = time.time() - t0

    # Build MatchResult list in event order
    from src.router import MatchResult
    final_matches = []
    for event in dataset.events:
        if event.id in all_matches:
            result = all_matches[event.id]
            # Deduplicate and trim to kappa=3
            seen = set()
            unique = []
            for sid in result.matched_subscription_ids:
                if sid not in seen:
                    seen.add(sid)
                    unique.append(sid)
            result.matched_subscription_ids = unique[:3]
            final_matches.append(result)
        else:
            final_matches.append(MatchResult(event_id=event.id, matched_subscription_ids=[]))

    eval_result = evaluate_matches(
        matches=final_matches,
        dataset=dataset,
        config_name="qoe",
        seed=seed,
        router_stats=total_stats,
    )

    return {
        "f1": eval_result.f1.mean,
        "precision": eval_result.precision.mean,
        "recall": eval_result.recall.mean,
        "invocations": eval_result.invocations,
        "cost_per_1k": eval_result.cost_per_1k,
        "latency_s": total_stats.total_time_s,
        "tokens_prompt": eval_result.tokens_prompt,
        "tokens_response": eval_result.tokens_response,
    }


def run_qoe_experiment(
    dataset: Dataset,
    llm_clients: dict[str, LLMClient],
    embedder: EmbeddingModel,
    strategies: list[str] = None,
    weight_presets: list[str] = None,
    seeds: list[int] = None,
    output_dir: Path = Path("results"),
    calibration_fraction: float = 0.1,
    perturbation: PerturbationSpec | None = None,
) -> pd.DataFrame:
    """Run the full QoE experiment: all (strategy, weight, seed) combos.

    Args:
        dataset: Full dataset.
        llm_clients: Dict mapping backend name to LLMClient.
        embedder: Embedding model.
        strategies: List of strategies to test.
        weight_presets: List of QoE weight preset names.
        seeds: Random seeds.
        output_dir: Output directory.

    Returns:
        DataFrame with all results.
    """
    strategies = strategies or DEFAULT_STRATEGIES
    weight_presets = weight_presets or DEFAULT_WEIGHT_PRESETS
    seeds = seeds or DEFAULT_SEEDS
    backend_names = list(llm_clients.keys())

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"qoe_{dataset.short_name}.csv"
    rows = []

    # L63: estimate total cells for the progress reporter
    n_homogeneous = len(backend_names) if "homogeneous" in strategies else 0
    n_round_robin = 1 if "round_robin" in strategies else 0
    n_qoe = len(weight_presets) if "qoe_optimised" in strategies else 0
    cells_per_seed = n_homogeneous + n_round_robin + n_qoe
    total_cells = cells_per_seed * len(seeds)
    completed_cells = 0

    _write_progress(output_dir, {
        "phase": "starting", "completed_cells": 0, "total_cells": total_cells,
        "n_seeds": len(seeds), "cells_per_seed": cells_per_seed,
        "strategies": strategies, "weight_presets": weight_presets,
        "backends": backend_names,
    })

    for seed in seeds:
        # Optimise subscriptions once per seed (using A3 config)
        # Use the first available backend for clustering
        first_llm = list(llm_clients.values())[0]
        config = RouterConfig(
            **{
                **ABLATION_CONFIGS["A3"].__dict__,
                "llm_model": first_llm.model,
                "seed": seed,
                "k": min(max(1, dataset.num_subscriptions // 2), 30),
            }
        )
        router = NeuralRouter(config=config, llm_client=first_llm, embedding_model=embedder)
        clusters = router.optimize_subscriptions(dataset.subscriptions)

        for strategy in strategies:
            if strategy == "homogeneous":
                # Run once per backend
                for backend_name in backend_names:
                    _write_progress(output_dir, {
                        "phase": "running", "cell": f"homogeneous/{backend_name}/seed{seed}",
                        "completed_cells": completed_cells, "total_cells": total_cells,
                    })
                    assignment = assign_homogeneous(clusters, backend_name)
                    metrics = _run_with_assignment(
                        dataset, clusters, assignment, llm_clients, embedder, seed,
                        perturbation=perturbation,
                    )
                    row = {
                        "strategy": "homogeneous",
                        "weight_preset": "n/a",
                        "backend": backend_name,
                        "seed": seed,
                        "dataset": dataset.short_name,
                        "calibration_fraction": calibration_fraction,
                        "perturbation_kind": (perturbation.kind if perturbation else "none"),
                        "assignment": str(assignment),
                        **metrics,
                    }
                    rows.append(row)
                    _append_row(row, csv_path)
                    completed_cells += 1
                    _write_progress(output_dir, {
                        "phase": "cell_done",
                        "cell": f"homogeneous/{backend_name}/seed{seed}",
                        "completed_cells": completed_cells, "total_cells": total_cells,
                        "last_f1": metrics["f1"], "last_cost": metrics["cost_per_1k"],
                        "last_latency_s": metrics.get("latency_s"),
                    })
                    logger.info(
                        f"  homogeneous/{backend_name} seed={seed}: "
                        f"F1={metrics['f1']:.4f}, cost={metrics['cost_per_1k']:.4f} "
                        f"({completed_cells}/{total_cells} cells)"
                    )

            elif strategy == "round_robin":
                _write_progress(output_dir, {
                    "phase": "running", "cell": f"round_robin/seed{seed}",
                    "completed_cells": completed_cells, "total_cells": total_cells,
                })
                assignment = assign_round_robin(clusters, backend_names)
                metrics = _run_with_assignment(
                    dataset, clusters, assignment, llm_clients, embedder, seed,
                    perturbation=perturbation,
                )
                row = {
                    "strategy": "round_robin",
                    "weight_preset": "n/a",
                    "backend": "mixed",
                    "seed": seed,
                    "dataset": dataset.short_name,
                    "calibration_fraction": calibration_fraction,
                    "perturbation_kind": (perturbation.kind if perturbation else "none"),
                    "assignment": str(assignment),
                    **metrics,
                }
                rows.append(row)
                _append_row(row, csv_path)
                completed_cells += 1
                _write_progress(output_dir, {
                    "phase": "cell_done", "cell": f"round_robin/seed{seed}",
                    "completed_cells": completed_cells, "total_cells": total_cells,
                    "last_f1": metrics["f1"], "last_cost": metrics["cost_per_1k"],
                    "last_latency_s": metrics.get("latency_s"),
                })
                logger.info(
                    f"  round_robin seed={seed}: "
                    f"F1={metrics['f1']:.4f}, cost={metrics['cost_per_1k']:.4f} "
                    f"({completed_cells}/{total_cells} cells)"
                )

            elif strategy == "qoe_optimised":
                _write_progress(output_dir, {
                    "phase": "calibrating",
                    "cell": f"qoe_optimised/calibration/seed{seed}",
                    "completed_cells": completed_cells, "total_cells": total_cells,
                })
                assigner = QoEAssigner(
                    clusters=clusters,
                    backends=backend_names,
                    calibration_fraction=calibration_fraction,
                    perturbation=perturbation,
                    seed=seed,
                )
                assigner.calibrate(
                    events=dataset.events,
                    llm_clients=llm_clients,
                    embedding_model=embedder,
                    dataset=dataset,
                )

                # Run with each weight preset
                for preset_name in weight_presets:
                    _write_progress(output_dir, {
                        "phase": "running",
                        "cell": f"qoe_optimised/{preset_name}/seed{seed}",
                        "completed_cells": completed_cells, "total_cells": total_cells,
                    })
                    weights = QOE_WEIGHT_PRESETS[preset_name]
                    assigner.weights = weights
                    assignment = assigner.assign(strategy="qoe_optimised")

                    metrics = _run_with_assignment(
                        dataset, clusters, assignment, llm_clients, embedder, seed,
                        perturbation=perturbation,
                    )
                    row = {
                        "strategy": "qoe_optimised",
                        "weight_preset": preset_name,
                        "backend": "mixed",
                        "seed": seed,
                        "dataset": dataset.short_name,
                        "calibration_fraction": calibration_fraction,
                        "perturbation_kind": (perturbation.kind if perturbation else "none"),
                        "assignment": str(assignment),
                        **metrics,
                    }
                    rows.append(row)
                    _append_row(row, csv_path)
                    completed_cells += 1
                    _write_progress(output_dir, {
                        "phase": "cell_done",
                        "cell": f"qoe_optimised/{preset_name}/seed{seed}",
                        "completed_cells": completed_cells, "total_cells": total_cells,
                        "last_f1": metrics["f1"], "last_cost": metrics["cost_per_1k"],
                        "last_latency_s": metrics.get("latency_s"),
                        "assignment": str(assignment),
                    })
                    logger.info(
                        f"  qoe_optimised/{preset_name} seed={seed}: "
                        f"F1={metrics['f1']:.4f}, cost={metrics['cost_per_1k']:.4f} "
                        f"({completed_cells}/{total_cells} cells)"
                    )

    _write_progress(output_dir, {
        "phase": "done", "completed_cells": completed_cells, "total_cells": total_cells,
    })

    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame(rows)


def run_qoe_stage(
    entry,
    profile: dict,
    llm_client: LLMClient | None = None,
    embedding_model: EmbeddingModel | None = None,
    progress_tracker=None,
) -> None:
    """Run a QoE entry from the manifest.

    Thin wrapper for integration with run_one.py.
    """
    dataset = load_dataset_by_name(entry.dataset, cache_dir="data")
    output_dir = Path(entry.result_file).parent

    qoe_cfg = profile.get("qoe", {})
    strategies = qoe_cfg.get("strategies", DEFAULT_STRATEGIES)
    weight_presets = qoe_cfg.get("weight_presets", DEFAULT_WEIGHT_PRESETS)
    seeds = qoe_cfg.get("seeds", DEFAULT_SEEDS)

    # Build LLM clients for each backend
    backends = qoe_cfg.get("backends", {})
    llm_clients = {}
    for name, model_id in backends.items():
        llm_clients[name] = LLMClient(model=model_id)

    if not llm_clients and llm_client is not None:
        llm_clients = {"default": llm_client}

    run_qoe_experiment(
        dataset=dataset,
        llm_clients=llm_clients,
        embedder=embedding_model,
        strategies=strategies,
        weight_presets=weight_presets,
        seeds=seeds,
        output_dir=output_dir,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="QoE heterogeneous backend experiment")
    parser.add_argument("--dataset", type=str, default="D1")
    parser.add_argument("--strategies", type=str, default=",".join(DEFAULT_STRATEGIES))
    parser.add_argument("--weight-presets", type=str, default=",".join(DEFAULT_WEIGHT_PRESETS))
    parser.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--backends", type=str,
                        default="qwen7b:ollama/qwen2.5:7b,haiku:anthropic/claude-3-haiku-20240307,sonnet:anthropic/claude-sonnet-4-20250514",
                        help="Comma-separated name:model_id pairs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--cache-dir", type=str, default="data")
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument(
        "--max-events", type=int, default=None,
        help="Truncate the dataset to this many events for L23 smoke runs. "
             "Default: full corpus.",
    )
    # H4: calibration-fraction flag (TAAS round-1 reviewer mandate).
    parser.add_argument(
        "--calibration-fraction", type=float, default=0.1,
        help="Fraction of events used for QoE calibration (default 0.1).",
    )
    # C9: perturbation hooks for the QoE adaptive loop.
    parser.add_argument(
        "--perturbation", type=str, default="none",
        choices=["none", "topic_restricted_cal", "latency_injection", "failure_injection"],
        help="C9 perturbation kind (default 'none' = no perturbation).",
    )
    parser.add_argument(
        "--topic-mask", type=str, default=None,
        help="topic_restricted_cal: comma-separated subscription IDs the calibration "
             "sample is restricted to (e.g., 's0,s1,s2').",
    )
    parser.add_argument(
        "--injection-event-index", type=int, default=None,
        help="latency_injection / failure_injection: 0-based event index at which "
             "the injection starts.",
    )
    parser.add_argument(
        "--injected-latency-s", type=float, default=None,
        help="latency_injection: extra wall-clock seconds added per LLM call "
             "for events with idx >= injection-event-index.",
    )
    parser.add_argument(
        "--backend-to-fail", type=str, default=None,
        help="failure_injection: name of the backend to mark as failed for "
             "events >= injection-event-index. Per L41, target ONE of N "
             "backends, not all.",
    )
    # L65: matched-pair LLM-call cache. When set, all LLMClient instances
    # share a single JSONL cache; baseline + perturbed runs that share the
    # same cache file see identical LLM responses for identical (model,
    # prompt) pairs, neutralising LLM-nondeterminism from the matched-cell
    # comparison.
    parser.add_argument(
        "--llm-cache", type=str, default=None,
        help="L65: path to a JSONL LLM-call cache shared across baseline and "
             "perturbed runs to neutralise LLM-nondeterminism in matched-pair "
             "analysis. Default: no cache (every call hits the API).",
    )
    return parser.parse_args()


def _build_perturbation_from_args(args) -> PerturbationSpec | None:
    """Construct a PerturbationSpec from argparse args, or return None for
    the no-op kind. Raises ValueError on malformed config (per L39)."""
    if args.perturbation == "none":
        return None
    kind = args.perturbation
    topic_mask = None
    if args.topic_mask:
        topic_mask = [t.strip() for t in args.topic_mask.split(",") if t.strip()]
    spec = PerturbationSpec(
        kind=kind,
        topic_mask=topic_mask,
        injection_event_index=args.injection_event_index,
        injected_latency_s=args.injected_latency_s,
        backend_to_fail=args.backend_to_fail,
    )
    return spec.validate()


def main():
    args = parse_args()
    load_dotenv(override=True)

    profile = None
    try:
        profile = load_profile(mode=args.mode)
    except (FileNotFoundError, ValueError):
        pass

    if profile and args.output_dir == "results":
        args.output_dir = str(output_dir_for_mode("results", profile["mode"], "qoe"))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset_by_name(args.dataset, cache_dir=args.cache_dir)
    if args.max_events is not None and args.max_events < len(dataset.events):
        logger.info("Truncating dataset to --max-events=%d (was %d)",
                    args.max_events, len(dataset.events))
        dataset.events = dataset.events[: args.max_events]
    logger.info(dataset.summary())

    embedder = EmbeddingModel("all-MiniLM-L6-v2", cache_dir=args.cache_dir)

    # Parse backends
    llm_clients = {}
    cache_path = args.llm_cache  # L65: shared cache path across all backends
    for pair in args.backends.split(","):
        name, model_id = pair.split(":", 1)
        if args.dry_run:
            llm_clients[name] = DryRunLLMClient(model=model_id)
        else:
            llm_clients[name] = LLMClient(model=model_id, cache_path=cache_path)

    strategies = [s.strip() for s in args.strategies.split(",")]
    weight_presets = [w.strip() for w in args.weight_presets.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    # C9: build the perturbation spec (or None for no-op).
    perturbation = _build_perturbation_from_args(args)
    if perturbation is not None:
        logger.info("C9 perturbation active: %s", perturbation)

    run_qoe_experiment(
        dataset=dataset,
        llm_clients=llm_clients,
        embedder=embedder,
        strategies=strategies,
        weight_presets=weight_presets,
        seeds=seeds,
        output_dir=output_dir,
        calibration_fraction=args.calibration_fraction,
        perturbation=perturbation,
    )


if __name__ == "__main__":
    main()
