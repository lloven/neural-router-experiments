"""Tests for QoE-optimised heterogeneous backend assignment (Phase 1d).

The QoE experiment validates that routing different subscription clusters
to different LLM backends (Qwen, Haiku, Sonnet) based on calibrated
quality scores achieves near-Sonnet F1 at significantly lower cost.

QoEAssigner:
  - Takes cluster objects + list of backends
  - Runs calibration (10% sample per cluster per backend)
  - Computes QoE scores with configurable weights
  - Returns per-cluster backend assignment

Three strategies:
  1. homogeneous -- one backend for all clusters
  2. round-robin -- cycle backends across clusters
  3. qoe-optimised -- assign via argmax QoE per cluster

Three weight configs: accuracy-first, balanced, cost-first

TDD RED phase: tests written before implementation.
"""

import pytest
import numpy as np

from src.data import Subscription, Event, Dataset
from src.router import Cluster


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cluster(cluster_id: int, n_subs: int = 5) -> Cluster:
    """Create a cluster with subscriptions for testing."""
    subs = [
        Subscription(id=f"sub_{cluster_id}_{i}", name=f"Sub {i}", description=f"Topic {i}")
        for i in range(n_subs)
    ]
    return Cluster(
        id=cluster_id,
        subscriptions=subs,
        centroid=np.random.randn(384),
    )


# ---------------------------------------------------------------------------
# 1. QoEAssigner creation and configuration
# ---------------------------------------------------------------------------

def test_qoe_assigner_creation():
    """QoEAssigner should accept clusters, backends, and weight config."""
    from src.qoe import QoEAssigner

    clusters = [_make_cluster(0), _make_cluster(1)]
    backends = ["ollama/qwen2.5:7b", "anthropic/claude-3-haiku-20240307", "anthropic/claude-sonnet-4-20250514"]
    weights = {"accuracy": 0.6, "cost": 0.2, "latency": 0.2}

    assigner = QoEAssigner(
        clusters=clusters,
        backends=backends,
        weights=weights,
    )
    assert assigner.clusters == clusters
    assert assigner.backends == backends
    assert assigner.weights == weights


def test_qoe_assigner_default_weights():
    """Default weights should be balanced (equal)."""
    from src.qoe import QoEAssigner

    assigner = QoEAssigner(
        clusters=[_make_cluster(0)],
        backends=["backend_a"],
    )
    assert "accuracy" in assigner.weights
    assert "cost" in assigner.weights
    assert "latency" in assigner.weights


# ---------------------------------------------------------------------------
# 2. QoE score computation
# ---------------------------------------------------------------------------

def test_compute_qoe_score_accuracy_first():
    """Accuracy-first weights should favor the highest-F1 backend."""
    from src.qoe import compute_qoe_score

    weights = {"accuracy": 0.8, "cost": 0.1, "latency": 0.1}
    # Backend A: high F1, high cost
    score_a = compute_qoe_score(f1=0.95, cost=10.0, latency=5.0, weights=weights)
    # Backend B: low F1, low cost
    score_b = compute_qoe_score(f1=0.60, cost=1.0, latency=1.0, weights=weights)
    assert score_a > score_b


def test_compute_qoe_score_cost_first():
    """Cost-first weights should favor the cheapest backend."""
    from src.qoe import compute_qoe_score

    weights = {"accuracy": 0.1, "cost": 0.8, "latency": 0.1}
    # Backend A: high F1, high cost
    score_a = compute_qoe_score(f1=0.95, cost=10.0, latency=5.0, weights=weights)
    # Backend B: low F1, low cost
    score_b = compute_qoe_score(f1=0.60, cost=0.1, latency=1.0, weights=weights)
    assert score_b > score_a


def test_compute_qoe_score_balanced():
    """Balanced weights should trade off accuracy and cost."""
    from src.qoe import compute_qoe_score

    weights = {"accuracy": 0.34, "cost": 0.33, "latency": 0.33}
    score = compute_qoe_score(f1=0.80, cost=2.0, latency=2.0, weights=weights)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# 3. Calibration results
# ---------------------------------------------------------------------------

def test_calibration_result_structure():
    """Calibration result should map (cluster_id, backend) to metrics."""
    from src.qoe import CalibrationResult

    result = CalibrationResult()
    result.add(cluster_id=0, backend="backend_a", f1=0.85, cost=2.0, latency=1.5)
    result.add(cluster_id=0, backend="backend_b", f1=0.70, cost=0.5, latency=0.3)

    metrics_a = result.get(cluster_id=0, backend="backend_a")
    assert metrics_a["f1"] == 0.85
    assert metrics_a["cost"] == 2.0
    assert metrics_a["latency"] == 1.5


# ---------------------------------------------------------------------------
# 4. Assignment strategies
# ---------------------------------------------------------------------------

def test_homogeneous_assignment():
    """Homogeneous strategy assigns the same backend to all clusters."""
    from src.qoe import assign_homogeneous

    clusters = [_make_cluster(i) for i in range(4)]
    assignment = assign_homogeneous(clusters, backend="backend_a")
    assert all(v == "backend_a" for v in assignment.values())
    assert len(assignment) == 4


def test_round_robin_assignment():
    """Round-robin strategy cycles backends across clusters."""
    from src.qoe import assign_round_robin

    clusters = [_make_cluster(i) for i in range(6)]
    backends = ["a", "b", "c"]
    assignment = assign_round_robin(clusters, backends=backends)
    assert len(assignment) == 6
    # Clusters 0,3 -> a; 1,4 -> b; 2,5 -> c
    assert assignment[0] == "a"
    assert assignment[1] == "b"
    assert assignment[2] == "c"
    assert assignment[3] == "a"


def test_qoe_optimised_assignment():
    """QoE-optimised strategy assigns the backend with highest QoE per cluster."""
    from src.qoe import CalibrationResult, assign_qoe_optimised

    cal = CalibrationResult()
    # Cluster 0: backend_a is much better on F1 (0.95 vs 0.50), enough to
    # overcome cost/latency disadvantage even with balanced weights
    cal.add(0, "backend_a", f1=0.95, cost=5.0, latency=2.0)
    cal.add(0, "backend_b", f1=0.50, cost=1.0, latency=0.5)
    # Cluster 1: backend_b is better (much cheaper, slightly lower F1)
    cal.add(1, "backend_a", f1=0.50, cost=5.0, latency=2.0)
    cal.add(1, "backend_b", f1=0.48, cost=0.1, latency=0.1)

    weights = {"accuracy": 0.34, "cost": 0.33, "latency": 0.33}
    assignment = assign_qoe_optimised(cal, weights=weights)

    assert assignment[0] == "backend_a"  # higher F1 wins with balanced weights
    assert assignment[1] == "backend_b"  # much cheaper, slightly lower F1


# ---------------------------------------------------------------------------
# 5. QoE weight presets
# ---------------------------------------------------------------------------

def test_qoe_weight_presets():
    """Weight presets should be accessible by name."""
    from src.qoe import QOE_WEIGHT_PRESETS

    assert "accuracy_first" in QOE_WEIGHT_PRESETS
    assert "balanced" in QOE_WEIGHT_PRESETS
    assert "cost_first" in QOE_WEIGHT_PRESETS

    for name, w in QOE_WEIGHT_PRESETS.items():
        total = sum(w.values())
        assert abs(total - 1.0) < 1e-6, f"Preset {name} weights should sum to 1.0"
