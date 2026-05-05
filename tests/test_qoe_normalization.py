"""QoE normalization regression tests.

Catches the saturation bug found in 6620413 (cancelled): the original
implementation hardcoded `cost_max=20.0` USD/1k and `latency_max=30.0` s.
For self-hosted Qwen tiers, cost/1k is near-zero (cost_score ≈ 1.0 always)
and latency varies on a small range that is either both clipped to zero
or both saturated. Both terms therefore become CONSTANT across backends,
leaving only F1 to discriminate. Result: all three weight presets
(accuracy_first/balanced/cost_first) produce identical assignments.

The fixed implementation must per-cluster min-max normalise each metric
across the candidate backends, so the weight differences actually flip
the argmax when F1 and cost (or F1 and latency) disagree.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.qoe import (
    CalibrationResult,
    QOE_WEIGHT_PRESETS,
    assign_qoe_optimised,
)


def _two_backend_disagreeing_calibration() -> CalibrationResult:
    """Build a 1-cluster calibration where F1 ranks backend_b above
    backend_a but cost ranks backend_a below backend_b. A correct
    QoE-optimised assignment should pick backend_b under accuracy_first
    and backend_a under cost_first.
    """
    cal = CalibrationResult()
    cal.add(cluster_id=0, backend="cheap_low_f1", f1=0.4, cost=0.001, latency=2.0)
    cal.add(cluster_id=0, backend="expensive_high_f1", f1=0.7, cost=0.05, latency=10.0)
    return cal


def test_weight_presets_actually_differ_in_value():
    """Sanity: the three weight presets are not identical dictionaries."""
    af = QOE_WEIGHT_PRESETS["accuracy_first"]
    bal = QOE_WEIGHT_PRESETS["balanced"]
    cf = QOE_WEIGHT_PRESETS["cost_first"]
    assert af != bal
    assert bal != cf
    assert af != cf


def test_accuracy_first_picks_higher_f1_backend():
    cal = _two_backend_disagreeing_calibration()
    assignment = assign_qoe_optimised(cal, QOE_WEIGHT_PRESETS["accuracy_first"])
    assert assignment[0] == "expensive_high_f1", (
        f"accuracy_first should pick the higher-F1 backend; got {assignment[0]}"
    )


def test_cost_first_picks_lower_cost_backend():
    cal = _two_backend_disagreeing_calibration()
    assignment = assign_qoe_optimised(cal, QOE_WEIGHT_PRESETS["cost_first"])
    assert assignment[0] == "cheap_low_f1", (
        f"cost_first should pick the lower-cost backend; got {assignment[0]} "
        "— this is the saturation bug if it returns the high-F1 backend"
    )


def test_three_presets_can_yield_different_assignments():
    """Aggregate guarantee: at least two presets must differ on a
    calibration where F1 and cost disagree. (The original buggy
    implementation produced identical assignments across all three.)"""
    cal = _two_backend_disagreeing_calibration()
    a = assign_qoe_optimised(cal, QOE_WEIGHT_PRESETS["accuracy_first"])
    b = assign_qoe_optimised(cal, QOE_WEIGHT_PRESETS["balanced"])
    c = assign_qoe_optimised(cal, QOE_WEIGHT_PRESETS["cost_first"])
    assert not (a == b == c), (
        f"All three weight presets gave the same assignment {a}. "
        "This is the saturation bug from job 6620413."
    )


def test_cost_term_in_realistic_qwen_range_is_not_saturated():
    """When costs lie in a realistic self-hosted-Qwen range
    ($0.01-$0.05 per 1k events), the cost_first preset must still
    favor the cheaper backend over an expensive one with marginally
    higher F1. The original `cost_max=20.0` constant flattened this
    discrimination."""
    cal = CalibrationResult()
    # Both costs WAY below the original cost_max=20.0 — both saturate to ~1.0
    # under the buggy implementation, so cost weight has no effect.
    cal.add(cluster_id=0, backend="qwen_7b", f1=0.50, cost=0.018, latency=12.0)
    cal.add(cluster_id=0, backend="qwen_32b", f1=0.55, cost=0.035, latency=42.0)
    assignment = assign_qoe_optimised(cal, QOE_WEIGHT_PRESETS["cost_first"])
    assert assignment[0] == "qwen_7b", (
        f"cost_first must prefer qwen_7b (cheaper) over qwen_32b "
        f"(slightly higher F1 but 2x cost) on realistic costs; got {assignment[0]}. "
        "The hardcoded cost_max=20.0 saturated the cost term and made cost_first "
        "indistinguishable from accuracy_first in 6620413."
    )


def test_latency_first_term_in_realistic_range_is_not_saturated():
    """Same shape as cost test but for latency: at realistic latencies
    (~10-50s for the calibration sub-run), the original latency_max=30.0
    either clipped both backends to 0 or made them identical to ~1.
    Per-cluster min-max normalisation must keep the latency term
    discriminative."""
    cal = CalibrationResult()
    cal.add(cluster_id=0, backend="fast_low_f1", f1=0.45, cost=0.02, latency=5.0)
    cal.add(cluster_id=0, backend="slow_high_f1", f1=0.50, cost=0.02, latency=45.0)
    # latency_first preset doesn't exist as a name; build it manually
    weights_latency_first = {"accuracy": 0.15, "cost": 0.15, "latency": 0.7}
    assignment = assign_qoe_optimised(cal, weights_latency_first)
    assert assignment[0] == "fast_low_f1", (
        f"latency-prioritising weights must pick the fast backend (5s) over "
        f"the 9x slower one (45s); got {assignment[0]}. The hardcoded "
        "latency_max=30.0 made both backends saturate the latency term."
    )
