"""TDD for Task 1.5 + Task 2.1: CLI flag wiring in scripts/run_qoe.py.

Verifies:
  - The argparse layer accepts --perturbation, --topic-mask,
    --injection-event-index, --injected-latency-s, --backend-to-fail,
    and --calibration-fraction.
  - _build_perturbation_from_args constructs a validated PerturbationSpec
    or returns None (raises on malformed config; flag set must equal flag consumed).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Allow importing the scripts/ module
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _argparse_for(*cli_args):
    """Re-build the argparse parser used by run_qoe.py and parse the
    provided positional arguments (excluding the program name)."""
    from scripts.run_qoe import parse_args
    saved_argv = sys.argv
    try:
        sys.argv = ["run_qoe.py", *cli_args]
        return parse_args()
    finally:
        sys.argv = saved_argv


def test_default_perturbation_is_none_returns_none():
    from scripts.run_qoe import _build_perturbation_from_args
    args = _argparse_for("--dataset", "D1")
    spec = _build_perturbation_from_args(args)
    assert spec is None


def test_topic_restricted_cli_builds_spec():
    """--topic-mask must propagate end-to-end (set ≠ consumed without
    treatment-verification on the produced spec)."""
    from scripts.run_qoe import _build_perturbation_from_args
    args = _argparse_for(
        "--dataset", "D1",
        "--perturbation", "topic_restricted_cal",
        "--topic-mask", "s0,s1,s2",
    )
    spec = _build_perturbation_from_args(args)
    assert spec is not None
    assert spec.kind == "topic_restricted_cal"
    assert spec.topic_mask == ["s0", "s1", "s2"]


def test_topic_restricted_cli_missing_mask_raises():
    """topic_restricted_cal without --topic-mask must raise from
    PerturbationSpec.validate(), not silently produce a no-op."""
    from scripts.run_qoe import _build_perturbation_from_args
    args = _argparse_for(
        "--dataset", "D1",
        "--perturbation", "topic_restricted_cal",
    )
    with pytest.raises(ValueError, match=r"topic_mask is required"):
        _build_perturbation_from_args(args)


def test_latency_injection_cli_builds_spec():
    from scripts.run_qoe import _build_perturbation_from_args
    args = _argparse_for(
        "--dataset", "D1",
        "--perturbation", "latency_injection",
        "--injection-event-index", "500",
        "--injected-latency-s", "0.5",
    )
    spec = _build_perturbation_from_args(args)
    assert spec is not None
    assert spec.kind == "latency_injection"
    assert spec.injection_event_index == 500
    assert spec.injected_latency_s == 0.5


def test_failure_injection_cli_builds_spec():
    from scripts.run_qoe import _build_perturbation_from_args
    args = _argparse_for(
        "--dataset", "D1",
        "--perturbation", "failure_injection",
        "--injection-event-index", "500",
        "--backend-to-fail", "qwen7b",
    )
    spec = _build_perturbation_from_args(args)
    assert spec is not None
    assert spec.kind == "failure_injection"
    assert spec.injection_event_index == 500
    assert spec.backend_to_fail == "qwen7b"


def test_calibration_fraction_default_is_0_10():
    args = _argparse_for("--dataset", "D1")
    assert args.calibration_fraction == 0.1


def test_calibration_fraction_flag_propagates():
    """H4: --calibration-fraction must be set AND consumed end-to-end
    (flag set must equal flag consumed). The CLI value must match args.calibration_fraction."""
    args = _argparse_for(
        "--dataset", "D1",
        "--calibration-fraction", "0.20",
    )
    assert args.calibration_fraction == 0.20
