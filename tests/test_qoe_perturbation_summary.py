"""TDD for analysis/qoe_perturbation_summary.py — the aggregator that
produces the JSON the manuscript prose at §5.9.1 / §5.9.2 quotes from.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_aggregate_perturbation_incomplete_when_cell_missing(tmp_path):
    from analysis.qoe_perturbation_summary import aggregate_perturbation
    out = aggregate_perturbation(tmp_path)
    assert out["status"] == "incomplete"
    assert "missing" in out


def test_aggregate_calfrac_incomplete_when_no_frac_dirs(tmp_path):
    from analysis.qoe_perturbation_summary import aggregate_calfrac
    out = aggregate_calfrac(tmp_path)
    assert out["status"] == "incomplete"


def test_aggregate_perturbation_matched_delta(tmp_path):
    """Verify the matched-cell delta computation: tier_mid latency
    +2.5s in latency_injection vs baseline; non-trivial deltas for
    other backends ignored when not joined."""
    from analysis.qoe_perturbation_summary import aggregate_perturbation
    base_rows = [
        {"strategy": "homogeneous", "weight_preset": "n/a", "backend": "tier_mid",
         "seed": 42, "f1": 0.31, "cost_per_1k": 0.024, "latency_s": 12.7,
         "calibration_fraction": 0.1, "perturbation_kind": "none"},
        {"strategy": "homogeneous", "weight_preset": "n/a", "backend": "tier_mid",
         "seed": 123, "f1": 0.32, "cost_per_1k": 0.024, "latency_s": 12.5,
         "calibration_fraction": 0.1, "perturbation_kind": "none"},
    ]
    tr_rows = [
        {**r, "f1": r["f1"] - 0.05, "perturbation_kind": "topic_restricted_cal"}
        for r in base_rows
    ]
    lat_rows = [
        {**r, "latency_s": r["latency_s"] + 2.5, "perturbation_kind": "latency_injection"}
        for r in base_rows
    ]
    _write(tmp_path / "baseline/qoe_D1.csv", base_rows)
    _write(tmp_path / "topic_restricted/qoe_D1.csv", tr_rows)
    _write(tmp_path / "latency_injection/qoe_D1.csv", lat_rows)

    summary = aggregate_perturbation(tmp_path)
    assert summary["status"] == "complete"
    assert summary["n_seeds"] == 2
    # latency_injection delta should be ~2.5s for tier_mid homogeneous
    deltas = {
        (r["strategy"], r["backend"]): r["delta_latency_mean"]
        for r in summary["latency_injection_matched_delta"]
    }
    assert abs(deltas[("homogeneous", "tier_mid")] - 2.5) < 0.01

    # topic_restricted F1 delta should be approximately -0.05 for homogeneous
    tr_deltas = {r["strategy"]: r["delta_f1_mean"] for r in summary["topic_restricted_f1_delta"]}
    assert abs(tr_deltas["homogeneous"] - (-0.05)) < 0.001


def test_aggregate_calfrac_round_trips_fraction(tmp_path):
    """Per L53: the calibration_fraction values from the CSVs must
    appear in the summary keys with the exact float values."""
    from analysis.qoe_perturbation_summary import aggregate_calfrac
    for frac in (0.05, 0.10, 0.20, 0.50):
        rows = [
            {"strategy": "homogeneous", "weight_preset": "balanced",
             "backend": "tier_mid", "seed": s, "f1": 0.30 + frac * 0.1,
             "cost_per_1k": 0.02, "latency_s": 12.0,
             "calibration_fraction": frac, "perturbation_kind": "none"}
            for s in (42, 123, 456, 789, 0)
        ]
        _write(tmp_path / f"frac_{frac}/qoe_D1.csv", rows)
    summary = aggregate_calfrac(tmp_path)
    assert summary["status"] == "complete"
    assert summary["n_fractions"] == 4
    fractions = sorted({r["calibration_fraction"] for r in summary["per_cell_summary"]})
    assert fractions == [0.05, 0.10, 0.20, 0.50]
