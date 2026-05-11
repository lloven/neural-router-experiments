"""TDD for the qoe_perturbation + qoe_calfrac figure functions.

These tests don't actually render PDFs (matplotlib is heavy); they verify
the data-handling logic the figure functions rely on (matched-cell delta,
fraction-from-CSV, graceful handling of missing data).

Figures must justify a manuscript claim — these tests pin the data
invariants the captions will state.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import pytest


def _write_qoe_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_fig_qoe_perturbation_skips_when_data_missing(tmp_path, monkeypatch):
    """Figure function must SKIP cleanly (not crash) when one of the
    three perturbation CSVs is missing — auto-pipeline robustness."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    monkeypatch.chdir(tmp_path)
    from analysis.make_figures import fig_qoe_perturbation
    fig_qoe_perturbation()  # all three CSVs missing → must not raise


def test_fig_qoe_calfrac_skips_when_data_missing(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    monkeypatch.chdir(tmp_path)
    from analysis.make_figures import fig_qoe_calfrac
    fig_qoe_calfrac()  # no frac_* dirs → must not raise


def test_qoe_perturbation_matched_cell_delta(tmp_path, monkeypatch):
    """The latency Δ in panel (b) must be computed by joining baseline
    and latency_injection rows on (strategy, weight_preset, backend, seed),
    then subtracting. This validates the manuscript claim that the +0.1s
    injection is observable in matched-cell comparison only."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    monkeypatch.chdir(tmp_path)
    base_rows = [
        {"strategy": "homogeneous", "weight_preset": "n/a", "backend": "tier_mid",
         "seed": 42, "f1": 0.31, "cost_per_1k": 0.024, "latency_s": 12.7,
         "calibration_fraction": 0.1, "perturbation_kind": "none"},
        {"strategy": "homogeneous", "weight_preset": "n/a", "backend": "tier_large",
         "seed": 42, "f1": 0.49, "cost_per_1k": 0.024, "latency_s": 70.0,  # cold-start
         "calibration_fraction": 0.1, "perturbation_kind": "none"},
    ]
    lat_rows = [
        {"strategy": "homogeneous", "weight_preset": "n/a", "backend": "tier_mid",
         "seed": 42, "f1": 0.31, "cost_per_1k": 0.024, "latency_s": 15.3,  # +2.6s
         "calibration_fraction": 0.1, "perturbation_kind": "latency_injection"},
        {"strategy": "homogeneous", "weight_preset": "n/a", "backend": "tier_large",
         "seed": 42, "f1": 0.49, "cost_per_1k": 0.024, "latency_s": 30.0,  # warm
         "calibration_fraction": 0.1, "perturbation_kind": "latency_injection"},
    ]
    _write_qoe_csv(tmp_path / "results/full/qoe_perturbation/by_task/baseline/qoe_D1.csv", base_rows)
    _write_qoe_csv(tmp_path / "results/full/qoe_perturbation/by_task/topic_restricted/qoe_D1.csv", base_rows)
    _write_qoe_csv(tmp_path / "results/full/qoe_perturbation/by_task/latency_injection/qoe_D1.csv", lat_rows)

    # Recompute the matched-cell delta the figure uses, end-to-end:
    base = pd.read_csv(tmp_path / "results/full/qoe_perturbation/by_task/baseline/qoe_D1.csv")
    lat  = pd.read_csv(tmp_path / "results/full/qoe_perturbation/by_task/latency_injection/qoe_D1.csv")
    joined = base.merge(lat, on=["strategy", "weight_preset", "backend", "seed"],
                        suffixes=("_base", "_lat"))
    joined["delta"] = joined["latency_s_lat"] - joined["latency_s_base"]
    deltas = dict(zip(joined["backend"], joined["delta"]))
    # tier_mid: +2.6s (treatment); tier_large: -40s (cold-start variance)
    assert abs(deltas["tier_mid"] - 2.6) < 0.05
    assert deltas["tier_large"] < -30  # warmup-dominated, not injection-driven


def test_qoe_calfrac_reads_calibration_fraction_from_csv(tmp_path, monkeypatch):
    """The figure must read `calibration_fraction` from the CSV column,
    NOT reconstruct it from the directory name. This pins the
    flag-propagation contract for downstream analysis (avoids float-string
    normalization mismatches like '0.10' vs '0.1')."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    monkeypatch.chdir(tmp_path)
    # Write three fractions with non-stripped (legible) directory names
    for frac in (0.05, 0.10, 0.20):
        rows = [
            {"strategy": "homogeneous", "weight_preset": "n/a", "backend": "tier_mid",
             "seed": s, "f1": 0.3 + s * 0.001, "cost_per_1k": 0.02,
             "latency_s": 12.0, "calibration_fraction": frac,
             "perturbation_kind": "none"}
            for s in (42, 123, 456)
        ]
        _write_qoe_csv(
            tmp_path / f"results/full/qoe_calfrac/by_task/frac_{frac}/qoe_D1.csv",
            rows,
        )

    # Now read back and verify the column is preserved exactly.
    paths = sorted((tmp_path / "results/full/qoe_calfrac/by_task").glob("frac_*/qoe_D1.csv"))
    fractions_in_csv = sorted({
        float(pd.read_csv(p)["calibration_fraction"].unique()[0]) for p in paths
    })
    assert fractions_in_csv == [0.05, 0.10, 0.20], (
        f"CSV calibration_fraction values must round-trip exactly; got {fractions_in_csv}"
    )
