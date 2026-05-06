"""Aggregator for the C9 perturbation full run + the H4 calibration-fraction
sweep. Emits a JSON summary keyed by (perturbation, strategy, backend) and
by (calibration_fraction, strategy), suitable for inlining into the
manuscript prose at §5.9.1 and §5.9.2.

Usage:
    python analysis/qoe_perturbation_summary.py
        --perturbation results/full/qoe_perturbation/by_task
        --calfrac      results/full/qoe_calfrac/by_task
        --out          results/summaries/qoe_round2.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def aggregate_perturbation(perturbation_dir: Path) -> dict:
    """Aggregate the 3-cell perturbation full into matched-cell deltas
    + per-(perturbation, strategy) F1 / latency means with 95% CIs."""
    cells = ["baseline", "topic_restricted", "latency_injection"]
    frames = []
    for cell in cells:
        csv = perturbation_dir / cell / "qoe_D1.csv"
        if not csv.exists():
            return {"status": "incomplete", "missing": str(csv)}
        df = pd.read_csv(csv)
        df["perturbation"] = cell
        frames.append(df)
    plot_df = pd.concat(frames, ignore_index=True)

    # Per (perturbation, strategy): mean F1, mean latency, 95% CI
    agg = plot_df.groupby(["perturbation", "strategy"]).agg(
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        latency_mean=("latency_s", "mean"),
        latency_std=("latency_s", "std"),
        n=("seed", "nunique"),
    ).reset_index()
    agg["f1_ci95"] = 1.96 * agg["f1_std"] / agg["n"].clip(lower=1).pow(0.5)
    agg["latency_ci95"] = 1.96 * agg["latency_std"] / agg["n"].clip(lower=1).pow(0.5)

    # Matched-cell latency deltas: latency_injection - baseline by (strategy, backend, seed)
    base = plot_df[plot_df["perturbation"] == "baseline"]
    lat = plot_df[plot_df["perturbation"] == "latency_injection"]
    key = ["strategy", "weight_preset", "backend", "seed"]
    joined = base.merge(lat, on=key, suffixes=("_base", "_lat"))
    joined["delta_latency"] = joined["latency_s_lat"] - joined["latency_s_base"]
    joined["delta_f1"] = joined["f1_lat"] - joined["f1_base"]

    delta_summary = joined.groupby(["strategy", "backend"]).agg(
        delta_latency_mean=("delta_latency", "mean"),
        delta_latency_std=("delta_latency", "std"),
        delta_f1_mean=("delta_f1", "mean"),
        n=("seed", "nunique"),
    ).reset_index()
    delta_summary["delta_latency_ci95"] = (
        1.96 * delta_summary["delta_latency_std"] / delta_summary["n"].clip(lower=1).pow(0.5)
    )

    # topic_restricted vs baseline F1 delta — workload-shift signal
    tr = plot_df[plot_df["perturbation"] == "topic_restricted"]
    tr_joined = base.merge(tr, on=key, suffixes=("_base", "_tr"))
    tr_joined["delta_f1_tr"] = tr_joined["f1_tr"] - tr_joined["f1_base"]
    tr_summary = tr_joined.groupby(["strategy"]).agg(
        delta_f1_mean=("delta_f1_tr", "mean"),
        delta_f1_std=("delta_f1_tr", "std"),
        n=("seed", "nunique"),
    ).reset_index()

    return {
        "status": "complete",
        "n_seeds": int(plot_df["seed"].nunique()),
        "n_rows": int(len(plot_df)),
        "per_cell_summary": agg.to_dict(orient="records"),
        "latency_injection_matched_delta": delta_summary.to_dict(orient="records"),
        "topic_restricted_f1_delta": tr_summary.to_dict(orient="records"),
    }


def aggregate_calfrac(calfrac_dir: Path) -> dict:
    """Aggregate the H4 calibration-fraction sweep into a summary keyed
    by (calibration_fraction, strategy) with mean F1 + 95% CI."""
    fraction_dirs = sorted([
        p for p in calfrac_dir.glob("frac_*") if (p / "qoe_D1.csv").exists()
    ])
    if not fraction_dirs:
        return {"status": "incomplete", "missing": str(calfrac_dir)}

    frames = []
    for d in fraction_dirs:
        df = pd.read_csv(d / "qoe_D1.csv")
        if "calibration_fraction" not in df.columns:
            return {"status": "incomplete",
                    "missing_column": f"calibration_fraction in {d}"}
        frames.append(df)
    plot_df = pd.concat(frames, ignore_index=True)

    summary = plot_df.groupby(["calibration_fraction", "strategy"]).agg(
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        latency_mean=("latency_s", "mean"),
        n=("seed", "nunique"),
    ).reset_index()
    summary["f1_ci95"] = 1.96 * summary["f1_std"] / summary["n"].clip(lower=1).pow(0.5)

    return {
        "status": "complete",
        "n_fractions": int(plot_df["calibration_fraction"].nunique()),
        "n_seeds": int(plot_df["seed"].nunique()),
        "per_cell_summary": summary.to_dict(orient="records"),
    }


def _main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--perturbation", type=Path,
                    default=Path("results/full/qoe_perturbation/by_task"))
    ap.add_argument("--calfrac", type=Path,
                    default=Path("results/full/qoe_calfrac/by_task"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/summaries/qoe_round2.json"))
    args = ap.parse_args(argv)

    out = {
        "perturbation": aggregate_perturbation(args.perturbation),
        "calfrac": aggregate_calfrac(args.calfrac),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=float))
    print(f"summary written to {args.out}")
    print(json.dumps({"perturbation_status": out["perturbation"]["status"],
                      "calfrac_status": out["calfrac"]["status"]}, indent=2))


if __name__ == "__main__":
    import sys
    _main(sys.argv[1:])
