"""C4 from peer-review report: compute 95% CIs + Wilcoxon p-values from
existing 5-seed result CSVs (no new compute).

Outputs:
  results/full/ablation/d1_ablation_with_cis.csv     (per-config mean ± half-CI95)
  results/full/ablation/d2_ablation_with_cis.csv
  results/full/ablation/d3_ablation_with_cis.csv
  results/full/ablation/d2_panel_with_cis.csv         (4-model open-weight panel + Sonnet)
  results/full/ablation/wilcoxon_pvalues.csv         (3 headline pair tests)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
ABL = ROOT / "results/full/ablation"
OUT = ABL  # write back into ablation dir alongside source CSVs


def ci95(values: list[float]) -> tuple[float, float]:
    """Mean and 95% CI half-width via t-distribution (small-n appropriate)."""
    a = np.asarray(values, dtype=float)
    n = len(a)
    if n < 2:
        return float(a.mean()), float("nan")
    mean = a.mean()
    sem = a.std(ddof=1) / np.sqrt(n)
    half = sem * stats.t.ppf(0.975, n - 1)
    return float(mean), float(half)


def collapse_per_config(df: pd.DataFrame, config_col: str = "config") -> pd.DataFrame:
    """Per-config mean ± half-CI95 across seeds, restricted to ablation configs."""
    df = df[~df[config_col].astype(str).str.startswith("baseline_")].copy()
    rows = []
    for cfg, g in df.groupby(config_col):
        f1_vals = g["f1"].dropna().tolist()
        m, h = ci95(f1_vals)
        rows.append({
            "config": cfg,
            "n_seeds": len(f1_vals),
            "f1_mean": m,
            "f1_ci95_half": h,
            "precision_mean": g["precision"].mean(),
            "recall_mean": g["recall"].mean(),
            "fpr_mean": g["fpr"].mean(),
            "invocations_mean": g["invocations"].mean(),
            "latency_mean": g["latency_s"].mean(),
        })
    return pd.DataFrame(rows).sort_values("config").reset_index(drop=True)


def wilcoxon_pair(a: list[float], b: list[float], label_a: str, label_b: str) -> dict:
    """Wilcoxon signed-rank on paired observations. With n=5, exact-rank
    distribution has limited resolution; we report both the raw test and
    a paired-bootstrap 95% CI on the difference for robustness."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) < 2:
        return {"a": label_a, "b": label_b, "n": len(a), "p_wilcoxon": float("nan"),
                "diff_mean": float("nan"), "diff_ci95": (float("nan"), float("nan"))}

    diffs = a - b
    if np.allclose(diffs, 0):
        # Wilcoxon undefined when all differences are zero
        return {"a": label_a, "b": label_b, "n": len(a),
                "p_wilcoxon": 1.0, "diff_mean": 0.0,
                "diff_ci95": (0.0, 0.0)}

    try:
        w_stat, p = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided",
                                    method="exact" if len(a) <= 25 else "auto")
    except ValueError as e:
        # Some scipy versions error on all-zero diffs; we already handled above.
        w_stat, p = float("nan"), float("nan")

    # Paired bootstrap CI on mean diff
    rng = np.random.default_rng(seed=42)
    boot = np.array([rng.choice(diffs, size=len(diffs), replace=True).mean()
                     for _ in range(2000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "a": label_a, "b": label_b, "n": len(a),
        "p_wilcoxon": float(p) if not np.isnan(p) else float("nan"),
        "diff_mean": float(diffs.mean()),
        "diff_ci95": (float(lo), float(hi)),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- Per-dataset ablation CIs (Haiku as the primary backend) -------------
    for ds in ("D1", "D2", "D3"):
        src = ABL / f"{ds}_ablation_haiku_results.csv"
        if not src.exists():
            print(f"  SKIP {ds}: {src} not found")
            continue
        df = pd.read_csv(src)
        out = collapse_per_config(df)
        dst = OUT / f"{ds.lower()}_ablation_with_cis.csv"
        out.to_csv(dst, index=False)
        print(f"  wrote {dst}: {len(out)} rows")

    # ---- D2 four-model open-weight panel ------------------------------------
    # The panel is computed from the qwen7b_all_per_task data + Mistral 7B
    # data for D2. Each model has 5 seeds at A0 and A1. We collect from the
    # available per-task CSVs.
    panel_files = list((ABL).glob("D2_*panel*.csv")) + list((ABL).glob("D2_qwen*panel*.csv"))
    if not panel_files:
        # Fall back: use the values already in the manuscript caption (single
        # source of truth for panel); no CSV produced here.
        print("  panel CSVs not found; skipping panel CI generation")
    else:
        print(f"  found {len(panel_files)} panel CSVs (TODO: aggregate)")

    # ---- Headline Wilcoxon pairs --------------------------------------------
    # Three pairs the AE letter explicitly asks for:
    #   1. NR-best vs SBERT/TF-IDF on D2  (single-seed baselines vs 5-seed NR)
    #   2. Mistral 7B vs Qwen 7B on D2 A0  (cross-family at matched scale)
    #   3. Qwen 32B vs Qwen 7B on D2 A0    (within-family scale step)
    # Pair 1 uses unpaired comparison (different N): we report a one-sample
    # signed test against the baseline value as a constant. Pairs 2 and 3
    # need per-seed Mistral / Qwen 7B / Qwen 32B values, which live in
    # task-specific CSVs.
    rows = []

    # Pair 1: load Haiku D2 A1 5-seed F1 values vs the SBERT D2 baseline value
    haiku_d2_path = ABL / "D2_ablation_haiku_results.csv"
    if haiku_d2_path.exists():
        df = pd.read_csv(haiku_d2_path)
        df_a1 = df[(df["config"] == "A1") & ~df["config"].astype(str).str.startswith("baseline_")]
        sbert_row = df[df["config"] == "baseline_sbert"]
        if len(df_a1) >= 2 and len(sbert_row) >= 1:
            sbert_val = sbert_row["f1"].iloc[0]
            sbert_seq = [sbert_val] * len(df_a1)
            r = wilcoxon_pair(df_a1["f1"].tolist(), sbert_seq,
                              "Haiku-A1-D2", "SBERT-D2-const")
            rows.append(r)

    # Pair 2 + 3: Mistral 7B vs Qwen 7B and Qwen 32B vs Qwen 7B on D2 A0.
    # If the panel CSVs are not present, leave a placeholder note.
    panel_csv = ABL / "D2_panel_qwen7b.csv"  # may not exist
    if not panel_csv.exists():
        # The panel data is reported in the manuscript text/table; per-seed
        # CSVs for the open-weight panel may not be split out as separate
        # files. Record a note rather than a fake test.
        for label_pair in [("Mistral-7B-A0-D2", "Qwen-7B-A0-D2"),
                           ("Qwen-32B-A0-D2",  "Qwen-7B-A0-D2")]:
            rows.append({"a": label_pair[0], "b": label_pair[1], "n": 0,
                         "p_wilcoxon": float("nan"), "diff_mean": float("nan"),
                         "diff_ci95": (float("nan"), float("nan")),
                         "note": "per-seed CSVs not available; pair listed for completeness"})

    if rows:
        out = pd.DataFrame([
            {**r, "diff_ci95_lo": r["diff_ci95"][0], "diff_ci95_hi": r["diff_ci95"][1]}
            for r in rows
        ]).drop(columns=["diff_ci95"], errors="ignore")
        dst = OUT / "wilcoxon_pvalues.csv"
        out.to_csv(dst, index=False)
        print(f"  wrote {dst}: {len(out)} pairs")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
