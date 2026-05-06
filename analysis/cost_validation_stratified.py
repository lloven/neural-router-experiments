"""R6 (TAAS round-1, AE recommended): cost-model 17% miss stratified analysis.

The headline `n=81, 83% within factor-of-two band` claim conceals which
clusters miss. This module splits the misses by whether the cluster
operates in the trivial regime (m_c <= b_max(c) — model collapses to
identity, miss is uninformative) or the non-trivial regime
(m_c > b_max(c) — cost model is genuinely under test, miss matters).

Usage (CLI):
    python analysis/cost_validation_stratified.py \
        results/full/cost_validation/full/cost_validation_D1_qwen7b.csv \
        results/full/cost_validation/full/cost_validation_D1_dryrun.csv

Output: a one-paragraph summary suitable for §5.4, plus a CSV per-cell.
"""
from __future__ import annotations

import argparse
import ast
import math
import sys
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


def _coerce_list(value):
    """Per-cluster columns may be either Python lists (already parsed) or
    string repr like '[5, 50]'. Coerce both to a list of ints."""
    if isinstance(value, str):
        return list(ast.literal_eval(value))
    return list(value)


def _explode_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Explode per_cluster_* arrays into one row per (run, cluster)."""
    rows = []
    for _, row in df.iterrows():
        invs = _coerce_list(row["per_cluster_invocations"])
        ms = _coerce_list(row["per_cluster_events"])
        subs = _coerce_list(row["per_cluster_active_subs"])
        if not (len(invs) == len(ms) == len(subs)):
            raise ValueError(
                f"per_cluster_* length mismatch in row config={row.get('config')} "
                f"seed={row.get('seed')}: |inv|={len(invs)}, |m|={len(ms)}, |subs|={len(subs)}"
            )
        for c, (i_meas, m_c, s_c) in enumerate(zip(invs, ms, subs)):
            rows.append({
                "config": row.get("config"),
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "k": row.get("k"),
                "cluster_idx": c,
                "I_meas": int(i_meas),
                "m_c": int(m_c),
                "active_subs": int(s_c),
            })
    return pd.DataFrame(rows)


def _bmax(
    active_subs: int,
    *,
    W: int,
    t_inst: int,
    t_s: int,
    t_e: int,
    t_resp: int,
    derate: float = 0.8,
) -> int:
    """b_max(c) per eq:bmax of the manuscript §3.5.

    The `derate` factor (default 0.8) matches `analysis/make_figures.py:179`,
    which is the script that produces `figs/cost_validation.pdf`. It
    represents an operational margin for response-token slack and was the
    convention used to generate the manuscript's headline 83\\%-in-band number.
    """
    available = W - t_inst - active_subs * t_s - t_resp
    if available <= 0:
        return 0
    return max(1, math.floor(available * derate / max(t_e, 1)))


def stratify_misses(
    df: pd.DataFrame,
    *,
    W: int = 4096,
    t_inst: int = 200,
    t_s: int = 80,
    t_e: int = 30,
    t_resp: int = 500,
    band_factor: float = 2.0,
    derate: float = 0.8,
) -> dict:
    """Stratify the cost-model miss cells by trivial vs non-trivial regime.

    Args:
        df: cost-validation DataFrame (one row per (config, k, seed) with
            per_cluster_* list columns).
        W, t_inst, t_s, t_e, t_resp: cost-model production constants per §3.5.
        band_factor: the factor-of-two band — miss iff
            max(I_pred, I_meas) / min(I_pred, I_meas) > band_factor.

    Returns:
        dict with keys:
          n_total, n_miss
          n_total_trivial, n_total_nontrivial
          n_miss_trivial, n_miss_nontrivial
          miss_rate, miss_rate_trivial, miss_rate_nontrivial
          median_ratio_overall, median_ratio_trivial, median_ratio_nontrivial
    """
    cells = _explode_clusters(df)
    # Per fig:cost-validation caption: cells with I_meas=0 are excluded.
    cells = cells[cells["I_meas"] > 0].copy()

    cells["b_max"] = cells["active_subs"].apply(
        lambda s: _bmax(s, W=W, t_inst=t_inst, t_s=t_s, t_e=t_e, t_resp=t_resp,
                        derate=derate)
    )
    cells["I_pred"] = cells.apply(
        lambda r: math.ceil(r["m_c"] / r["b_max"]) if r["b_max"] > 0 else math.inf,
        axis=1,
    )
    cells["ratio"] = cells["I_pred"] / cells["I_meas"]
    # Miss = outside the factor-of-`band_factor` band
    cells["miss"] = cells.apply(
        lambda r: (max(r["I_pred"], r["I_meas"]) / max(min(r["I_pred"], r["I_meas"]), 1))
                  > band_factor,
        axis=1,
    )
    cells["nontrivial"] = cells["m_c"] > cells["b_max"]

    n_total = int(len(cells))
    n_miss = int(cells["miss"].sum())
    triv = cells[~cells["nontrivial"]]
    nontriv = cells[cells["nontrivial"]]
    summary = {
        "n_total": n_total,
        "n_miss": n_miss,
        "miss_rate": n_miss / n_total if n_total else 0.0,
        "n_total_trivial": int(len(triv)),
        "n_total_nontrivial": int(len(nontriv)),
        "n_miss_trivial": int(triv["miss"].sum()),
        "n_miss_nontrivial": int(nontriv["miss"].sum()),
        "miss_rate_trivial": (
            int(triv["miss"].sum()) / len(triv) if len(triv) else 0.0
        ),
        "miss_rate_nontrivial": (
            int(nontriv["miss"].sum()) / len(nontriv) if len(nontriv) else 0.0
        ),
        "median_ratio_overall": float(cells["ratio"].median()) if n_total else float("nan"),
        "median_ratio_trivial": float(triv["ratio"].median()) if len(triv) else float("nan"),
        "median_ratio_nontrivial": float(nontriv["ratio"].median()) if len(nontriv) else float("nan"),
        "_per_cell": cells,  # for downstream CSV export
    }
    return summary


def format_summary_paragraph(summary: dict) -> str:
    """One-paragraph English summary suitable for §5.4 of the manuscript."""
    s = summary
    return (
        f"Stratifying the {s['n_total']} per-cluster cost-validation cells "
        f"by whether the cluster operates in the regime where the model is "
        f"non-trivially under test ($m_c > b_{{\\max}}(c)$, i.e., the ceiling "
        f"in $I_{{\\mathrm{{pred}}}}=\\lceil m_c/b_{{\\max}}\\rceil$ is "
        f"non-degenerate): "
        f"{s['n_total_nontrivial']} cells are non-trivial regime "
        f"with miss rate {s['miss_rate_nontrivial']:.0%} (median ratio "
        f"{s['median_ratio_nontrivial']:.2f}); "
        f"{s['n_total_trivial']} cells are trivial regime "
        f"(miss rate {s['miss_rate_trivial']:.0%}, median ratio "
        f"{s['median_ratio_trivial']:.2f}). The {s['n_miss']} total misses "
        f"split as {s['n_miss_nontrivial']} non-trivial and "
        f"{s['n_miss_trivial']} trivial; the cost model's misses concentrate "
        f"in the {'non-trivial' if s['n_miss_nontrivial'] >= s['n_miss_trivial'] else 'trivial'}-regime "
        f"stratum, where {'the model is genuinely under test' if s['n_miss_nontrivial'] >= s['n_miss_trivial'] else 'the model collapses to the identity ceiling and the miss is uninformative'}."
    )


def _main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", help="cost-validation CSV files (rows are runs)")
    ap.add_argument("--out-cells", type=Path, default=None, help="optional per-cell CSV output")
    ap.add_argument("--W", type=int, default=4096)
    ap.add_argument("--t-inst", type=int, default=200)
    ap.add_argument("--t-s", type=int, default=80)
    ap.add_argument("--t-e", type=int, default=30)
    ap.add_argument("--t-resp", type=int, default=500)
    ap.add_argument("--derate", type=float, default=0.8,
                    help="response-token slack derate factor (default 0.8 to match make_figures.py)")
    args = ap.parse_args(argv)

    frames = [pd.read_csv(p) for p in args.inputs]
    df = pd.concat(frames, ignore_index=True)
    summary = stratify_misses(
        df,
        W=args.W,
        t_inst=args.t_inst,
        t_s=args.t_s,
        t_e=args.t_e,
        t_resp=args.t_resp,
        derate=args.derate,
    )
    cells = summary.pop("_per_cell")
    print("--- Stratified miss summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print()
    print("--- Paragraph for §5.4 ---")
    print(format_summary_paragraph({**summary, "_per_cell": cells}))

    if args.out_cells is not None:
        args.out_cells.parent.mkdir(parents=True, exist_ok=True)
        cells.to_csv(args.out_cells, index=False)
        print(f"\nPer-cell CSV: {args.out_cells}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
