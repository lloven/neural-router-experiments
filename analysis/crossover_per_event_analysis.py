"""Per-event analysis of the description-aware crossover sweep — Tier 1
diagnostic for the §5.7 mechanism. Uses the per-event parquet logs from
6620172 to characterise A4's failure mode at high |S|.

The finding (verified in the parquets): A4 at |S|=2000 picks predictions
from a much narrower distinct vocabulary than at |S|=50, and the empty-
prediction rate grows from 31% to 54%. Merge-factor (compound IDs joined
by '+') is rarely picked at all (38 out of ~7700 at |S|=50, 0 at
|S|=2000), so the failure is *vocabulary collapse / refusal*, not
mid-prediction compound-description disambiguation.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
BY_TASK = ROOT / "results" / "full" / "crossover_desc" / "by_task"
OUT = ROOT.parent.parent / "Manuscripts" / "Neural Router (Elsevier FGCS)" / "figs"

S_VOLS = [50, 200, 2000]
EFF = {50: 30, 200: 125, 2000: 133}  # |S|_eff after CoverAndMerge from CSV

def parse_pred_ids(s: str) -> list[str]:
    """Split predicted_ids '[a]|[b]|[c+d]' style into individual atomic IDs."""
    if not s: return []
    out = []
    for piece in s.replace("+", "|").split("|"):
        piece = piece.strip().strip("[").strip("]").strip()
        if piece:
            out.append(piece)
    return out

def per_cell_stats(n: int, cfg: str) -> dict:
    p = BY_TASK / f"qwen7b_cross_desc_S{n}_{cfg}_s42" / f"per_event_{cfg}_S{n}_s42.parquet"
    df = pd.read_parquet(p)
    n_events = len(df)
    n_empty = (df["predicted_ids"].fillna("").str.len() == 0).sum()
    parsed = df["predicted_ids"].fillna("").apply(parse_pred_ids)
    distinct = set()
    for ids in parsed:
        distinct.update(ids)
    n_with_merged = df["predicted_ids"].str.contains(r"\+", na=False).sum()
    return {
        "n": n,
        "config": cfg,
        "n_events": n_events,
        "empty_rate": n_empty / n_events,
        "distinct_pred_ids": len(distinct),
        "S_eff": EFF[n] if cfg == "A4" else min(n, 162),
        "vocab_used_frac": len(distinct) / (EFF[n] if cfg == "A4" else min(n, 162)),
        "n_compound_predictions": n_with_merged,
    }

def main():
    rows = []
    for n in S_VOLS:
        for cfg in ["A0", "A4"]:
            rows.append(per_cell_stats(n, cfg))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print()

    # Two-panel figure: empty rate, distinct vocab fraction
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharex=True)
    for cfg, color, marker in [("A0", "C0", "o"), ("A4", "C1", "s")]:
        sub = df[df["config"] == cfg].sort_values("n")
        axes[0].plot(sub["n"], sub["empty_rate"] * 100, marker=marker, color=color, label=cfg)
        axes[1].plot(sub["n"], sub["vocab_used_frac"] * 100, marker=marker, color=color, label=cfg)
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].set_xlabel("|S|")
    axes[1].set_xlabel("|S|")
    axes[0].set_ylabel("Empty-prediction rate (%)")
    axes[1].set_ylabel("Distinct predicted IDs / |S|_eff (%)")
    axes[0].set_title("(a) Refusal rate")
    axes[1].set_title("(b) Vocabulary coverage")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    out = OUT / "crossover_mechanism.pdf"
    fig.savefig(out)
    print(f"saved {out}")

if __name__ == "__main__":
    main()
