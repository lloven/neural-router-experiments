#!/usr/bin/env python3
"""Run the BART-MNLI zero-shot classification baseline (Section 4.3, item 7).

Wraps `src.baselines.run_baseline("zero_shot", ...)` with a CLI suitable for
laptop-CPU execution. Output CSV matches the column shape of the other
baseline CSVs in `results/full/ablation/`, so downstream analysis and
manuscript-table population scripts treat the result row identically.

The MNLI model is configurable so tests can use a small distil model and
the manuscript run can use the full `facebook/bart-large-mnli`.

Usage:

    # Real manuscript run on D1 full corpus
    python scripts/run_baseline_zero_shot.py \\
        --dataset D1 \\
        --output results/full/ablation/D1_baseline_zero_shot_results.csv

    # Fast test-time run with a small distil model
    python scripts/run_baseline_zero_shot.py \\
        --dataset D1 --max-events 50 --output /tmp/d1.csv \\
        --model typeform/distilbart-mnli-12-1
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Allow running directly from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_dataset_by_name
from src.baselines import run_baseline
from src.evaluation import evaluate_matches


def main() -> int:
    parser = argparse.ArgumentParser(description="BART-MNLI zero-shot baseline runner")
    parser.add_argument(
        "--dataset", required=True,
        help="Dataset short name: D1, D2, or D3."
    )
    parser.add_argument(
        "--output", required=True,
        help="Output CSV path."
    )
    parser.add_argument(
        "--max-events", type=int, default=None,
        help="Cap event count (deterministic seed=42 subsample). "
             "Match the per-dataset cap used elsewhere in the campaign."
    )
    parser.add_argument(
        "--model", default="valhalla/distilbart-mnli-12-1",
        help="HuggingFace MNLI model id. Default DistilBART-MNLI is small "
             "enough to run on a 16-32 GB CPU without OOM; pass "
             "facebook/bart-large-mnli on a machine with more RAM."
    )
    parser.add_argument(
        "--cache-dir", default="data",
        help="Dataset cache directory."
    )
    parser.add_argument(
        "--kappa", type=int, default=3,
        help="Top-kappa cut-off, matching the rest of the manuscript."
    )
    args = parser.parse_args()

    ds = load_dataset_by_name(
        args.dataset, cache_dir=args.cache_dir, max_events=args.max_events,
    )

    bres = run_baseline(
        method="zero_shot",
        dataset=ds,
        kappa=args.kappa,
        model_name=args.model,
    )
    eres = evaluate_matches(
        matches=bres.matches,
        dataset=ds,
        config_name="baseline_zero_shot",
    )
    eres.latency_s = bres.latency_s

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "config", "dataset", "seed",
        "precision", "recall", "f1", "fpr",
        "invocations", "compression_ratio", "latency_s",
        "tokens_prompt", "tokens_response", "cost_per_1k",
    ]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({
            "config": "baseline_zero_shot",
            "dataset": ds.short_name,
            "seed": 0,
            "precision": eres.precision.mean,
            "recall": eres.recall.mean,
            "f1": eres.f1.mean,
            "fpr": eres.fpr.mean,
            "invocations": 0,
            "compression_ratio": "n/a",
            "latency_s": eres.latency_s,
            "tokens_prompt": 0,
            "tokens_response": 0,
            "cost_per_1k": 0,
        })

    print(
        f"done D={ds.short_name} "
        f"F1={eres.f1.mean:.4f} P={eres.precision.mean:.4f} "
        f"R={eres.recall.mean:.4f} latency={eres.latency_s:.1f}s -> {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
