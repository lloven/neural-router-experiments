#!/usr/bin/env python3
"""One-time backfill: reconstruct ablation CSVs from manifest metrics.

The orchestrator (run_one.py) previously stored metrics in the manifest
but did not flush them to CSV.  This script reads the manifest and writes
the expected CSV files so that migrate_existing_results() and analysis
notebooks work correctly.

Only ablation runs are affected (sensitivity/scaling stages already wrote
their own CSVs).

Usage::

    python scripts/backfill_csvs.py                        # dry-run
    python scripts/backfill_csvs.py --apply                # write files
    python scripts/backfill_csvs.py --manifest results/full/manifest.json --apply
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.manifest import Manifest

# CSV columns matching EvaluationResult.summary_row()
CSV_COLUMNS = [
    "config", "dataset", "seed",
    "precision", "recall", "f1", "fpr",
    "invocations", "compression_ratio", "latency_s",
    "tokens_prompt", "tokens_response", "cost_per_1k",
]


def backfill(manifest_path: Path, apply: bool = False) -> dict[str, int]:
    """Read manifest and write/report missing ablation CSVs.

    Returns dict with counts: written, skipped_no_metrics, skipped_exists, skipped_non_ablation.
    """
    manifest = Manifest.load(manifest_path)
    counts = defaultdict(int)

    # Group done ablation runs by result_file path
    by_file: dict[str, list] = defaultdict(list)
    for entry in manifest.runs.values():
        if entry.status != "done":
            continue
        if entry.stage != "ablation":
            counts["skipped_non_ablation"] += 1
            continue
        if not entry.metrics:
            counts["skipped_no_metrics"] += 1
            continue
        by_file[entry.result_file].append(entry)

    for result_file, entries in sorted(by_file.items()):
        path = Path(result_file)

        # Check if any rows need writing
        existing_keys = set()
        if path.exists():
            with open(path) as f:
                for row in csv.DictReader(f):
                    existing_keys.add((row.get("config", ""), row.get("seed", "")))

        new_entries = [
            e for e in entries
            if (e.config, str(e.seed)) not in existing_keys
        ]

        if not new_entries:
            counts["skipped_exists"] += len(entries)
            continue

        if apply:
            path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not path.exists()
            with open(path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                if write_header:
                    writer.writeheader()
                for e in new_entries:
                    m = e.metrics
                    writer.writerow({
                        "config": e.config,
                        "dataset": e.dataset,
                        "seed": e.seed,
                        "precision": m.get("precision", ""),
                        "recall": m.get("recall", ""),
                        "f1": m.get("f1", ""),
                        "fpr": m.get("fpr", ""),
                        "invocations": m.get("invocations", ""),
                        "compression_ratio": m.get("compression_ratio", ""),
                        "latency_s": m.get("latency_s", ""),
                        "tokens_prompt": m.get("tokens_prompt", ""),
                        "tokens_response": m.get("tokens_response", ""),
                        "cost_per_1k": m.get("cost_per_1k", ""),
                    })
            print(f"  WROTE {len(new_entries)} rows -> {result_file}")
        else:
            print(f"  WOULD WRITE {len(new_entries)} rows -> {result_file}")

        counts["written"] += len(new_entries)

    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description="Backfill ablation CSVs from manifest metrics")
    parser.add_argument("--manifest", default="results/full/manifest.json",
                        help="Path to manifest.json")
    parser.add_argument("--apply", action="store_true",
                        help="Actually write files (default: dry-run)")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        sys.exit(1)

    print(f"{'APPLYING' if args.apply else 'DRY RUN'}: backfilling from {manifest_path}")
    counts = backfill(manifest_path, apply=args.apply)
    print(f"\nSummary: {counts}")


if __name__ == "__main__":
    main()
