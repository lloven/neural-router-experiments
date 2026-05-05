"""Tests for scripts/run_baseline_zero_shot.py.

Validates that the BART-MNLI zero-shot CLI:
  * parses --dataset / --output / --max-events / --model arguments,
  * loads the chosen dataset (subsampled),
  * writes a CSV with the column shape used by the other baseline CSVs in
    results/full/ablation/,
  * produces exactly one data row labelled config="baseline_zero_shot".

The "slow" mark gates this on `pytest -m slow`; without the mark, tests
that download a HuggingFace model are skipped.
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.mark.slow
def test_zero_shot_runner_writes_csv_with_expected_columns(tmp_path):
    """End-to-end: run the CLI with a tiny model on 8 D1 events, expect one
    valid row written.
    """
    out = tmp_path / "D1_baseline_zero_shot_results.csv"
    cmd = [
        sys.executable, str(REPO / "scripts" / "run_baseline_zero_shot.py"),
        "--dataset", "D1",
        "--max-events", "8",
        "--output", str(out),
        # Smaller MNLI model to keep test runtime reasonable; the real run
        # uses facebook/bart-large-mnli.
        "--model", "valhalla/distilbart-mnli-12-1",
    ]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, (
        f"exit {proc.returncode}\n"
        f"stdout (last 1k):\n{proc.stdout[-1000:]}\n"
        f"stderr (last 1k):\n{proc.stderr[-1000:]}"
    )
    assert out.exists(), f"CSV not written at {out}"

    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1, f"expected exactly 1 row, got {len(rows)}"

    expected_cols = {
        "config", "dataset", "seed", "precision", "recall", "f1", "fpr",
        "invocations", "compression_ratio", "latency_s",
    }
    missing = expected_cols - rows[0].keys()
    assert not missing, f"missing columns: {missing}"

    assert rows[0]["config"] == "baseline_zero_shot"
    assert rows[0]["dataset"] == "D1"
    f1 = float(rows[0]["f1"])
    assert 0.0 <= f1 <= 1.0, f"F1 out of range: {f1}"


def test_runner_script_file_exists():
    """File-existence smoke check; doesn't run anything (cheap)."""
    assert (REPO / "scripts" / "run_baseline_zero_shot.py").exists(), \
        "scripts/run_baseline_zero_shot.py not created"
