"""Tests for scripts/run_cost_validation.py.

The cost-validation campaign re-runs a small representative grid of
(config, k) cells with the new per-cluster logging in place, so that
fig:cost-validation can plot honest predicted-vs-measured I per cluster
(figure data gap → fix the experiment, not the figure).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_run_cost_validation_help_lists_flags():
    """The cost-validation runner must support --dataset, --configs, --k,
    --seed, --output-dir, --max-events, --llm-model so the SLURM wrapper can
    parameterise it."""
    out = subprocess.run(
        [sys.executable, "scripts/run_cost_validation.py", "--help"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    h = out.stdout
    for flag in ("--dataset", "--configs", "--k-values", "--seed",
                 "--output-dir", "--max-events", "--llm-model"):
        assert flag in h, f"missing CLI flag {flag!r} in help: {h}"


def test_run_cost_validation_writes_per_cluster_csv(tmp_path):
    """End-to-end on a tiny in-memory dataset with the dry-run LLM client:
    the CSV must contain per_cluster_invocations, per_cluster_events,
    per_cluster_active_subs columns (JSON-encoded lists)."""
    out = subprocess.run(
        [sys.executable, "scripts/run_cost_validation.py",
         "--dataset", "D1",
         "--configs", "A0,A3",
         "--k-values", "1,3",
         "--seed", "42",
         "--max-events", "8",
         "--llm-model", "dry-run",
         "--output-dir", str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert out.returncode == 0, f"runner failed: stdout={out.stdout!r} stderr={out.stderr!r}"
    csv_path = tmp_path / "cost_validation_D1.csv"
    assert csv_path.exists(), f"expected CSV at {csv_path}"
    text = csv_path.read_text()
    header = text.splitlines()[0].split(",")
    for col in ("per_cluster_invocations", "per_cluster_events", "per_cluster_active_subs"):
        assert col in header, f"missing column {col} in {header}"
    # Exactly 4 data rows: 2 configs × 2 k-values × 1 seed
    data_rows = [r for r in text.splitlines()[1:] if r.strip()]
    assert len(data_rows) == 4, f"expected 4 data rows, got {len(data_rows)}: {data_rows}"
    import pandas as pd
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        per_inv = json.loads(row["per_cluster_invocations"])
        per_evt = json.loads(row["per_cluster_events"])
        per_sub = json.loads(row["per_cluster_active_subs"])
        assert len(per_inv) == len(per_evt) == len(per_sub), \
            f"per-cluster list lengths mismatch: {per_inv} {per_evt} {per_sub}"
        # Per-cluster counts MATCHING invocations only; for A0 (no CoverAndMerge)
        # this equals total llm_invocations, for A3 (with CoverAndMerge) it is
        # strictly less. Either way per_inv must sum to <= row["invocations"].
        assert 0 < sum(per_inv) <= row["invocations"], \
            f"matching-only sum should be in (0, total]: {sum(per_inv)} vs {row['invocations']}"
        if row["config"] == "A0":
            assert sum(per_inv) == row["invocations"], \
                f"A0 has no CoverAndMerge: matching sum should equal total"
        # Per-cluster events sum is at most the dataset event count
        assert sum(per_evt) > 0
        assert all(n > 0 for n in per_sub), f"each cluster has at least one sub: {per_sub}"
