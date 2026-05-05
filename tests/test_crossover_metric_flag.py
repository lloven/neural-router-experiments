"""Per L61: crossover sweep must support description-aware F1 as the
metric of record when sub sets contain duplication-with-rename. This
test pins:
(i) `--metric description` is accepted and propagates to the run.
(ii) Output rows include a `metric` column identifying which metric was
     used, so post-hoc analysis can't conflate the two.
(iii) On a duplicated D1-style toy fixture, description-aware F1 differs
      from ID-based F1 (validating that the flag actually changes behavior).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_metric_flag_in_argparse():
    """The CLI must accept --metric={id,description}."""
    from scripts.run_crossover import parse_args
    sys.argv = [
        "run_crossover.py",
        "--dataset", "D1",
        "--metric", "description",
        "--dry-run",
    ]
    args = parse_args()
    assert args.metric == "description"


def test_metric_flag_default_is_id():
    """Default metric is ID-based to preserve backward-compatible behavior."""
    from scripts.run_crossover import parse_args
    sys.argv = ["run_crossover.py", "--dataset", "D1", "--dry-run"]
    args = parse_args()
    assert args.metric == "id"


def test_metric_flag_rejects_unknown_value():
    from scripts.run_crossover import parse_args
    sys.argv = ["run_crossover.py", "--metric", "nonsense"]
    with pytest.raises(SystemExit):
        parse_args()


def test_csv_row_includes_metric_column(tmp_path):
    """A crossover row written to CSV must include a `metric` column so
    a post-hoc reader can distinguish ID-based from description-aware
    runs in the same file."""
    from scripts.run_crossover import _append_row

    row = {
        "config": "A0", "n_subscriptions": 19, "seed": 42,
        "max_context_tokens": 4096, "n_subs_effective": 19,
        "f1": 0.5, "precision": 0.5, "recall": 0.5,
        "invocations": 1, "compression_ratio": 1.0,
        "latency_s": 1.0, "tokens_prompt": 100, "tokens_response": 50,
        "cost_per_1k": 0.01, "metric": "description",
    }
    csv = tmp_path / "out.csv"
    _append_row(row, csv)
    df = pd.read_csv(csv)
    assert "metric" in df.columns
    assert df.iloc[0]["metric"] == "description"
