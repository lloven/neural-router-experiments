"""Tests for the QoE heterogeneous backend experiment script (Phase 1d).

Tests the run_qoe.py experiment script which orchestrates:
  - Three strategies: homogeneous, round-robin, QoE-optimised
  - Three weight configs: accuracy-first, balanced, cost-first
  - Multiple datasets and seeds

TDD RED phase: tests written before implementation.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data import Subscription, Event, Dataset
from src.llm import DryRunLLMClient
from src.embeddings import EmbeddingModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_dataset(n_subs: int = 10, n_events: int = 5) -> Dataset:
    subs = [
        Subscription(id=f"sub_{i}", name=f"Sub {i}", description=f"Topic about subject {i}")
        for i in range(n_subs)
    ]
    events = [
        Event(id=f"evt_{i}", text=f"Event text number {i}", ground_truth=["sub_0"])
        for i in range(n_events)
    ]
    return Dataset(name="Test", short_name="T1", events=events, subscriptions=subs)


# ---------------------------------------------------------------------------
# 1. run_qoe_experiment returns DataFrame with expected structure
# ---------------------------------------------------------------------------

def test_run_qoe_experiment_basic_structure():
    """run_qoe_experiment should return a DataFrame with strategy/weight columns."""
    from scripts.run_qoe import run_qoe_experiment

    dataset = _make_test_dataset()
    backends = {"dry_a": DryRunLLMClient(model="model-a"),
                "dry_b": DryRunLLMClient(model="model-b")}
    embedder = EmbeddingModel("all-MiniLM-L6-v2")

    with tempfile.TemporaryDirectory() as tmpdir:
        df = run_qoe_experiment(
            dataset=dataset,
            llm_clients=backends,
            embedder=embedder,
            strategies=["homogeneous"],
            weight_presets=["balanced"],
            seeds=[42],
            output_dir=Path(tmpdir),
        )

    assert isinstance(df, pd.DataFrame)
    assert "strategy" in df.columns
    assert "weight_preset" in df.columns
    assert "seed" in df.columns
    assert "f1" in df.columns
    assert "cost_per_1k" in df.columns
    assert len(df) >= 1


# ---------------------------------------------------------------------------
# 2. Homogeneous strategy runs for each backend
# ---------------------------------------------------------------------------

def test_qoe_homogeneous_one_row_per_backend():
    """Homogeneous strategy should produce one row per (backend, seed)."""
    from scripts.run_qoe import run_qoe_experiment

    dataset = _make_test_dataset()
    backends = {"dry_a": DryRunLLMClient(model="model-a"),
                "dry_b": DryRunLLMClient(model="model-b")}
    embedder = EmbeddingModel("all-MiniLM-L6-v2")

    with tempfile.TemporaryDirectory() as tmpdir:
        df = run_qoe_experiment(
            dataset=dataset,
            llm_clients=backends,
            embedder=embedder,
            strategies=["homogeneous"],
            weight_presets=["balanced"],  # weight irrelevant for homogeneous
            seeds=[42],
            output_dir=Path(tmpdir),
        )

    # Should have one row per backend (2 backends * 1 seed)
    homogeneous_rows = df[df["strategy"] == "homogeneous"]
    assert len(homogeneous_rows) == 2


# ---------------------------------------------------------------------------
# 3. CSV output
# ---------------------------------------------------------------------------

def test_qoe_experiment_writes_csv():
    """QoE experiment should write results to CSV."""
    from scripts.run_qoe import run_qoe_experiment

    dataset = _make_test_dataset()
    backends = {"dry_a": DryRunLLMClient(model="model-a")}
    embedder = EmbeddingModel("all-MiniLM-L6-v2")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        run_qoe_experiment(
            dataset=dataset,
            llm_clients=backends,
            embedder=embedder,
            strategies=["homogeneous"],
            weight_presets=["balanced"],
            seeds=[42],
            output_dir=output_dir,
        )

        csv_path = output_dir / "qoe_T1.csv"
        assert csv_path.exists()
