"""Tests for subscription-count scaling analysis (Phase 1e enhancement).

The scaling experiment generates subscription sets of varying sizes
by subsampling/duplicating from D1, then runs A0 and A4 at each scale.

Design:
  - Subscription counts: [50, 100, 200, 500, 1000, 2000, 5000]
  - Configs: A0 (raw) and A4 (full pipeline)
  - Backend: Qwen (local)
  - Metrics: F1, invocations, latency, memory

The existing run_scaling.py handles event scaling. This test covers
the new scale_subscription_count() function that scales |S|.

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

def _make_test_dataset(n_subs: int = 19, n_events: int = 10) -> Dataset:
    subs = [
        Subscription(id=f"sub_{i}", name=f"Sub {i}", description=f"Topic about subject number {i}")
        for i in range(n_subs)
    ]
    events = [
        Event(id=f"evt_{i}", text=f"Event text number {i}", ground_truth=["sub_0"])
        for i in range(n_events)
    ]
    return Dataset(name="Test", short_name="T1", events=events, subscriptions=subs)


# ---------------------------------------------------------------------------
# 1. scale_subscription_count function exists and returns correct structure
# ---------------------------------------------------------------------------

def test_scale_subscription_count_returns_dataframe():
    """scale_subscription_count should return a DataFrame with expected columns."""
    from scripts.run_scaling import scale_subscription_count

    dataset = _make_test_dataset(n_subs=19, n_events=5)
    llm = DryRunLLMClient(model="test-model")
    embedder = EmbeddingModel("all-MiniLM-L6-v2")

    with tempfile.TemporaryDirectory() as tmpdir:
        df = scale_subscription_count(
            dataset=dataset,
            embedder=embedder,
            llm_client=llm,
            seed=42,
            output_dir=Path(tmpdir),
            sub_counts=[5, 10, 19],
            configs=["A0", "A4"],
        )

    assert isinstance(df, pd.DataFrame)
    assert "n_subscriptions" in df.columns
    assert "config" in df.columns
    assert "f1" in df.columns
    assert "invocations" in df.columns
    assert "latency_s" in df.columns


def test_scale_subscription_count_produces_correct_rows():
    """Should produce one row per (config, sub_count) pair."""
    from scripts.run_scaling import scale_subscription_count

    dataset = _make_test_dataset(n_subs=19, n_events=3)
    llm = DryRunLLMClient(model="test-model")
    embedder = EmbeddingModel("all-MiniLM-L6-v2")

    sub_counts = [5, 10]
    configs = ["A0", "A4"]

    with tempfile.TemporaryDirectory() as tmpdir:
        df = scale_subscription_count(
            dataset=dataset,
            embedder=embedder,
            llm_client=llm,
            seed=42,
            output_dir=Path(tmpdir),
            sub_counts=sub_counts,
            configs=configs,
        )

    # 2 configs * 2 sub_counts = 4 rows
    assert len(df) == 4


# ---------------------------------------------------------------------------
# 2. Subscription duplication for counts > dataset size
# ---------------------------------------------------------------------------

def test_scale_subscription_count_duplicates_when_needed():
    """When requested count exceeds dataset, subscriptions should be duplicated."""
    from scripts.run_scaling import generate_scaled_subscriptions

    subs = [
        Subscription(id=f"sub_{i}", name=f"Sub {i}", description=f"Topic {i}")
        for i in range(5)
    ]

    # Request 12 subscriptions from a set of 5 -- should duplicate
    result = generate_scaled_subscriptions(subs, target_count=12, seed=42)
    assert len(result) == 12
    # All IDs should be unique (duplicated subs get suffixed IDs)
    ids = [s.id for s in result]
    assert len(set(ids)) == 12


def test_scale_subscription_count_subsamples_when_smaller():
    """When requested count is smaller than dataset, subscriptions should be subsampled."""
    from scripts.run_scaling import generate_scaled_subscriptions

    subs = [
        Subscription(id=f"sub_{i}", name=f"Sub {i}", description=f"Topic {i}")
        for i in range(20)
    ]

    result = generate_scaled_subscriptions(subs, target_count=10, seed=42)
    assert len(result) == 10
    # All should be from original set
    original_ids = {s.id for s in subs}
    for s in result:
        assert s.id in original_ids


# ---------------------------------------------------------------------------
# 3. CSV output and resume
# ---------------------------------------------------------------------------

def test_scale_subscription_count_writes_csv():
    """Should write results incrementally to CSV."""
    from scripts.run_scaling import scale_subscription_count

    dataset = _make_test_dataset(n_subs=10, n_events=3)
    llm = DryRunLLMClient(model="test-model")
    embedder = EmbeddingModel("all-MiniLM-L6-v2")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        scale_subscription_count(
            dataset=dataset,
            embedder=embedder,
            llm_client=llm,
            seed=42,
            output_dir=output_dir,
            sub_counts=[5],
            configs=["A0"],
        )

        csv_path = output_dir / f"scaling_subs_count_T1.csv"
        assert csv_path.exists()
        loaded = pd.read_csv(csv_path)
        assert len(loaded) == 1


# ---------------------------------------------------------------------------
# 4. Memory tracking
# ---------------------------------------------------------------------------

def test_scale_subscription_count_tracks_memory():
    """Results should include memory_mb column."""
    from scripts.run_scaling import scale_subscription_count

    dataset = _make_test_dataset(n_subs=10, n_events=3)
    llm = DryRunLLMClient(model="test-model")
    embedder = EmbeddingModel("all-MiniLM-L6-v2")

    with tempfile.TemporaryDirectory() as tmpdir:
        df = scale_subscription_count(
            dataset=dataset,
            embedder=embedder,
            llm_client=llm,
            seed=42,
            output_dir=Path(tmpdir),
            sub_counts=[5],
            configs=["A0"],
        )

    assert "memory_mb" in df.columns
