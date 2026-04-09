"""Tests for the crossover validation experiment (Phase 1c).

The crossover experiment demonstrates that A0 (raw LLM) dominates at low
subscription counts but A4 (full pipeline) wins when subscriptions exceed
the context window.  This is the paper's central figure.

Design:
  - Context-constrained model (simulated via max_context_tokens=4096)
  - Subscription volumes: [50, 100, 200, 500, 1000, 2000]
  - Configs: A0 (raw) vs A4 (full pipeline)
  - Seeds: [42, 123, 456]
  - Metrics: F1, latency, token cost
  - Dataset: D1 (CardiffNLP)

TDD RED phase: tests written before implementation.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# 1. run_crossover_point function signature and basic behavior
# ---------------------------------------------------------------------------

def test_run_crossover_point_returns_dataframe():
    """run_crossover_point should return a DataFrame with expected columns."""
    from scripts.run_crossover import run_crossover_point
    from src.data import Subscription, Event, Dataset
    from src.llm import DryRunLLMClient
    from src.embeddings import EmbeddingModel

    # Minimal dataset
    subs = [
        Subscription(id=f"sub_{i}", name=f"Sub {i}", description=f"Topic about subject {i}")
        for i in range(10)
    ]
    events = [
        Event(id=f"evt_{i}", text=f"Event text {i}", ground_truth=["sub_0"])
        for i in range(5)
    ]
    dataset = Dataset(name="Test", short_name="T1", events=events, subscriptions=subs)
    llm = DryRunLLMClient(model="test-model")
    embedder = EmbeddingModel("all-MiniLM-L6-v2")

    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_crossover_point(
            dataset=dataset,
            config_name="A0",
            n_subs=10,
            max_context_tokens=4096,
            seed=42,
            llm_client=llm,
            embedder=embedder,
            output_dir=Path(tmpdir),
        )

    assert isinstance(result, dict)
    assert "n_subscriptions" in result
    assert "config" in result
    assert "seed" in result
    assert "f1" in result
    assert "latency_s" in result
    assert "n_subs_effective" in result
    assert result["n_subscriptions"] == 10
    assert result["config"] == "A0"
    assert result["seed"] == 42


def test_run_crossover_point_subsamples_subscriptions():
    """When n_subs < dataset.num_subscriptions, subscriptions are subsampled."""
    from scripts.run_crossover import run_crossover_point
    from src.data import Subscription, Event, Dataset
    from src.llm import DryRunLLMClient
    from src.embeddings import EmbeddingModel

    subs = [
        Subscription(id=f"sub_{i}", name=f"Sub {i}", description=f"Topic {i}")
        for i in range(50)
    ]
    events = [
        Event(id="evt_0", text="Test event", ground_truth=["sub_0"])
    ]
    dataset = Dataset(name="Test", short_name="T1", events=events, subscriptions=subs)
    llm = DryRunLLMClient(model="test-model")
    embedder = EmbeddingModel("all-MiniLM-L6-v2")

    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_crossover_point(
            dataset=dataset,
            config_name="A0",
            n_subs=20,
            max_context_tokens=4096,
            seed=42,
            llm_client=llm,
            embedder=embedder,
            output_dir=Path(tmpdir),
        )

    assert result["n_subscriptions"] == 20


# ---------------------------------------------------------------------------
# 2. Full crossover sweep
# ---------------------------------------------------------------------------

def test_run_crossover_sweep_produces_all_combinations():
    """run_crossover_sweep should produce one row per (config, n_subs, seed)."""
    from scripts.run_crossover import run_crossover_sweep
    from src.data import Subscription, Event, Dataset
    from src.llm import DryRunLLMClient
    from src.embeddings import EmbeddingModel

    subs = [
        Subscription(id=f"sub_{i}", name=f"Sub {i}", description=f"Topic {i}")
        for i in range(30)
    ]
    events = [
        Event(id="evt_0", text="Test event", ground_truth=["sub_0"])
    ]
    dataset = Dataset(name="Test", short_name="T1", events=events, subscriptions=subs)
    llm = DryRunLLMClient(model="test-model")
    embedder = EmbeddingModel("all-MiniLM-L6-v2")

    sub_volumes = [10, 20]
    configs = ["A0", "A4"]
    seeds = [42]

    with tempfile.TemporaryDirectory() as tmpdir:
        df = run_crossover_sweep(
            dataset=dataset,
            configs=configs,
            sub_volumes=sub_volumes,
            max_context_tokens=4096,
            seeds=seeds,
            llm_client=llm,
            embedder=embedder,
            output_dir=Path(tmpdir),
        )

    assert isinstance(df, pd.DataFrame)
    # 2 configs * 2 volumes * 1 seed = 4 rows
    assert len(df) == 4
    assert set(df["config"].unique()) == {"A0", "A4"}
    assert set(df["n_subscriptions"].unique()) == {10, 20}


# ---------------------------------------------------------------------------
# 3. CSV output
# ---------------------------------------------------------------------------

def test_crossover_sweep_writes_csv():
    """Crossover sweep should write results to a CSV file."""
    from scripts.run_crossover import run_crossover_sweep
    from src.data import Subscription, Event, Dataset
    from src.llm import DryRunLLMClient
    from src.embeddings import EmbeddingModel

    subs = [
        Subscription(id=f"sub_{i}", name=f"Sub {i}", description=f"Topic {i}")
        for i in range(20)
    ]
    events = [
        Event(id="evt_0", text="Test event", ground_truth=["sub_0"])
    ]
    dataset = Dataset(name="Test", short_name="T1", events=events, subscriptions=subs)
    llm = DryRunLLMClient(model="test-model")
    embedder = EmbeddingModel("all-MiniLM-L6-v2")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        df = run_crossover_sweep(
            dataset=dataset,
            configs=["A0"],
            sub_volumes=[10],
            max_context_tokens=4096,
            seeds=[42],
            llm_client=llm,
            embedder=embedder,
            output_dir=output_dir,
        )

        csv_path = output_dir / "crossover_T1.csv"
        assert csv_path.exists(), f"Expected CSV at {csv_path}"
        loaded = pd.read_csv(csv_path)
        assert len(loaded) == 1


# ---------------------------------------------------------------------------
# 4. Resume support
# ---------------------------------------------------------------------------

def test_crossover_sweep_resumes_from_existing_csv():
    """Crossover sweep should skip already-completed (config, n_subs, seed) combos."""
    from scripts.run_crossover import run_crossover_sweep
    from src.data import Subscription, Event, Dataset
    from src.llm import DryRunLLMClient
    from src.embeddings import EmbeddingModel

    subs = [
        Subscription(id=f"sub_{i}", name=f"Sub {i}", description=f"Topic {i}")
        for i in range(20)
    ]
    events = [
        Event(id="evt_0", text="Test event", ground_truth=["sub_0"])
    ]
    dataset = Dataset(name="Test", short_name="T1", events=events, subscriptions=subs)
    llm = DryRunLLMClient(model="test-model")
    embedder = EmbeddingModel("all-MiniLM-L6-v2")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # First run
        run_crossover_sweep(
            dataset=dataset,
            configs=["A0"],
            sub_volumes=[10],
            max_context_tokens=4096,
            seeds=[42],
            llm_client=llm,
            embedder=embedder,
            output_dir=output_dir,
        )

        # Second run with additional volume -- should not re-run the first
        df = run_crossover_sweep(
            dataset=dataset,
            configs=["A0"],
            sub_volumes=[10, 15],
            max_context_tokens=4096,
            seeds=[42],
            llm_client=llm,
            embedder=embedder,
            output_dir=output_dir,
        )

        csv_path = output_dir / "crossover_T1.csv"
        loaded = pd.read_csv(csv_path)
        assert len(loaded) == 2  # one old + one new
