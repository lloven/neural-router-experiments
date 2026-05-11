"""Tests for the Puhti → local manifest reconciliation tool.

The tool reads CSVs that the SLURM array job (puhti_qwen7b_ablation.sh) writes
to /scratch/project_2018951/neural-router/code/results/full/ablation/by_task/
and updates the local manifest with status=done + metrics for matching runs.

Content enforcement: a CSV without a data row, or with empty/null F1,
must NOT be treated as done.

TDD RED phase: tests written before implementation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.manifest import Manifest, RunEntry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _entry(
    run_id: str,
    stage: str = "ablation",
    dataset: str = "D1",
    config: str = "A3",
    seed: int = 42,
    model_key: str = "qwen7b",
    status: str = "pending",
) -> RunEntry:
    return RunEntry(
        run_id=run_id,
        stage=stage,
        dataset=dataset,
        model_key=model_key,
        model_id="ollama/qwen2.5:7b",
        config=config,
        seed=seed,
        status=status,
        slot_type="ollama",
        max_events=None,
        max_event_words=None,
        started_at=None,
        finished_at=None,
        error=None,
        result_file=f"results/full/ablation/{dataset}_ablation_{model_key}_results.csv",
        metrics=None,
    )


@pytest.fixture
def manifest(tmp_path: Path) -> Manifest:
    runs = {
        "ablation__D1__A3__seed42__qwen7b": _entry(
            "ablation__D1__A3__seed42__qwen7b", config="A3"
        ),
        "ablation__D1__A0__seed42__qwen7b": _entry(
            "ablation__D1__A0__seed42__qwen7b", config="A0"
        ),
        "ablation__D2__A1__seed123__qwen7b": _entry(
            "ablation__D2__A1__seed123__qwen7b",
            dataset="D2", config="A1", seed=123,
        ),
        # Already done — must not be touched.
        "ablation__D3__A0__seed42__qwen7b": _entry(
            "ablation__D3__A0__seed42__qwen7b",
            dataset="D3", config="A0", status="done",
        ),
    }
    return Manifest(mode="full", runs=runs)


def _good_csv(path: Path, config: str, dataset: str, seed: int, f1: float = 0.66) -> None:
    """Write a CSV with one valid data row, matching run_experiment.py output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "config": config,
        "dataset": dataset,
        "seed": seed,
        "f1": f1,
        "precision": f1 - 0.05,
        "recall": f1 + 0.05,
        "fpr": 0.02,
        "invocations": 19,
        "compression_ratio": 0.6,
        "latency_s": 12.3,
        "tokens_prompt": 7514,
        "tokens_response": 2029,
        "cost_per_1k": 0.023,
    }]).to_csv(path, index=False)


def _empty_csv(path: Path) -> None:
    """Write a CSV with header only, no data rows (silent-failure mode)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=[
        "config", "dataset", "seed", "f1", "precision", "recall", "fpr",
        "invocations", "compression_ratio", "latency_s",
        "tokens_prompt", "tokens_response", "cost_per_1k",
    ]).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# 1. Path → run_id mapping
# ---------------------------------------------------------------------------


def test_task_dir_to_run_id_basic():
    """qwen7b_D1_A3_s42 → ablation__D1__A3__seed42__qwen7b."""
    from scripts.reconcile_puhti import task_dir_to_run_id

    assert (
        task_dir_to_run_id("qwen7b_D1_A3_s42")
        == "ablation__D1__A3__seed42__qwen7b"
    )
    assert (
        task_dir_to_run_id("qwen7b_D2_A1_s123")
        == "ablation__D2__A1__seed123__qwen7b"
    )


def test_task_dir_to_run_id_tier1c_unmatched():
    """Tier 1c result dirs (llama-8b, qwen2.5-32b, …) are unmatched on
    purpose: they are supplementary experiments not tracked in the local
    paper manifest. They go through the Tier 1c analysis pipeline instead.
    """
    from scripts.reconcile_puhti import task_dir_to_run_id

    assert task_dir_to_run_id("llama-8b_D2_A1_s42") is None
    assert task_dir_to_run_id("qwen2.5-32b_D2_A0_s1024") is None
    assert task_dir_to_run_id("qwen2.5-1.5b_D2_A0_s42") is None
    assert task_dir_to_run_id("phi3-mini_D1_A4_s123") is None


def test_task_dir_to_run_id_unknown_format_returns_none():
    from scripts.reconcile_puhti import task_dir_to_run_id

    assert task_dir_to_run_id("not_a_task_dir") is None
    assert task_dir_to_run_id("qwen7b_D1_A3") is None  # missing seed


# ---------------------------------------------------------------------------
# 2. CSV validation (content, not just existence)
# ---------------------------------------------------------------------------


def test_validate_csv_accepts_good_csv(tmp_path: Path):
    from scripts.reconcile_puhti import validate_csv

    csv = tmp_path / "qwen7b_D1_A3_s42_results.csv"
    _good_csv(csv, config="A3", dataset="D1", seed=42)

    result = validate_csv(csv, expected_config="A3", expected_dataset="D1", expected_seed=42)
    assert result is not None
    assert result["f1"] == pytest.approx(0.66)


def test_validate_csv_rejects_empty(tmp_path: Path):
    from scripts.reconcile_puhti import validate_csv

    csv = tmp_path / "qwen7b_D1_A3_s42_results.csv"
    _empty_csv(csv)

    assert validate_csv(csv, expected_config="A3", expected_dataset="D1", expected_seed=42) is None


def test_validate_csv_rejects_missing(tmp_path: Path):
    from scripts.reconcile_puhti import validate_csv

    csv = tmp_path / "does_not_exist.csv"
    assert validate_csv(csv, expected_config="A3", expected_dataset="D1", expected_seed=42) is None


def test_validate_csv_rejects_null_f1(tmp_path: Path):
    """A row with null F1 is a silent failure, not done."""
    from scripts.reconcile_puhti import validate_csv

    csv = tmp_path / "qwen7b_D1_A3_s42_results.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "config": "A3", "dataset": "D1", "seed": 42,
        "f1": None, "precision": None, "recall": None,
    }]).to_csv(csv, index=False)

    assert validate_csv(csv, expected_config="A3", expected_dataset="D1", expected_seed=42) is None


def test_validate_csv_rejects_config_mismatch(tmp_path: Path):
    """Defence in depth: if the CSV claims a different config, refuse."""
    from scripts.reconcile_puhti import validate_csv

    csv = tmp_path / "qwen7b_D1_A3_s42_results.csv"
    _good_csv(csv, config="A0", dataset="D1", seed=42)  # mismatched config

    assert validate_csv(csv, expected_config="A3", expected_dataset="D1", expected_seed=42) is None


# ---------------------------------------------------------------------------
# 3. Manifest reconciliation
# ---------------------------------------------------------------------------


def test_reconcile_marks_pending_done(manifest: Manifest, tmp_path: Path):
    """Given a good Puhti CSV for a pending run, mark it done with metrics."""
    from scripts.reconcile_puhti import reconcile_local_dir

    by_task = tmp_path / "by_task"
    task_dir = by_task / "qwen7b_D1_A3_s42"
    csv = task_dir / "qwen7b_D1_A3_s42_results.csv"
    _good_csv(csv, config="A3", dataset="D1", seed=42, f1=0.71)

    summary = reconcile_local_dir(manifest, by_task)
    assert summary["marked_done"] == 1
    assert summary["already_done"] == 0
    assert summary["unmatched_dirs"] == 0
    assert summary["invalid_csvs"] == 0

    entry = manifest.runs["ablation__D1__A3__seed42__qwen7b"]
    assert entry.status == "done"
    assert entry.metrics is not None
    assert entry.metrics["f1"] == pytest.approx(0.71)


def test_reconcile_skips_already_done(manifest: Manifest, tmp_path: Path):
    """Don't overwrite metrics for an already-done run."""
    from scripts.reconcile_puhti import reconcile_local_dir

    by_task = tmp_path / "by_task"
    task_dir = by_task / "qwen7b_D3_A0_s42"
    csv = task_dir / "qwen7b_D3_A0_s42_results.csv"
    _good_csv(csv, config="A0", dataset="D3", seed=42)

    summary = reconcile_local_dir(manifest, by_task)
    assert summary["marked_done"] == 0
    assert summary["already_done"] == 1
    # Existing entry untouched.
    assert manifest.runs["ablation__D3__A0__seed42__qwen7b"].status == "done"


def test_reconcile_counts_invalid_csvs(manifest: Manifest, tmp_path: Path):
    """Invalid CSVs must NOT mark runs done; they go in invalid_csvs."""
    from scripts.reconcile_puhti import reconcile_local_dir

    by_task = tmp_path / "by_task"
    task_dir = by_task / "qwen7b_D1_A3_s42"
    csv = task_dir / "qwen7b_D1_A3_s42_results.csv"
    _empty_csv(csv)

    summary = reconcile_local_dir(manifest, by_task)
    assert summary["marked_done"] == 0
    assert summary["invalid_csvs"] == 1
    assert manifest.runs["ablation__D1__A3__seed42__qwen7b"].status == "pending"


def test_reconcile_unmatched_dirs(manifest: Manifest, tmp_path: Path):
    """A by_task dir whose run_id has no manifest entry counts as unmatched."""
    from scripts.reconcile_puhti import reconcile_local_dir

    by_task = tmp_path / "by_task"
    # M0 doesn't exist in the manifest at all.
    task_dir = by_task / "qwen7b_D1_M0_s42"
    csv = task_dir / "qwen7b_D1_M0_s42_results.csv"
    _good_csv(csv, config="M0", dataset="D1", seed=42)

    summary = reconcile_local_dir(manifest, by_task)
    assert summary["unmatched_dirs"] == 1
    assert summary["marked_done"] == 0


def test_reconcile_dry_run_does_not_mutate(manifest: Manifest, tmp_path: Path):
    """dry_run=True returns the would-be summary without changing the manifest."""
    from scripts.reconcile_puhti import reconcile_local_dir

    by_task = tmp_path / "by_task"
    task_dir = by_task / "qwen7b_D1_A3_s42"
    csv = task_dir / "qwen7b_D1_A3_s42_results.csv"
    _good_csv(csv, config="A3", dataset="D1", seed=42)

    summary = reconcile_local_dir(manifest, by_task, dry_run=True)
    assert summary["marked_done"] == 1  # would-be count
    # But manifest entry is unchanged.
    assert manifest.runs["ablation__D1__A3__seed42__qwen7b"].status == "pending"
    assert manifest.runs["ablation__D1__A3__seed42__qwen7b"].metrics is None
