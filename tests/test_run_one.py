"""Tests for get_model_overrides(), stage wrapper signatures, and single-run executor.

TDD RED phase: tests written before implementation.
"""

import inspect
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import get_model_overrides
from src.evaluation import EvaluationResult, MetricResult
from src.manifest import Manifest, RunEntry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_entry() -> RunEntry:
    return RunEntry(
        run_id="ablation__D1__A3__seed42__qwen7b",
        stage="ablation",
        dataset="D1",
        model_key="qwen7b",
        model_id="ollama/qwen2.5:7b",
        config="A3",
        seed=42,
        status="pending",
        slot_type="ollama",
        max_events=100,
        max_event_words=None,
        started_at=None,
        finished_at=None,
        error=None,
        result_file="results/test/ablation/D1_ablation_qwen7b_results.csv",
        metrics=None,
    )


@pytest.fixture
def sample_profile() -> dict:
    return {
        "mode": "integration_smoke",
        "seeds": [42],
        "datasets": ["D1"],
        "max_events": 100,
        "max_event_words": None,
        "ablation": {"configs": ["A3"], "baselines": ["bm25"]},
        "sensitivity": {"sweeps": ["k"], "k_values": [5, 10]},
        "scaling": {"event_counts": [50, 100]},
        "llm_models": {
            "qwen7b": {"id": "ollama/qwen2.5:7b"},
        },
        "defaults": {},
    }


# ---------------------------------------------------------------------------
# 3a: get_model_overrides
# ---------------------------------------------------------------------------

class TestGetModelOverrides:

    def test_qwen7b(self):
        ov = get_model_overrides("qwen7b")
        assert ov["batch_size"] == 10
        assert ov["llm_timeout"] == 900
        assert ov["checkpoint_interval"] == 5
        assert ov["max_parallel"] == 1
        assert ov["use_async"] is False

    def test_haiku(self):
        ov = get_model_overrides("haiku")
        assert ov["batch_size"] is None
        assert ov["llm_timeout"] is None
        assert ov["checkpoint_interval"] == 25
        assert ov["max_parallel"] == 2
        assert ov["use_async"] is True

    def test_sonnet(self):
        ov = get_model_overrides("sonnet")
        assert ov["batch_size"] is None
        assert ov["llm_timeout"] is None
        assert ov["checkpoint_interval"] == 50
        assert ov["max_parallel"] == 4
        assert ov["use_async"] is True

    def test_unknown_returns_defaults(self):
        ov = get_model_overrides("unknown_model_xyz")
        # Should return sensible defaults, not crash
        assert "batch_size" in ov
        assert "llm_timeout" in ov
        assert "checkpoint_interval" in ov
        assert "max_parallel" in ov
        assert "use_async" in ov
        # Defaults should be conservative API-style
        assert ov["batch_size"] is None
        assert ov["llm_timeout"] is None
        assert ov["checkpoint_interval"] == 50
        assert ov["max_parallel"] == 4
        assert ov["use_async"] is True

    def test_return_type_is_dict(self):
        ov = get_model_overrides("qwen7b")
        assert isinstance(ov, dict)

    def test_all_keys_present(self):
        expected_keys = {"batch_size", "llm_timeout", "checkpoint_interval", "max_parallel", "use_async"}
        for model in ["qwen7b", "haiku", "sonnet", "unknown"]:
            ov = get_model_overrides(model)
            assert set(ov.keys()) == expected_keys, f"Missing keys for {model}"


# ---------------------------------------------------------------------------
# 3b: Stage wrapper signatures
# ---------------------------------------------------------------------------

class TestStageWrapperSignatures:

    def test_run_ablation_stage_exists_and_accepts_params(self):
        from scripts.run_experiment import run_ablation_stage

        sig = inspect.signature(run_ablation_stage)
        params = list(sig.parameters.keys())
        assert "entry" in params
        assert "profile" in params
        assert "llm_client" in params
        assert "embedding_model" in params

    def test_run_sensitivity_stage_exists_and_accepts_params(self):
        from scripts.run_sensitivity import run_sensitivity_stage

        sig = inspect.signature(run_sensitivity_stage)
        params = list(sig.parameters.keys())
        assert "entry" in params
        assert "profile" in params
        assert "llm_client" in params
        assert "embedding_model" in params

    def test_run_scaling_stage_exists_and_accepts_params(self):
        from scripts.run_scaling import run_scaling_stage

        sig = inspect.signature(run_scaling_stage)
        params = list(sig.parameters.keys())
        assert "entry" in params
        assert "profile" in params
        assert "llm_client" in params
        assert "embedding_model" in params

    def test_run_ablation_stage_type_hints(self):
        from scripts.run_experiment import run_ablation_stage

        sig = inspect.signature(run_ablation_stage)
        # entry should accept RunEntry
        assert sig.parameters["entry"].annotation is not inspect.Parameter.empty
        # llm_client and embedding_model should default to None
        assert sig.parameters["llm_client"].default is None
        assert sig.parameters["embedding_model"].default is None


# ---------------------------------------------------------------------------
# 4: Single-Run Executor (execute_run)
# ---------------------------------------------------------------------------

def _make_metric(name: str, value: float) -> MetricResult:
    """Helper: create a MetricResult with uniform CI."""
    return MetricResult(name=name, mean=value, ci_lower=value - 0.01,
                        ci_upper=value + 0.01, std=0.02, n=10)


def _make_eval_result(config: str = "A3", dataset: str = "D1",
                      seed: int = 42) -> EvaluationResult:
    """Helper: create a plausible EvaluationResult for mocking."""
    return EvaluationResult(
        config_name=config,
        dataset_name=dataset,
        seed=seed,
        precision=_make_metric("precision", 0.85),
        recall=_make_metric("recall", 0.80),
        f1=_make_metric("f1", 0.82),
        fpr=_make_metric("fpr", 0.05),
        invocations=10,
        compression_ratio=3.5,
        latency_s=12.0,
        tokens_prompt=5000,
        tokens_response=1000,
    )


def _make_entry(stage: str = "ablation", config: str = "A3",
                run_id: str | None = None) -> RunEntry:
    """Helper: build a RunEntry for the given stage."""
    rid = run_id or f"{stage}__D1__{config}__seed42__qwen7b"
    return RunEntry(
        run_id=rid,
        stage=stage,
        dataset="D1",
        model_key="qwen7b",
        model_id="ollama/qwen2.5:7b",
        config=config,
        seed=42,
        status="pending",
        slot_type="ollama",
        max_events=100,
        max_event_words=None,
        started_at=None,
        finished_at=None,
        error=None,
        result_file="results/test/ablation/D1_ablation_qwen7b_results.csv",
        metrics=None,
    )


class TestExecutorDispatch:
    """Verify execute_run dispatches to the correct stage wrapper."""

    @patch("scripts.run_one.run_ablation_stage")
    def test_executor_dispatches_ablation(self, mock_ablation, tmp_path, sample_profile):
        from scripts.run_one import execute_run

        entry = _make_entry(stage="ablation", config="A3")
        manifest = Manifest(mode="test", runs={entry.run_id: entry})
        manifest_path = tmp_path / "manifest.json"
        manifest.save(manifest_path)

        mock_ablation.return_value = [_make_eval_result()]

        execute_run(entry.run_id, manifest_path, sample_profile)

        mock_ablation.assert_called_once()
        call_args = mock_ablation.call_args
        assert call_args[1]["entry"].run_id == entry.run_id or call_args[0][0].run_id == entry.run_id

    @patch("scripts.run_one.run_sensitivity_stage")
    def test_executor_dispatches_sensitivity(self, mock_sensitivity, tmp_path, sample_profile):
        from scripts.run_one import execute_run

        entry = _make_entry(stage="sensitivity", config="sweep_k",
                            run_id="sensitivity__D1__sweep_k__seed42__qwen7b")
        manifest = Manifest(mode="test", runs={entry.run_id: entry})
        manifest_path = tmp_path / "manifest.json"
        manifest.save(manifest_path)

        mock_sensitivity.return_value = None

        execute_run(entry.run_id, manifest_path, sample_profile)

        mock_sensitivity.assert_called_once()

    @patch("scripts.run_one.run_scaling_stage")
    def test_executor_dispatches_scaling(self, mock_scaling, tmp_path, sample_profile):
        from scripts.run_one import execute_run

        entry = _make_entry(stage="scaling", config="scale_events_100",
                            run_id="scaling__D2__scale_events_100__seed0__qwen7b")
        manifest = Manifest(mode="test", runs={entry.run_id: entry})
        manifest_path = tmp_path / "manifest.json"
        manifest.save(manifest_path)

        mock_scaling.return_value = None

        execute_run(entry.run_id, manifest_path, sample_profile)

        mock_scaling.assert_called_once()


class TestExecutorFlushesResults:
    """Verify ablation results are written to CSV at entry.result_file."""

    @patch("scripts.run_one.run_ablation_stage")
    def test_ablation_result_flushed_to_csv(self, mock_ablation, tmp_path, sample_profile):
        """After a successful ablation run, the result CSV must exist and
        contain a row matching the config and seed."""
        import csv
        from scripts.run_one import execute_run

        result_file = tmp_path / "ablation" / "D1_ablation_qwen7b_results.csv"
        entry = _make_entry(stage="ablation", config="A3")
        entry.result_file = str(result_file)

        manifest = Manifest(mode="test", runs={entry.run_id: entry})
        manifest_path = tmp_path / "manifest.json"
        manifest.save(manifest_path)

        mock_ablation.return_value = [_make_eval_result()]

        execute_run(entry.run_id, manifest_path, sample_profile)

        # The result CSV must exist
        assert result_file.exists(), f"Result CSV not written at {result_file}"
        # It must contain a data row for config=A3, seed=42
        with open(result_file) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) >= 1, "Result CSV has no data rows"
        matching = [r for r in rows if r["config"] == "A3" and r["seed"] == "42"]
        assert len(matching) == 1, f"Expected 1 row for A3/seed42, got {len(matching)}"
        assert float(matching[0]["f1"]) == pytest.approx(0.82, abs=0.01)

    @patch("scripts.run_one.run_ablation_stage")
    def test_ablation_csv_appends_not_overwrites(self, mock_ablation, tmp_path, sample_profile):
        """Two ablation runs with different seeds should both appear in the CSV."""
        import csv
        from scripts.run_one import execute_run

        result_file = tmp_path / "ablation" / "D1_ablation_qwen7b_results.csv"

        # First run: seed=42
        entry1 = _make_entry(stage="ablation", config="A3",
                             run_id="ablation__D1__A3__seed42__qwen7b")
        entry1.result_file = str(result_file)
        entry1.seed = 42
        # Second run: seed=123
        entry2 = _make_entry(stage="ablation", config="A3",
                             run_id="ablation__D1__A3__seed123__qwen7b")
        entry2.result_file = str(result_file)
        entry2.seed = 123

        manifest = Manifest(mode="test", runs={
            entry1.run_id: entry1,
            entry2.run_id: entry2,
        })
        manifest_path = tmp_path / "manifest.json"
        manifest.save(manifest_path)

        mock_ablation.return_value = [_make_eval_result(seed=42)]
        execute_run(entry1.run_id, manifest_path, sample_profile)

        mock_ablation.return_value = [_make_eval_result(seed=123)]
        execute_run(entry2.run_id, manifest_path, sample_profile)

        with open(result_file) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2, f"Expected 2 rows, got {len(rows)}"
        seeds = {r["seed"] for r in rows}
        assert seeds == {"42", "123"}

    @patch("scripts.run_one.run_ablation_stage")
    def test_baseline_result_flushed_to_csv(self, mock_ablation, tmp_path, sample_profile):
        """Baseline ablation runs should also be flushed to CSV."""
        import csv
        from scripts.run_one import execute_run

        result_file = tmp_path / "ablation" / "D1_ablation_qwen7b_results.csv"
        entry = _make_entry(stage="ablation", config="baseline_bm25",
                            run_id="ablation__D1__baseline_bm25__seed0__qwen7b")
        entry.result_file = str(result_file)
        entry.seed = 0

        manifest = Manifest(mode="test", runs={entry.run_id: entry})
        manifest_path = tmp_path / "manifest.json"
        manifest.save(manifest_path)

        mock_ablation.return_value = [_make_eval_result(config="baseline_bm25", seed=0)]

        execute_run(entry.run_id, manifest_path, sample_profile)

        assert result_file.exists()
        with open(result_file) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) >= 1
        assert rows[0]["config"] == "baseline_bm25"

    @patch("scripts.run_one.run_ablation_stage")
    def test_no_csv_written_on_failure(self, mock_ablation, tmp_path, sample_profile):
        """A failed ablation run should NOT create/modify the CSV."""
        from scripts.run_one import execute_run

        result_file = tmp_path / "ablation" / "D1_ablation_qwen7b_results.csv"
        entry = _make_entry(stage="ablation", config="A3")
        entry.result_file = str(result_file)

        manifest = Manifest(mode="test", runs={entry.run_id: entry})
        manifest_path = tmp_path / "manifest.json"
        manifest.save(manifest_path)

        mock_ablation.side_effect = RuntimeError("LLM timeout")

        execute_run(entry.run_id, manifest_path, sample_profile)

        assert not result_file.exists(), "CSV should not be created on failure"


class TestExecutorManifestUpdates:
    """Verify manifest is updated on success and failure."""

    @patch("scripts.run_one.run_ablation_stage")
    def test_executor_updates_manifest_on_success(self, mock_ablation, tmp_path, sample_profile):
        from scripts.run_one import execute_run

        entry = _make_entry(stage="ablation", config="A3")
        manifest = Manifest(mode="test", runs={entry.run_id: entry})
        manifest_path = tmp_path / "manifest.json"
        manifest.save(manifest_path)

        mock_ablation.return_value = [_make_eval_result()]

        execute_run(entry.run_id, manifest_path, sample_profile)

        updated = Manifest.load(manifest_path)
        run = updated.runs[entry.run_id]
        assert run.status == "done"
        assert run.finished_at is not None
        assert run.error is None
        # Metrics should be populated from the EvaluationResult
        assert run.metrics is not None
        assert run.metrics["f1"] == pytest.approx(0.82, abs=0.01)

    @patch("scripts.run_one.run_ablation_stage")
    def test_executor_updates_manifest_on_failure(self, mock_ablation, tmp_path, sample_profile):
        from scripts.run_one import execute_run

        entry = _make_entry(stage="ablation", config="A3")
        manifest = Manifest(mode="test", runs={entry.run_id: entry})
        manifest_path = tmp_path / "manifest.json"
        manifest.save(manifest_path)

        mock_ablation.side_effect = RuntimeError("LLM timeout after 900s")

        # execute_run should not raise; it catches and records the error
        execute_run(entry.run_id, manifest_path, sample_profile)

        updated = Manifest.load(manifest_path)
        run = updated.runs[entry.run_id]
        assert run.status == "failed"
        assert "LLM timeout after 900s" in run.error
        assert run.metrics is None
