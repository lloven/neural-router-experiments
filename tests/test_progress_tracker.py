"""Tests for progress tracker task count propagation during checkpointing.

Bug: run_ablation_config() calls progress_tracker.update() during checkpoint
loops but only passes items_done/items_total, leaving tasks_done/tasks_total
at 0. The monitor therefore shows no task-level progress during the long
event-processing phase.
"""
import json
import tempfile
import time
from pathlib import Path

from src.progress import ProgressTracker, SidecarProgressCallback


def test_progress_update_preserves_task_counts():
    """ProgressTracker.update() should write tasks_done/tasks_total to JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ProgressTracker("test-stage", mode="test", results_root=tmpdir)
        tracker.update(
            current_task="A0 seed=42",
            items_done=50,
            items_total=200,
            tasks_done=3,
            tasks_total=7,
            detail="Events 50/200",
            force=True,
        )
        data = json.loads((Path(tmpdir) / "test" / ".progress" / "test-stage.json").read_text())
        assert data["tasks_done"] == 3
        assert data["tasks_total"] == 7


def test_checkpoint_update_includes_task_counts():
    """When run_ablation_config() calls tracker.update() during checkpointing,
    the written JSON must contain non-zero tasks_done and tasks_total.

    This simulates the checkpoint update path inside run_ablation_config().
    The bug: tasks_done/tasks_total are not passed, defaulting to 0.
    """
    import sys
    import types

    # We can't easily run run_ablation_config (needs LLM, data, etc.),
    # so we test the contract: the function signature must accept
    # tasks_done and tasks_total, and forward them to the tracker.
    from scripts.run_experiment import run_ablation_config
    import inspect

    sig = inspect.signature(run_ablation_config)
    param_names = list(sig.parameters.keys())

    assert "tasks_done" in param_names, (
        "run_ablation_config() must accept tasks_done parameter "
        "to forward task counts to progress tracker during checkpointing"
    )
    assert "tasks_total" in param_names, (
        "run_ablation_config() must accept tasks_total parameter "
        "to forward task counts to progress tracker during checkpointing"
    )


# ---------------------------------------------------------------------------
# SidecarProgressCallback tests
# ---------------------------------------------------------------------------

class TestSidecarProgressCallback:
    """Tests for the SidecarProgressCallback that writes run_id-keyed JSON."""

    def test_sidecar_callback_writes_json(self, tmp_path):
        """Create callback, call update(), verify JSON file exists with correct fields."""
        cb = SidecarProgressCallback(
            run_id="ablation__D1__A3__seed42__sonnet",
            mode="integration_smoke",
            results_root=str(tmp_path),
        )
        cb.update(
            current_task="A3 seed=42",
            items_done=50,
            items_total=200,
            tasks_done=1,
            tasks_total=5,
            detail="Processing events 50/200",
            force=True,
        )
        sidecar = tmp_path / "integration_smoke" / ".progress" / "ablation__D1__A3__seed42__sonnet.json"
        assert sidecar.exists(), f"Sidecar file should exist at {sidecar}"

        data = json.loads(sidecar.read_text())
        assert data["run_id"] == "ablation__D1__A3__seed42__sonnet"
        assert data["status"] == "running"
        assert data["items_done"] == 50
        assert data["items_total"] == 200
        assert data["tasks_done"] == 1
        assert data["tasks_total"] == 5
        assert data["detail"] == "Processing events 50/200"
        assert "elapsed_s" in data
        assert "eta_s" in data

    def test_sidecar_callback_respects_min_interval(self, tmp_path):
        """Two rapid updates: only first should write (unless force=True)."""
        cb = SidecarProgressCallback(
            run_id="test-run",
            mode="test",
            results_root=str(tmp_path),
        )
        # First update (force to ensure it writes)
        cb.update(items_done=10, items_total=100, force=True)
        sidecar = tmp_path / "test" / ".progress" / "test-run.json"
        data1 = json.loads(sidecar.read_text())
        assert data1["items_done"] == 10

        # Second update immediately (no force) -- should be throttled
        cb.update(items_done=20, items_total=100)
        data2 = json.loads(sidecar.read_text())
        assert data2["items_done"] == 10, "Second rapid update should be throttled"

        # Third update with force=True -- should write
        cb.update(items_done=30, items_total=100, force=True)
        data3 = json.loads(sidecar.read_text())
        assert data3["items_done"] == 30, "Forced update should always write"

    def test_sidecar_callback_finish_marks_done(self, tmp_path):
        """Call finish(), verify status='done' in JSON."""
        cb = SidecarProgressCallback(
            run_id="finish-test",
            mode="test",
            results_root=str(tmp_path),
        )
        cb.update(items_done=50, items_total=100, force=True)
        cb.finish()

        sidecar = tmp_path / "test" / ".progress" / "finish-test.json"
        data = json.loads(sidecar.read_text())
        assert data["status"] == "done"
        assert "finished_at" in data

    def test_sidecar_callback_uses_run_id_as_filename(self, tmp_path):
        """Verify file is named {run_id}.json."""
        run_id = "scaling__D2__scale_events_500__seed0__qwen7b"
        cb = SidecarProgressCallback(
            run_id=run_id,
            mode="full",
            results_root=str(tmp_path),
        )
        cb.update(items_done=1, items_total=10, force=True)

        expected = tmp_path / "full" / ".progress" / f"{run_id}.json"
        assert expected.exists(), f"Sidecar should be at {expected}"
        # Also confirm the run_id inside matches
        data = json.loads(expected.read_text())
        assert data["run_id"] == run_id
