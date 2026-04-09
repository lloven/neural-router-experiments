"""Tests for scripts/restart.sh (dry-run mode).

The restart script automates:
1. Kill stale orchestrator processes
2. Reset failed/stale-running runs in manifest
3. Append timestamped entry to OPERATIONS_LOG.md
4. Launch orchestrator with nohup
5. Print PID and manifest state

We test via --dry-run flag which performs steps 1-3 and 5 but skips the
actual nohup launch.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESTART_SCRIPT = PROJECT_ROOT / "scripts" / "restart.sh"


def _make_manifest_json(runs: dict, mode: str = "full") -> str:
    """Build a manifest JSON string."""
    return json.dumps({"mode": mode, "runs": runs}, indent=2)


def _make_run_dict(run_id: str, status: str = "pending") -> dict:
    return {
        "run_id": run_id,
        "stage": "ablation",
        "dataset": "D1",
        "model_key": "test",
        "model_id": "ollama/test",
        "config": "A0",
        "seed": 0,
        "status": status,
        "slot_type": "ollama",
        "max_events": None,
        "max_event_words": None,
        "started_at": None,
        "finished_at": None,
        "error": None,
        "result_file": "results/test.csv",
        "metrics": None,
    }


@pytest.fixture
def experiment_dir(tmp_path):
    """Create a minimal experiment directory with manifest and ops log."""
    # Create directory structure
    results_dir = tmp_path / "results" / "full"
    results_dir.mkdir(parents=True)
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    # Create src package (needed for the Python one-liner)
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "__init__.py").write_text("")

    # Copy manifest.py and ops_log.py
    import shutil
    shutil.copy(PROJECT_ROOT / "src" / "manifest.py", src_dir / "manifest.py")
    shutil.copy(PROJECT_ROOT / "src" / "config.py", src_dir / "config.py")
    shutil.copy(PROJECT_ROOT / "src" / "ops_log.py", src_dir / "ops_log.py")

    # Create manifest with mixed states
    runs = {
        "run-a": _make_run_dict("run-a", "done"),
        "run-b": _make_run_dict("run-b", "failed"),
        "run-c": _make_run_dict("run-c", "failed"),
        "run-d": _make_run_dict("run-d", "running"),
        "run-e": _make_run_dict("run-e", "pending"),
    }
    manifest_path = results_dir / "manifest.json"
    manifest_path.write_text(_make_manifest_json(runs))

    # Create ops log
    ops_log = tmp_path / "OPERATIONS_LOG.md"
    ops_log.write_text(
        "# Neural Router Experiment Operations Log\n\n"
        "## Status Summary\n\n"
        "| Date | Done | Running | Pending | Failed | Total | Notes |\n"
        "|---|---|---|---|---|---|---|\n\n"
        "## Operations\n\n"
    )

    return tmp_path


@pytest.mark.skipif(
    not RESTART_SCRIPT.exists(),
    reason="restart.sh not yet implemented",
)
class TestRestartScript:

    def test_script_is_executable(self):
        """restart.sh should have execute permission."""
        assert os.access(RESTART_SCRIPT, os.X_OK)

    def test_dry_run_resets_manifest(self, experiment_dir):
        """Dry run resets failed and running entries to pending."""
        result = subprocess.run(
            [
                str(RESTART_SCRIPT),
                "--dry-run",
                "--mode", "full",
                "--results-root", str(experiment_dir / "results"),
                "--project-root", str(experiment_dir),
            ],
            capture_output=True,
            text=True,
            cwd=str(experiment_dir),
            env={**os.environ, "PATH": os.environ["PATH"]},
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

        # Verify manifest was updated
        manifest_path = experiment_dir / "results" / "full" / "manifest.json"
        data = json.loads(manifest_path.read_text())
        # run-b and run-c were failed -> pending
        assert data["runs"]["run-b"]["status"] == "pending"
        assert data["runs"]["run-c"]["status"] == "pending"
        # run-d was running -> pending
        assert data["runs"]["run-d"]["status"] == "pending"
        # run-a stays done
        assert data["runs"]["run-a"]["status"] == "done"

    def test_dry_run_appends_ops_log(self, experiment_dir):
        """Dry run appends a restart entry to OPERATIONS_LOG.md."""
        result = subprocess.run(
            [
                str(RESTART_SCRIPT),
                "--dry-run",
                "--mode", "full",
                "--results-root", str(experiment_dir / "results"),
                "--project-root", str(experiment_dir),
            ],
            capture_output=True,
            text=True,
            cwd=str(experiment_dir),
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

        ops_log = experiment_dir / "OPERATIONS_LOG.md"
        content = ops_log.read_text()
        assert "Automated restart" in content

    def test_dry_run_prints_state(self, experiment_dir):
        """Dry run outputs manifest state summary."""
        result = subprocess.run(
            [
                str(RESTART_SCRIPT),
                "--dry-run",
                "--mode", "full",
                "--results-root", str(experiment_dir / "results"),
                "--project-root", str(experiment_dir),
            ],
            capture_output=True,
            text=True,
            cwd=str(experiment_dir),
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        # Should print done/pending/failed counts
        assert "done" in result.stdout.lower()
        assert "pending" in result.stdout.lower()

    def test_dry_run_reports_orphan_check(self, experiment_dir):
        """Dry run checks for orphaned run_one.py processes."""
        result = subprocess.run(
            [
                str(RESTART_SCRIPT),
                "--dry-run",
                "--mode", "full",
                "--results-root", str(experiment_dir / "results"),
                "--project-root", str(experiment_dir),
            ],
            capture_output=True,
            text=True,
            cwd=str(experiment_dir),
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        # Should mention checking for orphaned run_one.py processes
        assert "run_one" in result.stdout.lower() or "orphan" in result.stdout.lower()

    def test_accepts_slot_args(self, experiment_dir):
        """Script accepts --api-slots and --ollama-slots."""
        result = subprocess.run(
            [
                str(RESTART_SCRIPT),
                "--dry-run",
                "--mode", "full",
                "--results-root", str(experiment_dir / "results"),
                "--project-root", str(experiment_dir),
                "--api-slots", "3",
                "--ollama-slots", "2",
            ],
            capture_output=True,
            text=True,
            cwd=str(experiment_dir),
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
