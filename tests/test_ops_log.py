"""Tests for the ops_log module (auto-logging to OPERATIONS_LOG.md).

TDD RED phase: these tests define the expected interface for src/ops_log.py.
"""

from __future__ import annotations

import re
from pathlib import Path

from src.ops_log import (
    log_api_limit,
    log_milestone,
    log_shutdown,
    log_startup,
)


def _make_manifest_state(done: int = 0, pending: int = 0, failed: int = 0,
                         running: int = 0) -> dict:
    """Helper to build a manifest state dict."""
    return {
        "done": done,
        "pending": pending,
        "failed": failed,
        "running": running,
        "total": done + pending + failed + running,
    }


# -- Feature 2 tests --------------------------------------------------------

class TestOpsLogStartup:

    def test_log_startup_appends_entry(self, tmp_path):
        """Starting the orchestrator appends a startup entry to ops log."""
        ops_log = tmp_path / "OPERATIONS_LOG.md"
        # Create a minimal ops log with the expected header
        ops_log.write_text(
            "# Neural Router Experiment Operations Log\n\n"
            "## Status Summary\n\n"
            "| Date | Done | Running | Pending | Failed | Total | Notes |\n"
            "|---|---|---|---|---|---|---|\n\n"
            "## Operations\n\n"
        )
        state = _make_manifest_state(done=50, pending=200, failed=10, running=2)
        slot_config = {"api_slots": 2, "ollama_slots": 1}

        log_startup(ops_log, state, slot_config)

        content = ops_log.read_text()
        # Should have a status summary row
        assert "| 50 |" in content
        assert "| 200 |" in content
        # Should have an operations entry with "Startup"
        assert "Startup" in content
        assert "api_slots=2" in content
        assert "ollama_slots=1" in content

    def test_ops_log_created_if_missing(self, tmp_path):
        """If OPERATIONS_LOG.md doesn't exist, create it with header."""
        ops_log = tmp_path / "OPERATIONS_LOG.md"
        assert not ops_log.exists()

        state = _make_manifest_state(done=0, pending=100)
        slot_config = {"api_slots": 1, "ollama_slots": 1}

        log_startup(ops_log, state, slot_config)

        assert ops_log.exists()
        content = ops_log.read_text()
        # Should contain the standard header
        assert "# Neural Router Experiment Operations Log" in content
        assert "## Status Summary" in content
        assert "## Operations" in content
        # And the startup entry
        assert "Startup" in content


class TestOpsLogMilestone:

    def test_log_milestone_at_25_runs(self, tmp_path):
        """After 25 completions, a milestone line appears in ops log."""
        ops_log = tmp_path / "OPERATIONS_LOG.md"
        ops_log.write_text(
            "# Neural Router Experiment Operations Log\n\n"
            "## Status Summary\n\n"
            "| Date | Done | Running | Pending | Failed | Total | Notes |\n"
            "|---|---|---|---|---|---|---|\n\n"
            "## Operations\n\n"
        )
        state = _make_manifest_state(done=25, pending=347)

        log_milestone(ops_log, state)

        content = ops_log.read_text()
        # Should add a row to the status summary table
        assert "| 25 |" in content
        assert "Milestone: 25 done" in content


class TestOpsLogApiLimit:

    def test_log_api_limit_hit(self, tmp_path):
        """When a run fails with 'usage limit', ops log records it."""
        ops_log = tmp_path / "OPERATIONS_LOG.md"
        ops_log.write_text(
            "# Neural Router Experiment Operations Log\n\n"
            "## Status Summary\n\n"
            "| Date | Done | Running | Pending | Failed | Total | Notes |\n"
            "|---|---|---|---|---|---|---|\n\n"
            "## Operations\n\n"
        )

        log_api_limit(ops_log, "ablation__D1__A0__seed42__haiku", "usage limit exceeded")

        content = ops_log.read_text()
        assert "API limit" in content or "api limit" in content.lower()
        assert "ablation__D1__A0__seed42__haiku" in content
        assert "usage limit" in content.lower()


class TestOpsLogShutdown:

    def test_log_shutdown_appends_entry(self, tmp_path):
        """Clean shutdown appends final state to ops log."""
        ops_log = tmp_path / "OPERATIONS_LOG.md"
        ops_log.write_text(
            "# Neural Router Experiment Operations Log\n\n"
            "## Status Summary\n\n"
            "| Date | Done | Running | Pending | Failed | Total | Notes |\n"
            "|---|---|---|---|---|---|---|\n\n"
            "## Operations\n\n"
        )
        state = _make_manifest_state(done=372, pending=0, failed=0)

        log_shutdown(ops_log, state, reason="clean exit")

        content = ops_log.read_text()
        assert "Shutdown" in content
        assert "clean exit" in content
        # Status summary row should be present
        assert "| 372 |" in content

    def test_log_shutdown_with_error_reason(self, tmp_path):
        """Shutdown due to error records the reason."""
        ops_log = tmp_path / "OPERATIONS_LOG.md"
        ops_log.write_text(
            "# Neural Router Experiment Operations Log\n\n"
            "## Status Summary\n\n"
            "| Date | Done | Running | Pending | Failed | Total | Notes |\n"
            "|---|---|---|---|---|---|---|\n\n"
            "## Operations\n\n"
        )
        state = _make_manifest_state(done=100, pending=200, failed=72)

        log_shutdown(ops_log, state, reason="SIGTERM received")

        content = ops_log.read_text()
        assert "SIGTERM" in content


class TestOpsLogIntegration:

    def test_multiple_entries_preserve_structure(self, tmp_path):
        """Multiple log calls preserve the markdown structure."""
        ops_log = tmp_path / "OPERATIONS_LOG.md"

        # Startup (creates file)
        state = _make_manifest_state(done=0, pending=372)
        log_startup(ops_log, state, {"api_slots": 2, "ollama_slots": 1})

        # Milestone
        state = _make_manifest_state(done=25, pending=347)
        log_milestone(ops_log, state)

        # API limit
        log_api_limit(ops_log, "run-xyz", "usage limit exceeded")

        # Shutdown
        state = _make_manifest_state(done=50, pending=300, failed=22)
        log_shutdown(ops_log, state, reason="clean exit")

        content = ops_log.read_text()

        # Structure checks
        assert content.index("## Status Summary") < content.index("## Operations")

        # Count status rows (excluding header row)
        table_lines = [
            line for line in content.split("\n")
            if line.startswith("|") and "Date" not in line and "---" not in line
        ]
        # At least 3 status rows: startup, milestone, shutdown
        assert len(table_lines) >= 3
