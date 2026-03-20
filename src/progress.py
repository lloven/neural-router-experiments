"""Fine-grained progress tracking for experiment stages.

Each running script writes a small JSON file to ``results/{mode}/.progress/``
on every event batch, parameter value, or scale point.  The monitor reads
these files to show within-stage progress and ETAs.

Usage in scripts::

    from src.progress import ProgressTracker

    tracker = ProgressTracker("ablation-D1-sonnet", mode="integration_smoke")
    tracker.update(
        current_task="A3 seed=42",
        items_done=450,
        items_total=1000,
        tasks_done=2,
        tasks_total=5,
        detail="Processing events 451-500/1000",
    )
    tracker.finish()   # marks stage as complete
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Writes fine-grained progress to a JSON file for the monitor."""

    def __init__(
        self,
        stage_name: str,
        mode: str = "integration_smoke",
        results_root: str | Path = "results",
    ):
        self.stage_name = stage_name
        self.mode = mode
        self._dir = Path(results_root) / mode / ".progress"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / f"{stage_name}.json"
        self._start_time = time.time()
        self._last_write = 0.0
        self._min_interval = 2.0  # don't write more often than every 2s

    def update(
        self,
        current_task: str = "",
        items_done: int = 0,
        items_total: int = 0,
        tasks_done: int = 0,
        tasks_total: int = 0,
        detail: str = "",
        force: bool = False,
        **extra: Any,
    ) -> None:
        """Write a progress update.

        Args:
            current_task: Current subtask label (e.g., "A3 seed=42").
            items_done: Fine-grained items completed within current task
                        (e.g., events processed).
            items_total: Total fine-grained items in current task.
            tasks_done: Coarse tasks completed (e.g., config×seed pairs).
            tasks_total: Total coarse tasks.
            detail: Human-readable status string.
            force: Write even if min_interval hasn't elapsed.
            **extra: Additional key-value pairs to include.
        """
        now = time.time()
        if not force and (now - self._last_write) < self._min_interval:
            return

        elapsed = now - self._start_time

        # ETA calculation
        eta_seconds = -1
        if tasks_total > 0 and tasks_done > 0:
            # Weighted progress: completed tasks + fraction of current task
            if items_total > 0:
                current_fraction = items_done / items_total
            else:
                current_fraction = 0.0
            effective_done = tasks_done + current_fraction
            if effective_done > 0:
                rate = elapsed / effective_done
                eta_seconds = rate * (tasks_total - effective_done)
        elif items_total > 0 and items_done > 0:
            rate = elapsed / items_done
            eta_seconds = rate * (items_total - items_done)

        data = {
            "stage": self.stage_name,
            "mode": self.mode,
            "status": "running",
            "current_task": current_task,
            "items_done": items_done,
            "items_total": items_total,
            "tasks_done": tasks_done,
            "tasks_total": tasks_total,
            "detail": detail,
            "elapsed_s": round(elapsed, 1),
            "eta_s": round(eta_seconds, 1) if eta_seconds >= 0 else -1,
            "started_at": datetime.fromtimestamp(self._start_time).isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        data.update(extra)

        try:
            self._path.write_text(json.dumps(data, indent=2))
            self._last_write = now
        except OSError as e:
            logger.debug(f"Could not write progress: {e}")

    def finish(self, tasks_done: int = 0, tasks_total: int = 0) -> None:
        """Mark the stage as complete."""
        data = {
            "stage": self.stage_name,
            "mode": self.mode,
            "status": "done",
            "tasks_done": tasks_done or tasks_total,
            "tasks_total": tasks_total,
            "elapsed_s": round(time.time() - self._start_time, 1),
            "started_at": datetime.fromtimestamp(self._start_time).isoformat(),
            "finished_at": datetime.now().isoformat(),
        }
        try:
            self._path.write_text(json.dumps(data, indent=2))
        except OSError:
            pass


class SidecarProgressCallback:
    """Writes progress to a sidecar JSON file for the monitor to read.

    Unlike ProgressTracker (which uses stage_name), this uses run_id
    to match the manifest's run entries.  The file is written to
    ``results/{mode}/.progress/{run_id}.json``.
    """

    def __init__(
        self,
        run_id: str,
        mode: str,
        results_root: str | Path = "results",
    ):
        self.run_id = run_id
        self.mode = mode
        self._dir = Path(results_root) / mode / ".progress"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / f"{run_id}.json"
        self._start = time.time()
        self._last_write = 0.0
        self._min_interval = 2.0

    def update(
        self,
        current_task: str = "",
        items_done: int = 0,
        items_total: int = 0,
        tasks_done: int = 0,
        tasks_total: int = 0,
        detail: str = "",
        force: bool = False,
        **extra: Any,
    ) -> None:
        """Write a progress update.  Same signature as ProgressTracker.update()."""
        now = time.time()
        if not force and (now - self._last_write) < self._min_interval:
            return

        elapsed = now - self._start

        # ETA calculation (same logic as ProgressTracker)
        eta_seconds = -1
        if tasks_total > 0 and tasks_done > 0:
            if items_total > 0:
                current_fraction = items_done / items_total
            else:
                current_fraction = 0.0
            effective_done = tasks_done + current_fraction
            if effective_done > 0:
                rate = elapsed / effective_done
                eta_seconds = rate * (tasks_total - effective_done)
        elif items_total > 0 and items_done > 0:
            rate = elapsed / items_done
            eta_seconds = rate * (items_total - items_done)

        data = {
            "run_id": self.run_id,
            "mode": self.mode,
            "status": "running",
            "current_task": current_task,
            "items_done": items_done,
            "items_total": items_total,
            "tasks_done": tasks_done,
            "tasks_total": tasks_total,
            "detail": detail,
            "elapsed_s": round(elapsed, 1),
            "eta_s": round(eta_seconds, 1) if eta_seconds >= 0 else -1,
            "started_at": datetime.fromtimestamp(self._start).isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        data.update(extra)

        try:
            self._path.write_text(json.dumps(data, indent=2))
            self._last_write = now
        except OSError as e:
            logger.debug(f"Could not write sidecar progress: {e}")

    def finish(self, tasks_done: int = 0, tasks_total: int = 0) -> None:
        """Mark the run as complete."""
        data = {
            "run_id": self.run_id,
            "mode": self.mode,
            "status": "done",
            "tasks_done": tasks_done or tasks_total,
            "tasks_total": tasks_total,
            "elapsed_s": round(time.time() - self._start, 1),
            "started_at": datetime.fromtimestamp(self._start).isoformat(),
            "finished_at": datetime.now().isoformat(),
        }
        try:
            self._path.write_text(json.dumps(data, indent=2))
        except OSError:
            pass


def read_progress(mode: str, results_root: str | Path = "results") -> dict[str, dict]:
    """Read all progress files for a mode.

    Returns:
        Dict mapping stage_name -> progress data dict.
    """
    progress_dir = Path(results_root) / mode / ".progress"
    result = {}
    if not progress_dir.exists():
        return result
    for p in progress_dir.glob("*.json"):
        try:
            data = json.loads(p.read_text())
            result[data.get("stage", p.stem)] = data
        except (json.JSONDecodeError, OSError):
            pass
    return result
