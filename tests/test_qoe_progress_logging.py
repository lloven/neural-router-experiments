"""L63 regression: run_qoe.py must emit per-cell progress to .progress.json
during the sweep. Atomic writes (no partial visibility); readable mid-run.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_write_progress_creates_atomic_file(tmp_path):
    from scripts.run_qoe import _write_progress
    payload = {"phase": "running", "cell": "homogeneous/qwen-7b/seed42",
               "completed_cells": 3, "total_cells": 21}
    _write_progress(tmp_path, payload)
    p = tmp_path / ".progress.json"
    assert p.exists()
    data = json.loads(p.read_text())
    assert data["phase"] == "running"
    assert data["cell"] == "homogeneous/qwen-7b/seed42"
    assert data["completed_cells"] == 3
    assert data["total_cells"] == 21
    assert "ts" in data  # timestamp added automatically


def test_write_progress_overwrites_previous(tmp_path):
    """Each call should fully replace the previous payload (not append)."""
    from scripts.run_qoe import _write_progress
    _write_progress(tmp_path, {"phase": "starting", "completed_cells": 0})
    _write_progress(tmp_path, {"phase": "running", "completed_cells": 5})
    data = json.loads((tmp_path / ".progress.json").read_text())
    assert data["completed_cells"] == 5
    assert data["phase"] == "running"


def test_write_progress_does_not_leave_tmp_files(tmp_path):
    """Atomic write via tmp+rename — no .tmp leftovers in output_dir."""
    from scripts.run_qoe import _write_progress
    for i in range(5):
        _write_progress(tmp_path, {"phase": "running", "completed_cells": i})
    leftovers = [p for p in tmp_path.iterdir()
                 if p.name.startswith(".progress_") and p.name.endswith(".tmp")]
    assert leftovers == [], f"unexpected tmp files: {leftovers}"
