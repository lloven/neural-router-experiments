"""Verify scripts/run_crossover.py supports --max-events for unit-smoke runs."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_run_crossover_help_lists_max_events():
    out = subprocess.run(
        [sys.executable, "scripts/run_crossover.py", "--help"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    assert "--max-events" in out.stdout, out.stdout
