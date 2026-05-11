"""Verify scripts/run_qoe.py supports --max-events for unit-smoke runs.

Required because the 2026-05-04 timeout-failed smokes used full-corpus
single-points instead of progressive scaling. With --max-events the
unit smoke can run 8 events in seconds; integration 50; full corpus."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_run_qoe_help_lists_max_events():
    """--max-events must appear in --help (so SLURM can parameterise it)."""
    out = subprocess.run(
        [sys.executable, "scripts/run_qoe.py", "--help"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    assert "--max-events" in out.stdout, out.stdout
