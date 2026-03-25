"""Tests for the no-fork watchdog script (L44).

The watchdog must:
1. Read /proc/loadavg directly (no ps, grep, wc, pipe)
2. Kill python/python3/grep when load exceeds threshold
3. NOT exit after killing (keep watching for respawns)
4. Log actions to the logfile
"""
import os
import subprocess
from pathlib import Path

WATCHDOG = Path(__file__).parent.parent / "scripts" / "remote" / "watchdog.sh"


def test_watchdog_exists():
    assert WATCHDOG.exists()


def test_watchdog_is_executable():
    assert os.access(WATCHDOG, os.X_OK)


def test_watchdog_no_ps_command():
    """Watchdog must NOT use ps (spawns subprocess under load)."""
    content = WATCHDOG.read_text()
    # Allow 'ps' in comments but not as a command
    lines = [l for l in content.splitlines() if not l.strip().startswith('#')]
    code = '\n'.join(lines)
    assert ' ps ' not in code, "Watchdog uses 'ps' which forks under load"


def test_watchdog_no_grep_pipe():
    """Watchdog must NOT pipe through grep (spawns subprocess)."""
    content = WATCHDOG.read_text()
    lines = [l for l in content.splitlines() if not l.strip().startswith('#')]
    code = '\n'.join(lines)
    assert '| grep' not in code, "Watchdog pipes through grep which forks"
    assert '| wc' not in code, "Watchdog pipes through wc which forks"


def test_watchdog_reads_proc_loadavg():
    """Watchdog must read /proc/loadavg directly."""
    content = WATCHDOG.read_text()
    assert '/proc/loadavg' in content


def test_watchdog_kills_grep():
    """Watchdog must kill stale grep processes (from old watchdog)."""
    content = WATCHDOG.read_text()
    assert 'grep' in content and 'killall' in content


def test_watchdog_does_not_exit_after_kill():
    """Watchdog must keep running after killing (processes may respawn)."""
    content = WATCHDOG.read_text()
    # After killall, should sleep and continue, not 'break' or 'exit'
    lines = content.splitlines()
    for i, line in enumerate(lines):
        if 'killall' in line and not line.strip().startswith('#'):
            # Check next few non-empty lines don't have 'break' or 'exit'
            remaining = [l.strip() for l in lines[i+1:i+5] if l.strip() and not l.strip().startswith('#')]
            assert 'break' not in ' '.join(remaining), "Watchdog exits after kill — should keep watching"


def test_watchdog_has_configurable_threshold():
    content = WATCHDOG.read_text()
    assert 'THRESHOLD' in content


def test_watchdog_shellcheck_clean():
    """Watchdog should pass basic bash syntax check."""
    result = subprocess.run(
        ['bash', '-n', str(WATCHDOG)],
        capture_output=True, text=True
    )
    assert result.returncode == 0, f"Syntax error: {result.stderr}"
