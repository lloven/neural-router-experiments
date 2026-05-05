"""Static checks: each smoke script accepts SMOKE_LEVEL={integration,full}
and the integration level uses --max-events <= 100 (per L23 — proper
multi-level smoke testing, not full-corpus single points)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CROSS = ROOT / "scripts/slurm/mahti_crossover_smoke.sh"
QOE = ROOT / "scripts/slurm/mahti_qoe_smoke.sh"
COSTVAL = ROOT / "scripts/slurm/mahti_cost_validation.sh"


def _shell(p: Path) -> str:
    return p.read_text() if p.exists() else ""


def test_crossover_smoke_supports_smoke_level():
    s = _shell(CROSS)
    assert "SMOKE_LEVEL" in s, "crossover smoke must take a SMOKE_LEVEL parameter"
    assert "integration" in s and "full" in s, \
        "crossover smoke must define integration and full levels"


def test_crossover_integration_uses_max_events():
    """Integration level must pass --max-events with a small value (<=100)."""
    s = _shell(CROSS)
    import re
    # Match the integration branch text from `integration)` up to the next `;;`
    integ_block = re.search(r"integration\)(.*?);;", s, re.DOTALL)
    assert integ_block, f"could not locate integration case branch in {CROSS}"
    block = integ_block.group(1)
    m = re.search(r"MAX_E\s*=\s*(\d+)", block) or re.search(r"--max-events\s+(\d+)", block)
    assert m, f"integration block must set MAX_E/--max-events: {block!r}"
    assert int(m.group(1)) <= 100, f"integration MAX_E should be <= 100: {m.group(1)}"


def test_qoe_smoke_supports_smoke_level():
    s = _shell(QOE)
    assert "SMOKE_LEVEL" in s
    assert "integration" in s and "full" in s


def test_qoe_integration_uses_max_events():
    s = _shell(QOE)
    import re
    integ_block = re.search(r"integration\)(.*?);;", s, re.DOTALL)
    assert integ_block, f"could not locate integration case branch in {QOE}"
    block = integ_block.group(1)
    m = re.search(r"MAX_E\s*=\s*(\d+)", block) or re.search(r"--max-events\s+(\d+)", block)
    assert m, f"integration block must set MAX_E/--max-events: {block!r}"
    assert int(m.group(1)) <= 100


def test_cost_validation_smoke_supports_smoke_level():
    """Cost-validation already has SMOKE_LEVEL — keep it locked in."""
    s = _shell(COSTVAL)
    assert "SMOKE_LEVEL" in s
    assert "integration" in s and "full" in s


def test_smoke_scripts_have_valid_bash_syntax():
    """`bash -n` must succeed on all three smoke scripts."""
    import subprocess
    for script in (CROSS, QOE, COSTVAL):
        result = subprocess.run(["bash", "-n", str(script)],
                                capture_output=True, text=True)
        assert result.returncode == 0, \
            f"bash -n failed on {script}:\n{result.stderr}"
