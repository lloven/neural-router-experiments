"""Static checks on scripts/slurm/mahti_qoe_openweight_d1.sh.

QoE on three Qwen 2.5 open-weight tiers — validates Contribution #4 of the
manuscript without Anthropic spend.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "slurm" / "mahti_qoe_openweight_d1.sh"
SMOKE = REPO / "scripts" / "slurm" / "mahti_qoe_smoke.sh"


def test_full_script_exists():
    assert SCRIPT.exists(), f"missing: {SCRIPT}"


def test_full_script_bash_syntax():
    proc = subprocess.run(
        ["bash", "-n", str(SCRIPT)], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr


def test_three_qwen_tiers_specified():
    text = SCRIPT.read_text()
    for m in ["qwen2.5:1.5b", "qwen2.5:7b", "qwen2.5:32b"]:
        assert m in text, f"missing backend: {m}"


def test_partition_is_gpusmall():
    text = SCRIPT.read_text()
    assert "--partition=gpusmall" in text
    assert "--gres=gpu:a100:1" in text


def test_invokes_run_qoe():
    text = SCRIPT.read_text()
    assert "scripts/run_qoe.py" in text


def test_writes_to_qoe_by_task():
    text = SCRIPT.read_text()
    assert "results/full/qoe/by_task" in text


def test_smoke_script_exists():
    assert SMOKE.exists(), f"missing: {SMOKE}"


def test_smoke_bash_syntax():
    proc = subprocess.run(
        ["bash", "-n", str(SMOKE)], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr


def test_smoke_uses_gputest_partition():
    text = SMOKE.read_text()
    assert "--partition=gputest" in text


def test_smoke_validates_csv_with_l30_check():
    """Per L30, a smoke must verify CSV CONTENT (data rows present), not
    just file existence."""
    text = SMOKE.read_text()
    assert re.search(r"\$\{?SMOKE_CSV\}?", text), \
        "smoke must reference its CSV output"
    assert "wc -l" in text or "head -" in text or "tail -" in text, \
        "smoke must inspect CSV contents"
