"""Static checks on scripts/slurm/mahti_crossover_d1.sh.

The crossover SLURM wrapper is shipped to scratch and submitted via sbatch;
we cannot unit-test the GPU run, but we CAN check the wrapper's bash syntax
and that it sets the variables the parametric pattern requires.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "slurm" / "mahti_crossover_d1.sh"


def test_script_exists():
    assert SCRIPT.exists(), f"missing: {SCRIPT}"


def test_bash_syntax_is_valid():
    proc = subprocess.run(
        ["bash", "-n", str(SCRIPT)], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr


def test_sbatch_directives_are_present():
    text = SCRIPT.read_text()
    for d in [
        "--account=project_2018951",
        "--partition=gpusmall",
        "--gres=gpu:a100:1",
        "--array=",
    ]:
        assert d in text, f"missing #SBATCH directive: {d}"


def test_invokes_run_crossover():
    text = SCRIPT.read_text()
    assert "scripts/run_crossover.py" in text, \
        "must invoke the existing run_crossover.py runner"


def test_uses_max_context_tokens_4096():
    text = SCRIPT.read_text()
    assert re.search(r"--max-context-tokens\s+4096", text), \
        "must enforce W=4096 token budget on the matching prompt"


def test_array_indexes_full_36_grid():
    """6 sub-volumes × 2 configs × 3 seeds = 36 array tasks."""
    text = SCRIPT.read_text()
    assert re.search(r"#SBATCH\s+--array=0-35", text), \
        "expected --array=0-35 (36 tasks)"


def test_uses_array_task_id_for_index_decode():
    text = SCRIPT.read_text()
    assert "$SLURM_ARRAY_TASK_ID" in text, \
        "must derive (sub_volume, config, seed) from SLURM_ARRAY_TASK_ID"


def test_writes_per_task_output_dir():
    """Each array task should write to a unique by_task subdirectory so
    parallel tasks don't race on the same CSV."""
    text = SCRIPT.read_text()
    assert "by_task" in text and "$TAG" in text, \
        "must write to results/full/crossover/by_task/$TAG"
