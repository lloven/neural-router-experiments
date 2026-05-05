# Validation Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the three load-bearing experimental gaps in `Manuscripts/Neural Router (Elsevier FGCS)` so the manuscript's stated contributions C2b (CoverAndMerge benefit at low W), C5 (QoE-based heterogeneous backend assignment), and the seventh baseline (BART-MNLI zero-shot) are all empirically backed.

**Architecture:** Three independent experiments, each a thin shell over existing implementations in `src/`. (i) Local CPU runner for `_run_zero_shot()` on D1/D2/D3. (ii) Mahti SLURM array calling existing `scripts/run_crossover.py` over an `|S| × {A0,A4} × 3 seeds` grid with `max_context_tokens=4096`. (iii) Single Mahti job calling existing `scripts/run_qoe.py` with three Qwen 2.5 backends (1.5B/7B/32B) sharing one Ollama instance via hot-swap, on D1, 5 seeds × 3 strategies × 3 weight presets.

**Tech Stack:** Python 3.11/3.12 (laptop venv / CSC `pytorch/2.9` module), pytest, transformers (BART), Ollama for Qwen, SLURM `gputest` (smoke) and `gpusmall` (full) on Mahti, existing experiment harness in `src/router.py`, `src/baselines.py`, `src/qoe.py`.

**Estimated cost:** ~385 BU CSC (Mahti gpusmall, 16 BU/GPU-h, with 1.5× safety) — ~0.16 % of the 240 k envelope. Zero Anthropic spend.

---

## Scope Check

These three experiments target three independently-valid manuscript claims and one baseline. They are independent (no shared infrastructure beyond Ollama on Mahti) and each produces standalone results. Splitting into sub-projects is unnecessary; doing them in sequence in one plan keeps the operational and logging context unified.

## File Structure

| Path | Status | Responsibility |
|---|---|---|
| `scripts/run_baseline_zero_shot.py` | **CREATE** | Thin CLI: load dataset → call `run_baseline("zero_shot", …)` → write CSV with columns matching other baselines |
| `tests/test_run_baseline_zero_shot.py` | **CREATE** | TDD: validate CLI argument parsing, dataset selection, output-CSV shape |
| `scripts/slurm/mahti_crossover_d1.sh` | **CREATE** | SLURM array (gpusmall) calling `scripts/run_crossover.py` once per `(sub_volume, config, seed)` task |
| `scripts/slurm/mahti_crossover_smoke.sh` | **CREATE** | gputest smoke: 1 task at \|S\|=200, A0+A4, seed 42 |
| `scripts/slurm/mahti_qoe_openweight_d1.sh` | **CREATE** | Single Mahti job (gpusmall, longer walltime) calling `scripts/run_qoe.py` for three Qwen 2.5 tiers |
| `scripts/slurm/mahti_qoe_smoke.sh` | **CREATE** | gputest smoke: minimal QoE run with 1 seed, homogeneous + qoe-optimised |
| `analysis/make_figures.py` | **MODIFY** | Add `fig_crossover()` and `fig_qoe()` after their data lands; skip BART-MNLI (re-uses existing `pareto.pdf` + `tab:cross-dataset` table population) |
| `Manuscripts/Neural Router (Elsevier FGCS)/figs/crossover.pdf` | **GENERATE** | After crossover data lands |
| `Manuscripts/Neural Router (Elsevier FGCS)/figs/qoe_pareto.pdf` | **GENERATE** | After QoE data lands |
| `Manuscripts/Neural Router (Elsevier FGCS)/txt/Results.tex` | **MODIFY** | Replace the "deferred to future work" note in §5.7 with a real `\includegraphics{figs/crossover.pdf}`; populate the `--` rows of `tab:qoe`; restore the BART-MNLI rows in `tab:ablation` and `tab:cross-dataset`. |
| `OPERATIONS_LOG.md` (local + Puhti + Mahti) | **APPEND** | Log every action with timestamp, command, reason, outcome, BU cost (per L55 + `feedback_remote_logging.md`) |

## Pre-flight Checks

- [ ] **PF1: CSC reachable**

  Run: `ssh -o BatchMode=yes mahti 'hostname; date'` and `ssh -o BatchMode=yes puhti 'hostname; date'`
  Expected: both return host + date with exit 0. If not, surface error and stop.

- [ ] **PF2: Mahti billing rate is still 16 BU/GPU-h on gpusmall**

  Run: `ssh mahti 'sacct -j 6585620 -X -P --format=JobID,Elapsed,ElapsedRaw,AllocTRES'`
  Expected: shows `billing=16,...` for gpusmall. (Confirms the calibration in `feedback_csc_billing_estimates.md`.)

- [ ] **PF3: laptop venv has transformers + bart-large-mnli reachable**

  Run: `cd Experiments/neural-router && .venv/bin/python -c "from transformers import pipeline; print('ok')"`
  Expected: `ok`. Model download happens on first use; this just verifies the import.

---

## Experiment 1 — BART-MNLI zero-shot baseline (laptop, CPU)

**Why:** Manuscript abstract claims "seven baselines (including zero-shot NLI classification)". Body §4.3 enumerates BART-MNLI as the seventh. `tab:ablation` and `tab:cross-dataset` show only six. Running this restores the claim. Free (CPU on laptop). Expected runtime: minutes per dataset.

### Task 1: Failing test for the CLI runner

**Files:**
- Create: `tests/test_run_baseline_zero_shot.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for scripts/run_baseline_zero_shot.py.

Verifies that the CLI parses dataset selection, writes a CSV with the same
column shape as other baseline CSVs, and writes one row per (dataset, seed=0).
"""
from __future__ import annotations
import sys, subprocess, tempfile, csv
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parent.parent

@pytest.mark.slow
def test_zero_shot_runner_writes_csv_with_expected_columns(tmp_path):
    out = tmp_path / "D1_baseline_zero_shot_results.csv"
    cmd = [
        sys.executable, str(REPO / "scripts" / "run_baseline_zero_shot.py"),
        "--dataset", "D1",
        "--max-events", "8",
        "--output", str(out),
        "--model", "typeform/distilbart-mnli-12-1",  # smaller for tests
    ]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"exit {proc.returncode}, stderr={proc.stderr[-1000:]}"
    assert out.exists(), "CSV not written"
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1, f"expected 1 row, got {len(rows)}"
    expected_cols = {"config", "dataset", "seed", "precision", "recall", "f1", "fpr", "latency_s"}
    assert expected_cols.issubset(rows[0].keys()), f"missing cols: {expected_cols - rows[0].keys()}"
    assert rows[0]["config"] == "baseline_zero_shot"
    assert rows[0]["dataset"] == "D1"
    f1 = float(rows[0]["f1"])
    assert 0 <= f1 <= 1, f"F1 out of range: {f1}"
```

- [ ] **Step 2: Run test to verify it fails (RED)**

Run: `cd Experiments/neural-router && .venv/bin/python -m pytest tests/test_run_baseline_zero_shot.py -m slow -x`
Expected: `FAILED ... no such file or directory: scripts/run_baseline_zero_shot.py` (or a runtime error if the file doesn't exist).

### Task 2: Minimal CLI implementation

**Files:**
- Create: `scripts/run_baseline_zero_shot.py`

- [ ] **Step 3: Write minimal implementation**

```python
#!/usr/bin/env python3
"""Run the BART-MNLI zero-shot classification baseline (Section 4.3, item 7).

Wraps `src.baselines.run_baseline("zero_shot", …)` with a CLI suitable for
laptop CPU execution. Output CSV matches the column shape of the other
baseline CSVs in `results/full/ablation/`.

Usage:
    python scripts/run_baseline_zero_shot.py --dataset D1 \\
        --output results/full/ablation/D1_baseline_zero_shot_results.csv
"""
from __future__ import annotations
import argparse, sys, time, csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_dataset_by_name
from src.baselines import run_baseline
from src.evaluation import evaluate_matches


def main() -> int:
    parser = argparse.ArgumentParser(description="BART-MNLI zero-shot baseline runner")
    parser.add_argument("--dataset", required=True, help="D1 / D2 / D3")
    parser.add_argument("--output", required=True, help="CSV path")
    parser.add_argument("--max-events", type=int, default=None,
                        help="Cap event count (matches the per-dataset cap used elsewhere)")
    parser.add_argument("--model", default="facebook/bart-large-mnli",
                        help="HF model id; pass typeform/distilbart-mnli-12-1 for fast testing")
    parser.add_argument("--cache-dir", default="data")
    args = parser.parse_args()

    ds = load_dataset_by_name(args.dataset, cache_dir=args.cache_dir, max_events=args.max_events)
    t0 = time.time()
    bres = run_baseline("zero_shot", ds, kappa=3, model_name=args.model)
    eres = evaluate_matches(matches=bres.matches, dataset=ds, config_name="baseline_zero_shot")
    eres.latency_s = bres.latency_s

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "config", "dataset", "seed", "precision", "recall", "f1", "fpr",
        "invocations", "compression_ratio", "latency_s",
        "tokens_prompt", "tokens_response", "cost_per_1k",
    ]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({
            "config": "baseline_zero_shot",
            "dataset": ds.short_name,
            "seed": 0,
            "precision": eres.precision.mean,
            "recall": eres.recall.mean,
            "f1": eres.f1.mean,
            "fpr": eres.fpr.mean,
            "invocations": 0,
            "compression_ratio": "n/a",
            "latency_s": eres.latency_s,
            "tokens_prompt": 0,
            "tokens_response": 0,
            "cost_per_1k": 0,
        })
    print(f"done D={ds.short_name} F1={eres.f1.mean:.4f} P={eres.precision.mean:.4f} "
          f"R={eres.recall.mean:.4f} latency={eres.latency_s:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify GREEN**

Run: `cd Experiments/neural-router && .venv/bin/python -m pytest tests/test_run_baseline_zero_shot.py -m slow -x`
Expected: `1 passed` with valid CSV containing one data row, F1 in [0, 1].

### Task 3: Run on D1, D2, D3 (laptop CPU, real BART-large)

- [ ] **Step 5: D1 run (full corpus, ~6000 events; expected ~10 min on CPU)**

Run:
```bash
cd Experiments/neural-router && .venv/bin/python scripts/run_baseline_zero_shot.py \
  --dataset D1 \
  --output results/full/ablation/D1_baseline_zero_shot_results.csv \
  --max-events 6000
```
Expected: prints `done D=D1 F1=...` and CSV exists.

- [ ] **Step 6: D2 run (5000-event subsample to match Sonnet protocol; expected ~30-60 min on CPU because of long EUR-Lex documents)**

Run:
```bash
cd Experiments/neural-router && .venv/bin/python scripts/run_baseline_zero_shot.py \
  --dataset D2 \
  --output results/full/ablation/D2_baseline_zero_shot_results.csv \
  --max-events 5000
```
Expected: prints `done D=D2 F1=...`.

- [ ] **Step 7: D3 run (full corpus, ~23K events; expected ~30 min on CPU)**

Run:
```bash
cd Experiments/neural-router && .venv/bin/python scripts/run_baseline_zero_shot.py \
  --dataset D3 \
  --output results/full/ablation/D3_baseline_zero_shot_results.csv
```
Expected: prints `done D=D3 F1=...`.

### Task 4: Wire BART-MNLI rows back into manuscript tables

- [ ] **Step 8: Read each of the three CSVs; format the rows for `tab:ablation` (D1) and `tab:cross-dataset` (D1/D2/D3 columns)**

- [ ] **Step 9: Edit `Manuscripts/Neural Router (Elsevier FGCS)/txt/Results.tex` to insert the "Zero-shot (BART-MNLI)" row in both tables. Restore the abstract+intro "seven baselines" claim's full backing.**

- [ ] **Step 10: Compile-check (per L58 + verification-before-completion skill)**

Run:
```bash
cd "Manuscripts/Neural Router (Elsevier FGCS)" && rm -f main.aux main.log main.bbl && \
  pdflatex -interaction=nonstopmode main.tex >/dev/null && \
  bibtex main >/dev/null && \
  pdflatex -interaction=nonstopmode main.tex >/dev/null && \
  pdflatex -interaction=nonstopmode main.tex >/dev/null && \
  echo "fatal: $(grep -cE '^! ' main.log), undef: $(grep -cE 'Reference.*undefined' main.log)"
```
Expected: `fatal: 0, undef: 0`.

- [ ] **Step 11: Commit (or append to OPERATIONS_LOG; this manuscript dir is Overleaf-synced via Dropbox per CLAUDE.md, no `olcli push` needed)**

---

## Experiment 2 — Crossover empirical sweep (Mahti, Qwen-2.5-7B, W=4096)

**Why:** Manuscript C2b claim: CoverAndMerge / compression saves cost when W is binding. The §5.1 ablation actively shows compression failing at full cloud-LLM context. Without this experiment, the algorithm's only justification is the §3.5 cost-model worked example. Sweep validates the prediction empirically.

**Setting:** D1 (|S|=19 dataset, but we artificially grow |S| by sub-sampling/duplicating subscriptions to {50, 100, 200, 500, 1000, 2000}). Backend Qwen-2.5-7B on Mahti A100 with `max_context_tokens=4096` enforced via `RouterConfig.max_context_tokens` (already in `router.py`). Configs: A0 (raw, must truncate) vs A4 (CoverAndMerge + reunite, compresses-then-truncates). 3 seeds × 6 sub_volumes × 2 configs = 36 array tasks.

`scripts/run_crossover.py` already implements `run_crossover_sweep()`. Need only the SLURM wrapper.

### Task 5: Failing test for the SLURM wrapper

**Files:**
- Create: `tests/test_mahti_crossover_sbatch.py`

- [ ] **Step 12: Write the failing test**

```python
"""Static checks on scripts/slurm/mahti_crossover_d1.sh.

This SLURM wrapper is shipped to scratch and submitted via sbatch; we cannot
unit-test the GPU run, but we CAN check the wrapper's bash syntax and that
it sets the variables the parametric pattern requires.
"""
from pathlib import Path
import subprocess, re

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "slurm" / "mahti_crossover_d1.sh"

def test_script_exists():
    assert SCRIPT.exists()

def test_bash_syntax_is_valid():
    proc = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

def test_sbatch_directives_are_present():
    text = SCRIPT.read_text()
    for d in ["--account=project_2018951", "--partition=gpusmall",
              "--gres=gpu:a100:1", "--array="]:
        assert d in text, f"missing #SBATCH directive: {d}"

def test_invokes_run_crossover():
    text = SCRIPT.read_text()
    assert "scripts/run_crossover.py" in text

def test_uses_max_context_tokens_4096():
    text = SCRIPT.read_text()
    assert re.search(r"--max-context-tokens\s+4096", text), \
        "must enforce W=4096 budget"

def test_writes_distinct_task_dirs():
    text = SCRIPT.read_text()
    # sub_volume + config + seed should appear in the TAG
    assert "$SLURM_ARRAY_TASK_ID" in text
```

- [ ] **Step 13: Run test to verify RED**

Run: `cd Experiments/neural-router && .venv/bin/python -m pytest tests/test_mahti_crossover_sbatch.py -x`
Expected: `FAILED ... script does not exist`.

### Task 6: Minimal SLURM wrapper

**Files:**
- Create: `scripts/slurm/mahti_crossover_d1.sh`

- [ ] **Step 14: Write the script (full code below; treat as one paste, not iterative edits)**

```bash
#!/bin/bash
#SBATCH --job-name=nrouter-cross
#SBATCH --account=project_2018951
#SBATCH --partition=gpusmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1,nvme:100
#SBATCH --time=02:00:00
#SBATCH --array=0-35%4
#SBATCH --output=/scratch/project_2018951/neural-router/logs/cross-%A_%a.out
#SBATCH --error=/scratch/project_2018951/neural-router/logs/cross-%A_%a.err
# =============================================================================
# Crossover empirical sweep (manuscript §5.7 / Experiment 2 of the
# 2026-05-04 validation plan).
#
# Array layout (36 tasks):
#   idx = sub_idx * 6 + cfg_idx * 3 + seed_idx
#   sub_volumes ∈ {50, 100, 200, 500, 1000, 2000} (6 values)
#   configs ∈ {A0, A4} (2 values)
#   seeds ∈ {42, 123, 456} (3 values)
# Total = 6 × 2 × 3 = 36.
#
# Backend: Qwen-2.5-7B (open-weight, no API spend). Context budget enforced
# via --max-context-tokens 4096 in run_crossover.py (RouterConfig field
# `max_context_tokens` is already supported).
#
# Estimated cost (A100 gpusmall = 16 BU/GPU-h, mean ~30 min/task with 1.5×
# safety): 36 × 0.5 × 16 × 1.5 ≈ 432 BU upper bound. p95 worst case
# 36 × 2 × 16 × 1.5 ≈ 1700 BU.
# =============================================================================

set -eo pipefail

SUB_VOLUMES=(50 100 200 500 1000 2000)
CONFIGS=(A0 A4)
SEEDS=(42 123 456)

idx=$SLURM_ARRAY_TASK_ID
sub_idx=$((idx / 6))
cfg_idx=$(( (idx % 6) / 3 ))
seed_idx=$((idx % 3))
SUB=${SUB_VOLUMES[$sub_idx]}
CONFIG=${CONFIGS[$cfg_idx]}
SEED=${SEEDS[$seed_idx]}
TAG="qwen7b_cross_S${SUB}_${CONFIG}_s${SEED}"

NR_ROOT=/scratch/project_2018951/neural-router
OLLAMA=$NR_ROOT/bin/ollama-install/bin/ollama

source /etc/profile.d/lmod.sh 2>/dev/null || source /usr/share/lmod/lmod/init/bash
source /appl/profile/zz-csc-env.sh
module load pytorch/2.9
source $NR_ROOT/venv/py312-neural-router/bin/activate

PORT=$((25000 + SLURM_ARRAY_TASK_ID + 1))
export OLLAMA_MODELS=$NR_ROOT/weights/ollama
export OLLAMA_HOST=127.0.0.1:$PORT
export OLLAMA_API_BASE=http://127.0.0.1:$PORT
export HF_HOME=$NR_ROOT/weights/hf
export TOKENIZERS_PARALLELISM=false

echo "=== ${TAG} (idx $idx) on $(hostname) ==="
echo "SUB=$SUB CONFIG=$CONFIG SEED=$SEED W=4096"
nvidia-smi -L
date

$OLLAMA serve > $NR_ROOT/logs/ollama-cross-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log 2>&1 &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT
for i in $(seq 1 30); do
    curl -sS --max-time 2 http://$OLLAMA_HOST/api/tags >/dev/null 2>&1 && { echo "ollama ready after ${i}s"; break; }
    sleep 1
done

# Defensive pull (qwen2.5:7b should already be cached from prior runs).
if ! $OLLAMA list | grep -q "^qwen2.5:7b"; then
    echo "Pulling qwen2.5:7b ..."
    $OLLAMA pull qwen2.5:7b
fi

cd $NR_ROOT/code
TASK_OUT=$NR_ROOT/code/results/full/crossover/by_task/$TAG
mkdir -p $TASK_OUT

# run_crossover.py iterates SUB_VOLUMES, CONFIGS, SEEDS by default; override
# each via single-element CLI lists so this task runs exactly one point.
python scripts/run_crossover.py \
    --dataset D1 \
    --configs $CONFIG \
    --sub-volumes $SUB \
    --seeds $SEED \
    --max-context-tokens 4096 \
    --llm-model ollama/qwen2.5:7b \
    --output-dir $TASK_OUT \
    --batch-size 50 \
    --llm-timeout 600

echo "=== DONE $TAG ==="
date
```

- [ ] **Step 15: Run static-check tests to verify GREEN**

Run: `cd Experiments/neural-router && .venv/bin/python -m pytest tests/test_mahti_crossover_sbatch.py -x`
Expected: `5 passed`.

### Task 7: Smoke wrapper for crossover

**Files:**
- Create: `scripts/slurm/mahti_crossover_smoke.sh`

- [ ] **Step 16: Write smoke script (single-task gputest, 15 min)**

```bash
#!/bin/bash
#SBATCH --job-name=nrouter-cross-smoke
#SBATCH --account=project_2018951
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1,nvme:50
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project_2018951/neural-router/logs/cross-smoke-%j.out
#SBATCH --error=/scratch/project_2018951/neural-router/logs/cross-smoke-%j.err
# Crossover smoke: 1 sub_volume, 1 config, 1 seed under the W=4K budget.
# Validates the parametric pattern + ollama port allocation + output CSV
# shape before launching the 36-task array.

set -eo pipefail
NR_ROOT=/scratch/project_2018951/neural-router
OLLAMA=$NR_ROOT/bin/ollama-install/bin/ollama

source /etc/profile.d/lmod.sh 2>/dev/null || source /usr/share/lmod/lmod/init/bash
source /appl/profile/zz-csc-env.sh
module load pytorch/2.9
source $NR_ROOT/venv/py312-neural-router/bin/activate

export OLLAMA_MODELS=$NR_ROOT/weights/ollama
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_API_BASE=http://127.0.0.1:11434
export HF_HOME=$NR_ROOT/weights/hf
export TOKENIZERS_PARALLELISM=false

echo "=== crossover smoke $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi -L
date
$OLLAMA serve > $NR_ROOT/logs/ollama-cross-smoke-$SLURM_JOB_ID.log 2>&1 &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT
for i in $(seq 1 30); do
    curl -sS --max-time 2 http://$OLLAMA_HOST/api/tags >/dev/null 2>&1 && break
    sleep 1
done

cd $NR_ROOT/code
SMOKE_OUT=$NR_ROOT/code/results/full/crossover/by_task/smoke
mkdir -p $SMOKE_OUT

python scripts/run_crossover.py \
    --dataset D1 \
    --configs A0 A4 \
    --sub-volumes 200 \
    --seeds 42 \
    --max-context-tokens 4096 \
    --llm-model ollama/qwen2.5:7b \
    --output-dir $SMOKE_OUT \
    --batch-size 50 \
    --llm-timeout 600

# L30 content check.
SMOKE_CSV=$SMOKE_OUT/crossover_D1.csv
if [ -s "$SMOKE_CSV" ] && [ "$(wc -l < "$SMOKE_CSV")" -gt 1 ]; then
    echo "PASS smoke: CSV has data"
    head -3 "$SMOKE_CSV"
else
    echo "FAIL smoke: empty/missing CSV at $SMOKE_CSV"
    exit 1
fi
echo "=== DONE smoke ==="
date
```

- [ ] **Step 17: Bash syntax check**

Run: `cd Experiments/neural-router && bash -n scripts/slurm/mahti_crossover_smoke.sh && echo OK`
Expected: `OK`.

---

## Experiment 3 — QoE on open-weight tiers (Mahti, Qwen 2.5 1.5B/7B/32B)

**Why:** Manuscript Contribution #4 is "QoE-based heterogeneous backend assignment". The proposed mechanism (algorithm CalibrateAndAssign + the QoE score in §3.3) is never empirically demonstrated. Reframing as "open-weight tiers" (1.5B cheap-fast / 7B mid / 32B expensive-accurate) avoids Anthropic spend AND makes a stronger paradigm-independent point.

**Setting:** D1 (the dataset where the LLM-vs-baseline gap is largest, so cluster-difficulty heterogeneity is most visible). Three Qwen 2.5 backends sharing one Ollama instance (hot-swap). Strategies: homogeneous-Qwen-1.5B, homogeneous-Qwen-7B, homogeneous-Qwen-32B, round-robin, qoe-optimised. Three weight presets: accuracy-first (0.7/0.15/0.15), balanced (0.34/0.33/0.33), cost-first (0.15/0.7/0.15). Five seeds.

`scripts/run_qoe.py` already implements `run_qoe_experiment()`. Need: (i) a small modification to accept Ollama backends keyed by model id (already supported via `--backends name:ollama/qwen2.5:7b,…` CLI), (ii) the SLURM wrapper.

### Task 8: Failing test for the QoE SLURM script

**Files:**
- Create: `tests/test_mahti_qoe_sbatch.py`

- [ ] **Step 18: Write failing test**

```python
"""Static checks on scripts/slurm/mahti_qoe_openweight_d1.sh."""
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "slurm" / "mahti_qoe_openweight_d1.sh"

def test_exists_and_bash_valid():
    assert SCRIPT.exists()
    proc = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

def test_three_qwen_backends_specified():
    text = SCRIPT.read_text()
    for m in ["qwen2.5:1.5b", "qwen2.5:7b", "qwen2.5:32b"]:
        assert m in text, f"missing backend: {m}"

def test_long_walltime_for_full_qoe():
    text = SCRIPT.read_text()
    assert "--time=" in text and "--partition=gpusmall" in text
```

- [ ] **Step 19: Run test to verify RED**

Run: `cd Experiments/neural-router && .venv/bin/python -m pytest tests/test_mahti_qoe_sbatch.py -x`
Expected: failure (script does not exist).

### Task 9: QoE SLURM wrapper

**Files:**
- Create: `scripts/slurm/mahti_qoe_openweight_d1.sh`

- [ ] **Step 20: Write script**

```bash
#!/bin/bash
#SBATCH --job-name=nrouter-qoe
#SBATCH --account=project_2018951
#SBATCH --partition=gpusmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1,nvme:100
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/project_2018951/neural-router/logs/qoe-%j.out
#SBATCH --error=/scratch/project_2018951/neural-router/logs/qoe-%j.err
# =============================================================================
# QoE heterogeneous backend assignment on open-weight Qwen 2.5 tiers
# (manuscript §5.8 / Experiment 3 of the 2026-05-04 validation plan).
#
# Three backends share one Ollama instance via hot-swap on the A100:
#   1.5B (cheap-fast), 7B (mid), 32B (expensive-accurate).
# Strategies: homogeneous, round-robin, qoe-optimised.
# Weight presets: accuracy-first / balanced / cost-first.
# Dataset D1; 5 seeds.
#
# run_qoe.py iterates the full grid internally in one process. A100 40GB
# can hold 7B + 1.5B simultaneously and swaps 32B in/out as needed.
#
# Estimated cost (16 BU/GPU-h × ~6 GPU-h × 1.5 safety) ≈ 145 BU.
# =============================================================================

set -eo pipefail
NR_ROOT=/scratch/project_2018951/neural-router
OLLAMA=$NR_ROOT/bin/ollama-install/bin/ollama

source /etc/profile.d/lmod.sh 2>/dev/null || source /usr/share/lmod/lmod/init/bash
source /appl/profile/zz-csc-env.sh
module load pytorch/2.9
source $NR_ROOT/venv/py312-neural-router/bin/activate

export OLLAMA_MODELS=$NR_ROOT/weights/ollama
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_API_BASE=http://127.0.0.1:11434
export HF_HOME=$NR_ROOT/weights/hf
export TOKENIZERS_PARALLELISM=false

echo "=== qoe open-weight $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi -L
date

$OLLAMA serve > $NR_ROOT/logs/ollama-qoe-$SLURM_JOB_ID.log 2>&1 &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT
for i in $(seq 1 30); do
    curl -sS --max-time 2 http://$OLLAMA_HOST/api/tags >/dev/null 2>&1 && break
    sleep 1
done

# Defensive pulls
for m in "qwen2.5:1.5b" "qwen2.5:7b" "qwen2.5:32b"; do
    if ! $OLLAMA list | grep -q "^${m}"; then
        echo "Pulling ${m} ..."
        $OLLAMA pull "${m}"
    fi
done

cd $NR_ROOT/code
TASK_OUT=$NR_ROOT/code/results/full/qoe/by_task/qwen-tiers_D1
mkdir -p $TASK_OUT

python scripts/run_qoe.py \
    --dataset D1 \
    --strategies homogeneous round_robin qoe_optimised \
    --weight-presets accuracy_first balanced cost_first \
    --seeds 42 123 456 789 1024 \
    --backends "tier_small:ollama/qwen2.5:1.5b,tier_mid:ollama/qwen2.5:7b,tier_large:ollama/qwen2.5:32b" \
    --output-dir $TASK_OUT

echo "=== DONE qoe ==="
date
```

- [ ] **Step 21: Verify static tests GREEN**

Run: `cd Experiments/neural-router && .venv/bin/python -m pytest tests/test_mahti_qoe_sbatch.py -x`
Expected: `3 passed`.

### Task 10: Verify run_qoe.py CLI accepts the flags this script uses

The existing `scripts/run_qoe.py` parses `--strategies`, `--weight-presets`, `--seeds`, `--backends`, `--output-dir`. **However** it currently parses these as comma-separated. The bash script above uses space-separated lists. Need to verify behaviour.

- [ ] **Step 22: Read scripts/run_qoe.py argparse and confirm format**

Run: `grep -E 'add_argument.*(strategies|weight-presets|seeds|backends|output-dir)' Experiments/neural-router/scripts/run_qoe.py`
Expected: each argument's nargs/type configuration. If they're string CSV (`type=str`), update the SLURM wrapper to use comma-separated; if `nargs='+'`, leave space-separated.

- [ ] **Step 23: If mismatch, fix the SLURM wrapper to match the actual CLI (do NOT change `run_qoe.py` — it's already in production use elsewhere).**

### Task 11: QoE smoke

**Files:**
- Create: `scripts/slurm/mahti_qoe_smoke.sh`

- [ ] **Step 24: Write smoke**

```bash
#!/bin/bash
#SBATCH --job-name=qoe-smoke
#SBATCH --account=project_2018951
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1,nvme:50
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project_2018951/neural-router/logs/qoe-smoke-%j.out
#SBATCH --error=/scratch/project_2018951/neural-router/logs/qoe-smoke-%j.err
set -eo pipefail
NR_ROOT=/scratch/project_2018951/neural-router
OLLAMA=$NR_ROOT/bin/ollama-install/bin/ollama
source /etc/profile.d/lmod.sh 2>/dev/null || source /usr/share/lmod/lmod/init/bash
source /appl/profile/zz-csc-env.sh
module load pytorch/2.9
source $NR_ROOT/venv/py312-neural-router/bin/activate
export OLLAMA_MODELS=$NR_ROOT/weights/ollama
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_API_BASE=http://127.0.0.1:11434
export HF_HOME=$NR_ROOT/weights/hf
export TOKENIZERS_PARALLELISM=false

echo "=== qoe smoke $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi -L
date
$OLLAMA serve > $NR_ROOT/logs/ollama-qoe-smoke-$SLURM_JOB_ID.log 2>&1 &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT
for i in $(seq 1 30); do
    curl -sS --max-time 2 http://$OLLAMA_HOST/api/tags >/dev/null 2>&1 && break
    sleep 1
done

cd $NR_ROOT/code
SMOKE_OUT=$NR_ROOT/code/results/full/qoe/by_task/smoke
mkdir -p $SMOKE_OUT

# Minimal: 1 strategy (homogeneous) + 1 backend tier, 1 seed
python scripts/run_qoe.py \
    --dataset D1 \
    --strategies homogeneous \
    --weight-presets balanced \
    --seeds 42 \
    --backends "tier_mid:ollama/qwen2.5:7b" \
    --output-dir $SMOKE_OUT

# L30: validate result CSV
SMOKE_CSV=$SMOKE_OUT/qoe_D1.csv
if [ -s "$SMOKE_CSV" ] && [ "$(wc -l < "$SMOKE_CSV")" -gt 1 ]; then
    echo "PASS smoke"
    head -3 "$SMOKE_CSV"
else
    echo "FAIL smoke: empty/missing $SMOKE_CSV"; exit 1
fi
echo "=== DONE smoke ==="
date
```

- [ ] **Step 25: Bash syntax check**

Run: `bash -n scripts/slurm/mahti_qoe_smoke.sh && echo OK`
Expected: `OK`.

---

## Submission and verification

### Task 12: Sync code to Mahti scratch

- [ ] **Step 26: rsync src + scripts to Mahti**

Run:
```bash
cd Experiments/neural-router && \
  rsync -az src/ scripts/ tests/ \
    mahti:/scratch/project_2018951/neural-router/code/{src,scripts,tests}/ --rsync-path='mkdir -p /scratch/project_2018951/neural-router/code/{src,scripts,tests} && rsync'
```
(Or — simpler — three separate rsyncs.)

Expected: rsync prints "sent X bytes received Y bytes" with no errors.

### Task 13: Run BART-MNLI baseline locally (Experiment 1)

- [ ] **Step 27: Execute steps 5-7 above, write the resulting CSVs into `results/full/ablation/`.**

- [ ] **Step 28: VERIFY (per verification-before-completion skill):**

Run:
```bash
cd Experiments/neural-router && for d in D1 D2 D3; do
  echo "$d:"; head -2 "results/full/ablation/${d}_baseline_zero_shot_results.csv" 2>/dev/null
done
```
Expected: each CSV exists with header + one data row, F1 in [0, 1].

### Task 14: Submit crossover smoke + dependent array

- [ ] **Step 29: ssh mahti and submit smoke + array (gated)**

Run:
```bash
ssh mahti '
cd /scratch/project_2018951/neural-router/code
SBATCH_OUT=$(sbatch scripts/slurm/mahti_crossover_smoke.sh)
echo "$SBATCH_OUT"
SMOKE=$(echo "$SBATCH_OUT" | awk "{print \$NF}")
sbatch --dependency=afterok:$SMOKE scripts/slurm/mahti_crossover_d1.sh
squeue -u $USER -o "%.10i %.20j %.10P %.8T %.10M %.20R"
'
```
Expected: 2 jobs queued (smoke + dependent array).

- [ ] **Step 30: Monitor smoke completion (Monitor tool, until-loop polling sacct State)**

When smoke COMPLETED: `sacct -j <smoke_jid> -X -P --format=State` returns COMPLETED.
Then validate: pull `results/full/crossover/by_task/smoke/crossover_D1.csv`, confirm has data rows.

### Task 15: Submit QoE smoke + dependent run

- [ ] **Step 31: Same pattern: smoke first, dependent full run gated on smoke COMPLETED**

Run:
```bash
ssh mahti '
cd /scratch/project_2018951/neural-router/code
SBATCH_OUT=$(sbatch scripts/slurm/mahti_qoe_smoke.sh)
SMOKE=$(echo "$SBATCH_OUT" | awk "{print \$NF}")
echo "smoke: $SMOKE"
sbatch --dependency=afterok:$SMOKE scripts/slurm/mahti_qoe_openweight_d1.sh
squeue -u $USER
'
```

### Task 16: After all jobs done — pull results, generate figures, populate manuscript

- [ ] **Step 32: rsync results back to laptop**

```bash
cd Experiments/neural-router && \
  rsync -az --include='*/' --include='*.csv' --exclude='*' \
    mahti:/scratch/project_2018951/neural-router/code/results/full/crossover/ \
    results/mahti_mirror/crossover/ && \
  rsync -az --include='*/' --include='*.csv' --exclude='*' \
    mahti:/scratch/project_2018951/neural-router/code/results/full/qoe/ \
    results/mahti_mirror/qoe/
```

- [ ] **Step 33: Aggregate crossover results, write `analysis/make_figures.py::fig_crossover()`**

Crossover figure: F1 (solid) and per-event token cost (dashed) for A0 (blue) and A4 (orange) as |S| grows under W=4K. Two y-axes; vertical dashed line at the analytical |S|·t_s = W threshold (≈ 51 subscriptions for t_s=80, t_inst=200, t_resp=500).

- [ ] **Step 34: Aggregate QoE results, write `analysis/make_figures.py::fig_qoe()` (Pareto F1 vs cost across strategies)**

QoE figure: scatter F1 vs cost-per-1k-events; colour by strategy, shape by weight preset; mark each homogeneous-tier as a coloured anchor. Pareto frontier connects best non-dominated points.

- [ ] **Step 35: Generate the figure PDFs**

Run: `cd Experiments/neural-router && .venv/bin/python analysis/make_figures.py`
Expected: prints `saved crossover.pdf`, `saved qoe_pareto.pdf` along with the existing 8.

- [ ] **Step 36: Replace the §5.7 "deferred to future work" note with `\includegraphics{figs/crossover.pdf}`. Populate `tab:qoe` mixed-strategy rows with real F1 / $/1k / latency.**

- [ ] **Step 37: Final compile check**

Run:
```bash
cd "Manuscripts/Neural Router (Elsevier FGCS)" && rm -f main.aux main.log main.bbl && \
  pdflatex -interaction=nonstopmode main.tex >/dev/null && \
  bibtex main >/dev/null && \
  pdflatex -interaction=nonstopmode main.tex >/dev/null && \
  pdflatex -interaction=nonstopmode main.tex >/dev/null && \
  echo "fatal: $(grep -cE '^! ' main.log) undef: $(grep -cE 'Reference.*undefined' main.log)"
```
Expected: `fatal: 0 undef: 0`.

### Task 17: Logging (per L55 / feedback_remote_logging.md)

- [ ] **Step 38: Append a 2026-05-04 entry to local OPERATIONS_LOG.md**: every action, reason, outcome, BU cost.

- [ ] **Step 39: Mirror that entry to Puhti and Mahti scratch logs**

Run: standard scp + ssh-cat-append pattern used in prior log appends.

---

## Verification gate (verification-before-completion)

Before claiming the campaign closed, run:

```bash
cd Experiments/neural-router && \
ls results/full/ablation/D{1,2,3}_baseline_zero_shot_results.csv && \
ls results/mahti_mirror/crossover/by_task/qwen7b_cross_S*_A{0,4}_s*/crossover_D1.csv | wc -l && \
ls results/mahti_mirror/qoe/by_task/qwen-tiers_D1/qoe_D1.csv && \
cd "../../Manuscripts/Neural Router (Elsevier FGCS)" && \
grep -c "Zero-shot" txt/Results.tex
```

Expected:
- 3 BART-MNLI CSVs.
- 36 crossover task CSVs (one per array task).
- 1 QoE CSV with all (strategy, preset, seed) rows.
- ≥2 mentions of "Zero-shot" in Results.tex.

If any check fails: surface the failure with concrete missing items and continue debugging using systematic-debugging skill.

---

## Out of scope

- Cloud-API QoE (Anthropic credits unconfirmed; user said don't overspend).
- Sonnet D3 ablation (acknowledged as a `--` cell with caption note, manuscript currently consistent).
- D1 5000-event scaling point (D3 scaling is complete, D1 partial; soft gap, deferred).
- Any change to `src/router.py`, `src/baselines.py`, `src/qoe.py`, `scripts/run_crossover.py`, `scripts/run_qoe.py` — all already implemented and used in production. Modifying them risks breaking existing reproducibility.

---

## Estimated total cost

| Item | GPU-h | Mahti BU | Anthropic | Notes |
|---|---|---|---|---|
| BART-MNLI D1+D2+D3 | 0 (laptop CPU) | 0 | 0 | Free |
| Crossover smoke (gputest) | ~0.2 | ~3 | 0 | gputest = 8 BU/h not 16 |
| Crossover full array (36 tasks × ~30 min × 16 BU/h × 1.5 safety) | ~18 | ~432 | 0 | nominal upper bound |
| QoE smoke (gputest) | ~0.2 | ~3 | 0 |  |
| QoE full (~6 GPU-h × 16 × 1.5) | ~6 | ~145 | 0 |  |
| **Total** | **~24 GPU-h** | **~580 BU** | **0** | **0.24 % of envelope** |

p95 worst-case (all walltimes hit ceiling): ~1700 BU still <0.71 %.

---

## Out-of-scope items (not part of this plan)

- Re-running parts of Tier 0 already complete.
- Manuscript narrative changes beyond table-row inserts and figure-block replacement.
- Reconciling the local manifest (the validation experiments aren't tracked in the local manifest; their results live in `results/full/{crossover,qoe}/by_task/` and analysis pulls them directly).
