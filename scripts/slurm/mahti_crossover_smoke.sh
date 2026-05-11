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
# =============================================================================
# Crossover smoke (gputest, 15 min): single (|S|=200, A0+A4, seed 42) point
# under the W=4096 budget. Validates the parametric pattern + ollama port +
# output CSV shape before launching the 36-task array.
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

echo "=== crossover smoke $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi -L
date
$OLLAMA serve > $NR_ROOT/logs/ollama-cross-smoke-$SLURM_JOB_ID.log 2>&1 &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT
for i in $(seq 1 30); do
    curl -sS --max-time 2 http://$OLLAMA_HOST/api/tags >/dev/null 2>&1 && { echo "ollama ready after ${i}s"; break; }
    sleep 1
done

# Defensive pull (even when weights cache is on scratch, the local server
# needs to register the model on first start on this compute node).
echo "=== ollama list before pull ==="
$OLLAMA list
if ! $OLLAMA list | grep -q "^qwen2.5:7b"; then
    echo "Pulling qwen2.5:7b ..."
    $OLLAMA pull qwen2.5:7b
fi

# Multi-level smoke: integration (50 events, ~1 min) → full (1000 events, ~10 min).
# Default integration so accidental re-submission is cheap; full requires
# explicit SMOKE_LEVEL=full export.
SMOKE_LEVEL=${SMOKE_LEVEL:-integration}
case "$SMOKE_LEVEL" in
    integration) MAX_E=50;   N_SUBS=50;  CFGS=A0    ;;  # ~1 min, validates schema
    full)        MAX_E=1000; N_SUBS=200; CFGS=A0,A4 ;;  # ~10 min, proxies prod task
    *) echo "unknown SMOKE_LEVEL=$SMOKE_LEVEL (use integration|full)" >&2; exit 2 ;;
esac
echo "=== SMOKE_LEVEL=$SMOKE_LEVEL  configs=$CFGS  n_subs=$N_SUBS  max_events=$MAX_E ==="

cd $NR_ROOT/code
SMOKE_OUT=$NR_ROOT/code/results/full/crossover/by_task/smoke_$SMOKE_LEVEL
mkdir -p $SMOKE_OUT

python scripts/run_crossover.py \
    --dataset D1 \
    --configs "$CFGS" \
    --sub-volumes "$N_SUBS" \
    --seeds 42 \
    --max-context-tokens 4096 \
    --llm-model ollama/qwen2.5:7b \
    --output-dir $SMOKE_OUT \
    --batch-size 50 \
    --llm-timeout 600 \
    --max-events $MAX_E

# CSV content check (data row present, not just file existence).
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
