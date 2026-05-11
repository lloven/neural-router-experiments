#!/bin/bash
#SBATCH --job-name=nrouter-cross-d2
#SBATCH --account=project_2018951
#SBATCH --partition=gpusmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1,nvme:50
#SBATCH --time=01:00:00
#SBATCH --array=0-7%2
#SBATCH --output=/scratch/project_2018951/neural-router/logs/cross-d2-%A_%a.out
#SBATCH --error=/scratch/project_2018951/neural-router/logs/cross-d2-%A_%a.err
# =============================================================================
# Crossover empirical sweep on D2 native — re-execution of §5.7 on a
# distribution where every subscription has a UNIQUE description, so
# ID-based F1 is the correct metric (no duplication-with-rename artifact).
# Replaces the D1-with-duplication sweep whose negative result was traced
# to a metric artifact, not the cost model.
#
# D2 has 127 native subscriptions with unique descriptions. With W=4096
# the prompt fits all 127 subs (no truncation, no crossover). To exercise
# the cost-model prediction, we sweep budget W ∈ {1024, 4096} across
# |S| ∈ {32, 64, 96, 127} for A0 vs A4, single seed.
#
# At W=1024: ~36 subs fit → A0 truncates above |S|=36, A4 compresses.
# At W=4096: ~377 subs fit → no truncation, A0 ≈ A4 (sanity check).
#
# Array layout (8 tasks):
#   idx = w_idx * 4 + sub_idx  (CONFIG hardcoded to A0,A4 inside the run)
#   W ∈ {1024, 4096} (2 values)
#   |S| ∈ {32, 64, 96, 127} (4 values)
# Total = 2 × 4 = 8 tasks running BOTH A0 and A4 per task.
# Throttled to %2 concurrent on gpusmall (matches single-A100 capacity).
#
# Estimated cost (gpusmall = 16 BU/GPU-h, ~25 min/task with 1.5× safety):
#   8 × (25/60) × 16 × 1.5 ≈ 80 BU (point estimate).
#   p95: 8 × 1 × 16 × 1.5 ≈ 192 BU upper bound.
# Single seed because this is a confirmation run (not a power study) and
# the original D1 sweep was also single-seed for the salient cells.
# =============================================================================

set -eo pipefail

W_VALUES=(1024 4096)
SUB_VOLUMES=(32 64 96 127)
SEED=42

idx=$SLURM_ARRAY_TASK_ID
w_idx=$((idx / 4))
sub_idx=$((idx % 4))
W=${W_VALUES[$w_idx]}
SUB=${SUB_VOLUMES[$sub_idx]}
TAG="qwen7b_cross_d2_W${W}_S${SUB}_s${SEED}"

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
echo "DATASET=D2 W=$W SUB=$SUB SEED=$SEED CONFIGS=A0,A4"
nvidia-smi -L
date

$OLLAMA serve > $NR_ROOT/logs/ollama-cross-d2-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log 2>&1 &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT
for i in $(seq 1 30); do
    curl -sS --max-time 2 http://$OLLAMA_HOST/api/tags >/dev/null 2>&1 && { echo "ollama ready after ${i}s"; break; }
    sleep 1
done

if ! $OLLAMA list | grep -q "^qwen2.5:7b"; then
    echo "Pulling qwen2.5:7b ..."
    $OLLAMA pull qwen2.5:7b
fi

cd $NR_ROOT/code
TASK_OUT=$NR_ROOT/code/results/full/crossover/by_task/$TAG
mkdir -p $TASK_OUT

# D2 events are long (~1150 tokens median); --batch-size 5 keeps the
# prompt safely under qwen2.5:7b's 32K context cap. At batch=50 the
# prompt overflows and the LLM returns empty matches even when the
# cost-model logic is fine (B58 smoke 6619841 diagnostic).
# --max-events 200 keeps wall-time bounded while giving stable F1.
python scripts/run_crossover.py \
    --dataset D2 \
    --configs A0,A4 \
    --sub-volumes $SUB \
    --seeds $SEED \
    --max-context-tokens $W \
    --llm-model ollama/qwen2.5:7b \
    --output-dir $TASK_OUT \
    --batch-size 5 \
    --llm-timeout 600 \
    --max-events 200

echo "=== DONE $TAG ==="
date
