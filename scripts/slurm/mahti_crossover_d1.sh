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
# Crossover empirical sweep — manuscript §5.7 / Experiment 2 of the
# 2026-05-04 validation plan.
#
# Validates the cost-model prediction that CoverAndMerge (A4) preserves
# accuracy where A0 (raw LLM) must drop subscriptions to fit the prompt
# budget. Uses Qwen-2.5-7B with a hard W=4096-token budget enforced via
# RouterConfig.max_context_tokens (already in router.py).
#
# Array layout (36 tasks):
#   idx = sub_idx * 6 + cfg_idx * 3 + seed_idx
#   sub_volumes ∈ {50, 100, 200, 500, 1000, 2000} (6 values)
#   configs ∈ {A0, A4} (2 values)
#   seeds ∈ {42, 123, 456} (3 values)
# Total = 6 × 2 × 3 = 36, throttled to %4 concurrent on gpusmall.
#
# Per-task output dir: results/full/crossover/by_task/<TAG>/ — each array
# task writes to its own directory so parallel tasks don't race.
#
# Estimated cost (A100 gpusmall = 16 BU/GPU-h, mean ~30 min/task with 1.5×
# safety): 36 × 0.5 × 16 × 1.5 ≈ 432 BU upper bound. p95 worst-case 36 × 2
# × 16 × 1.5 ≈ 1700 BU.
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

# Per-task ollama port avoids cross-contamination if SLURM ever packs.
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

# Defensive pull (qwen2.5:7b should already be cached from the Tier 0 campaign).
if ! $OLLAMA list | grep -q "^qwen2.5:7b"; then
    echo "Pulling qwen2.5:7b ..."
    $OLLAMA pull qwen2.5:7b
fi

cd $NR_ROOT/code
TASK_OUT=$NR_ROOT/code/results/full/crossover/by_task/$TAG
mkdir -p $TASK_OUT

# run_crossover.py iterates SUB_VOLUMES × CONFIGS × SEEDS by default; we
# constrain to a single point per array task by passing single-element
# CLI lists.
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
