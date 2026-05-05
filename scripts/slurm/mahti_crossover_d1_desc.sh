#!/bin/bash
#SBATCH --job-name=nrouter-cross-desc
#SBATCH --account=project_2018951
#SBATCH --partition=gpusmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1,nvme:100
#SBATCH --time=02:00:00
#SBATCH --array=0-5%2
#SBATCH --output=/scratch/project_2018951/neural-router/logs/cross-desc-%A_%a.out
#SBATCH --error=/scratch/project_2018951/neural-router/logs/cross-desc-%A_%a.err
# =============================================================================
# D1 crossover, description-aware F1 — replaces the L61-flagged D1 sweep
# whose ID-based F1 collapse on duplication was a metric artifact.
#
# D1 native |S|=19; sweep targets |S| ∈ {50, 200, 2000} require duplication-
# with-rename (multiple IDs sharing a description). Under ID-based F1,
# CoverAndMerge's merging is mistaken for missed matches; under description-
# aware F1, the metric collapses both predictions and ground-truth to
# description sets, so merging duplicates does not artifactually drop F1.
# Description-aware F1 is the correct metric for any sub set with semantic-
# duplicate IDs (Section 5.7, Lessons L61).
#
# Array layout (6 tasks):
#   idx = sub_idx * 2 + cfg_idx
#   sub_volumes ∈ {50, 200, 2000} (3 values)
#   configs ∈ {A0, A4} (2 values)
#   seed = 42 (single seed; confirmation sweep)
#
# Estimated cost (gpusmall = 16 BU/GPU-h × ~25 min/cell × 1.5 safety):
#   6 × (25/60) × 16 × 1.5 ≈ 60 BU.
# Throttle %2 to leave headroom for parallel work.
# =============================================================================

set -eo pipefail

SUB_VOLUMES=(50 200 2000)
CONFIGS=(A0 A4)
SEED=42
W=4096

idx=$SLURM_ARRAY_TASK_ID
sub_idx=$((idx / 2))
cfg_idx=$((idx % 2))
SUB=${SUB_VOLUMES[$sub_idx]}
CONFIG=${CONFIGS[$cfg_idx]}
TAG="qwen7b_cross_desc_S${SUB}_${CONFIG}_s${SEED}"

NR_ROOT=/scratch/project_2018951/neural-router
OLLAMA=$NR_ROOT/bin/ollama-install/bin/ollama

source /etc/profile.d/lmod.sh 2>/dev/null || source /usr/share/lmod/lmod/init/bash
source /appl/profile/zz-csc-env.sh
module load pytorch/2.9
source $NR_ROOT/venv/py312-neural-router/bin/activate

PORT=$((26000 + SLURM_ARRAY_TASK_ID + 1))
export OLLAMA_MODELS=$NR_ROOT/weights/ollama
export OLLAMA_HOST=127.0.0.1:$PORT
export OLLAMA_API_BASE=http://127.0.0.1:$PORT
export HF_HOME=$NR_ROOT/weights/hf
export TOKENIZERS_PARALLELISM=false

echo "=== ${TAG} (idx $idx) on $(hostname) ==="
echo "DATASET=D1 SUB=$SUB CONFIG=$CONFIG SEED=$SEED W=$W metric=description"
nvidia-smi -L
date

$OLLAMA serve > $NR_ROOT/logs/ollama-cross-desc-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log 2>&1 &
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
TASK_OUT=$NR_ROOT/code/results/full/crossover_desc/by_task/$TAG
mkdir -p $TASK_OUT

python scripts/run_crossover.py \
    --dataset D1 \
    --configs $CONFIG \
    --sub-volumes $SUB \
    --seeds $SEED \
    --max-context-tokens $W \
    --metric description \
    --save-per-event \
    --llm-model ollama/qwen2.5:7b \
    --output-dir $TASK_OUT \
    --batch-size 50 \
    --llm-timeout 600

echo "=== DONE $TAG ==="
date
