#!/bin/bash
#SBATCH --job-name=nrouter-d2
#SBATCH --account=project_2018951
#SBATCH --partition=gpusmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1,nvme:100
#SBATCH --time=04:00:00
#SBATCH --array=0-9%5
#SBATCH --output=/scratch/project_2018951/neural-router/logs/d2-%x-%A_%a.out
#SBATCH --error=/scratch/project_2018951/neural-router/logs/d2-%x-%A_%a.err
# =============================================================================
# Parametric D2 ablation on Mahti A100 — used for Tier 1c hypothesis tests.
#
# REQUIRED env vars (passed via `sbatch --export=ALL,...`):
#   MODEL_ID     -- LiteLLM model id, e.g. "ollama/llama3.1:8b"
#   TAG_PREFIX   -- short result-dir prefix, e.g. "llama-8b" or "qwen2.5-32b"
#   OLLAMA_PULL  -- ollama model name to pull, e.g. "llama3.1:8b"
#
# Array layout (10 tasks): cfg_idx ∈ {0,1} × seed_idx ∈ {0..4}
#   idx = cfg_idx * 5 + seed_idx
#   config ∈ {A0, A1}     seed ∈ {42, 123, 456, 789, 1024}
#
# Comparability with the existing Puhti Qwen-2.5-7B D2 results:
#   * Same dataset (D2 = MultiEURLEX, |S|=201)
#   * Same configs A0, A1 (the strongest two on D2 per Discussion §5.2)
#   * Same MAX_EVENTS=300 (matches puhti_qwen7b_ablation.sh — apples-to-apples
#     for the F1 comparison, deliberate)
#   * Same 5 seeds {42, 123, 456, 789, 1024}
# Only the LLM backend differs. This is the design that lets the per-block
# F1 numbers be plotted directly against the existing Qwen-2.5-7B baseline
# without re-running the anchor.
#
# Hypothesis tests served by this script:
#   - B1 (cross-family generalisation):  MODEL_ID=ollama/llama3.1:8b
#   - B2 (size-up scaling):              MODEL_ID=ollama/qwen2.5:32b
#   - B3 (size-down scaling):            MODEL_ID=ollama/qwen2.5:1.5b
#
# Submission examples (run from /scratch/project_2018951/neural-router/code):
#
#   # B1 — Llama 3.1 8B (cross-family at matched compute):
#   sbatch --job-name=nr-b1-llama \
#       --export=ALL,MODEL_ID=ollama/llama3.1:8b,TAG_PREFIX=llama-8b,OLLAMA_PULL=llama3.1:8b \
#       scripts/slurm/mahti_d2_ablation.sh
#
#   # B2 — Qwen 2.5 32B (size-up):
#   sbatch --job-name=nr-b2-qwen32b \
#       --export=ALL,MODEL_ID=ollama/qwen2.5:32b,TAG_PREFIX=qwen2.5-32b,OLLAMA_PULL=qwen2.5:32b \
#       scripts/slurm/mahti_d2_ablation.sh
#
#   # B3 — Qwen 2.5 1.5B (size-down):
#   sbatch --job-name=nr-b3-qwen1.5b \
#       --export=ALL,MODEL_ID=ollama/qwen2.5:1.5b,TAG_PREFIX=qwen2.5-1.5b,OLLAMA_PULL=qwen2.5:1.5b \
#       scripts/slurm/mahti_d2_ablation.sh
# =============================================================================

set -eo pipefail

# --- Required env vars ---
: "${MODEL_ID:?Set MODEL_ID via sbatch --export, e.g. ollama/llama3.1:8b}"
: "${TAG_PREFIX:?Set TAG_PREFIX via sbatch --export, e.g. llama-8b}"
: "${OLLAMA_PULL:?Set OLLAMA_PULL via sbatch --export, e.g. llama3.1:8b}"

DATASETS=(D2)
CONFIGS=(A0 A1)
SEEDS=(42 123 456 789 1024)

idx=$SLURM_ARRAY_TASK_ID
cfg_idx=$((idx / 5))
seed_idx=$((idx % 5))
DATASET=${DATASETS[0]}
CONFIG=${CONFIGS[$cfg_idx]}
SEED=${SEEDS[$seed_idx]}
TAG="${TAG_PREFIX}_${DATASET}_${CONFIG}_s${SEED}"

NR_ROOT=/scratch/project_2018951/neural-router
OLLAMA=$NR_ROOT/bin/ollama-install/bin/ollama

source /etc/profile.d/lmod.sh 2>/dev/null || source /usr/share/lmod/lmod/init/bash
source /appl/profile/zz-csc-env.sh
module load pytorch/2.9
source $NR_ROOT/venv/py312-neural-router/bin/activate

# Per-task ollama port — avoids cross-contamination if SLURM ever packs.
PORT=$((23000 + SLURM_ARRAY_TASK_ID + 1))
export OLLAMA_MODELS=$NR_ROOT/weights/ollama
export OLLAMA_HOST=127.0.0.1:$PORT
export OLLAMA_API_BASE=http://127.0.0.1:$PORT
export HF_HOME=$NR_ROOT/weights/hf
export TOKENIZERS_PARALLELISM=false

echo "=== ${TAG} (task $SLURM_ARRAY_TASK_ID) on $(hostname) ==="
echo "MODEL_ID=$MODEL_ID  TAG=$TAG"
nvidia-smi -L
date

$OLLAMA serve > $NR_ROOT/logs/ollama-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log 2>&1 &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT
for i in $(seq 1 30); do
    curl -sS --max-time 2 http://$OLLAMA_HOST/api/tags >/dev/null 2>&1 && { echo "ollama ready after ${i}s"; break; }
    sleep 1
done

# Defensive pull — idempotent, no-op if model is already cached.
if ! $OLLAMA list | grep -q "^${OLLAMA_PULL}"; then
    echo "Pulling ${OLLAMA_PULL} ..."
    $OLLAMA pull "${OLLAMA_PULL}"
fi

cd $NR_ROOT/code
TASK_OUT=$NR_ROOT/code/results/full/ablation/by_task/$TAG
mkdir -p $TASK_OUT

# MAX_EVENTS=300 matches the Puhti Qwen-2.5-7B D2 cap so F1 figures are
# directly comparable. This cap is documented in OPERATIONS_LOG (a self-
# imposed walltime budget; lift if the discrimination-capacity sweep
# needs full corpus).
MAX_EVENTS=300

python scripts/run_experiment.py \
    --dataset $DATASET \
    --configs $CONFIG \
    --seeds $SEED \
    --llm-model $MODEL_ID \
    --mode full \
    --baselines none \
    --max-events $MAX_EVENTS \
    --output-tag $TAG \
    --output-dir $TASK_OUT \
    --resume

echo "=== DONE $TAG ==="
date
