#!/bin/bash
#SBATCH --job-name=q7b-sens
#SBATCH --account=project_2018951
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1,nvme:100
#SBATCH --time=03:00:00
#SBATCH --array=0-7%4
#SBATCH --output=/scratch/project_2018951/neural-router/logs/q7b-swp-%A_%a.out
#SBATCH --error=/scratch/project_2018951/neural-router/logs/q7b-swp-%A_%a.err
# =============================================================================
# Tier 0 fill-in: Qwen-2.5-7B sensitivity sweeps NOT covered by
# puhti_qwen7b_ablation.sh.
#
# Array layout (8 tasks): sweep_idx ∈ {0..3} × dataset_idx ∈ {0,1}
#   idx = sweep_idx * 2 + dataset_idx
#   sweeps:   k, tau, kappa, embedding
#   datasets: D1, D3 (D2 sensitivity not in local manifest pendings)
#
# Throttled to %4 concurrent. Each sweep runs all parameter values inside
# one task (run_sensitivity.py iterates internally).
#
# Scaling (7 event-count points on D3) is NOT in this script — run
# `puhti_qwen7b_scaling.sh` separately for that (different walltime).
#
# Estimated cost (Puhti gpu V100 = 10 BU/GPU-h, mean ~1 h/task with 1.5×
# safety): 8 × 1 h × 10 × 1.5 ≈ 120 BU. p95 ~360 BU.
#
# Submission:
#   ssh puhti
#   cd /scratch/project_2018951/neural-router/code
#   sbatch scripts/slurm/puhti_qwen7b_sweeps.sh
# =============================================================================

set -eo pipefail

SWEEPS=(k tau kappa embedding)
DATASETS=(D1 D3)

idx=$SLURM_ARRAY_TASK_ID
sweep_idx=$((idx / 2))
ds_idx=$((idx % 2))
SWEEP=${SWEEPS[$sweep_idx]}
DATASET=${DATASETS[$ds_idx]}
TAG="qwen7b_sens_${SWEEP}_${DATASET}"

NR_ROOT=/scratch/project_2018951/neural-router
OLLAMA=$NR_ROOT/bin/ollama-install/bin/ollama

source /etc/profile.d/lmod.sh 2>/dev/null || source /usr/share/lmod/lmod/init/bash
source /appl/profile/zz-csc-env.sh
module load pytorch/2.9
source $NR_ROOT/venv/py312-neural-router/bin/activate

# Per-task ollama port
PORT=$((24000 + SLURM_ARRAY_TASK_ID + 1))
export OLLAMA_MODELS=$NR_ROOT/weights/ollama
export OLLAMA_HOST=127.0.0.1:$PORT
export OLLAMA_API_BASE=http://127.0.0.1:$PORT
export HF_HOME=$NR_ROOT/weights/hf
export TOKENIZERS_PARALLELISM=false

echo "=== ${TAG} (idx $idx) on $(hostname) ==="
nvidia-smi -L
date

$OLLAMA serve > $NR_ROOT/logs/ollama-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log 2>&1 &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT
for i in $(seq 1 30); do
    curl -sS --max-time 2 http://$OLLAMA_HOST/api/tags >/dev/null 2>&1 && { echo "ollama ready after ${i}s"; break; }
    sleep 1
done

cd $NR_ROOT/code
TASK_OUT=$NR_ROOT/code/results/full/sensitivity/by_task/$TAG
mkdir -p $TASK_OUT

# Cap events to fit walltime (matches ablation convention).
MAX_EVENTS=1000

python scripts/run_sensitivity.py \
    --dataset $DATASET \
    --sweep $SWEEP \
    --llm-model ollama/qwen2.5:7b \
    --max-events $MAX_EVENTS \
    --output-dir $TASK_OUT

echo "=== DONE $TAG ==="
date
