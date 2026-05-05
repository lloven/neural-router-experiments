#!/bin/bash
#SBATCH --job-name=q7b-scale
#SBATCH --account=project_2018951
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1,nvme:100
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/project_2018951/neural-router/logs/q7b-scale-%j.out
#SBATCH --error=/scratch/project_2018951/neural-router/logs/q7b-scale-%j.err
# =============================================================================
# Tier 0 fill-in: Qwen-2.5-7B event-scaling sweep on D3.
#
# run_scaling.py iterates all event counts {50, 100, 200, 500, 1000, 2000,
# 5000} inside a single execution and writes one CSV. Therefore this is a
# *single* sbatch (not an array) with a longer walltime to cover the 5000-
# event tail.
#
# Estimated cost (Puhti gpu V100 = 10 BU/GPU-h, ~3-5 h with 1.5× safety):
# ~50 BU nominal, ~100 BU p95.
#
# Submission:
#   ssh puhti
#   cd /scratch/project_2018951/neural-router/code
#   sbatch scripts/slurm/puhti_qwen7b_scaling.sh
# =============================================================================

set -eo pipefail

DATASET=D3
TAG="qwen7b_scaling_events_${DATASET}"

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

echo "=== ${TAG} on $(hostname) ==="
nvidia-smi -L
date

$OLLAMA serve > $NR_ROOT/logs/ollama-$SLURM_JOB_ID.log 2>&1 &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT
for i in $(seq 1 30); do
    curl -sS --max-time 2 http://$OLLAMA_HOST/api/tags >/dev/null 2>&1 && { echo "ollama ready after ${i}s"; break; }
    sleep 1
done

cd $NR_ROOT/code
TASK_OUT=$NR_ROOT/code/results/full/scaling/by_task/$TAG
mkdir -p $TASK_OUT

python scripts/run_scaling.py \
    --dataset $DATASET \
    --dimension events \
    --llm-model ollama/qwen2.5:7b \
    --output-dir $TASK_OUT \
    --mode full

echo "=== DONE $TAG ==="
date
