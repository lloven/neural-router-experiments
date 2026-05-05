#!/bin/bash
#SBATCH --job-name=q7b-seq
#SBATCH --account=project_2018951
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1,nvme:100
#SBATCH --time=1-12:00:00
#SBATCH --output=/scratch/project_2018951/neural-router/logs/q7b-seq-%j.out
#SBATCH --error=/scratch/project_2018951/neural-router/logs/q7b-seq-%j.err

# Sequential runner for all 44 remaining Qwen-7B ablation runs.
# Invoked as one SLURM job to stay under the daily submit-job quota.
# Tasks to run: 4 D1 A3 seeds + 35 D2 (all configs) + 5 D3 A6 seeds.

NR_ROOT=/scratch/project_2018951/neural-router
OLLAMA=$NR_ROOT/bin/ollama-install/bin/ollama

source /etc/profile.d/lmod.sh 2>/dev/null || source /usr/share/lmod/lmod/init/bash
source /appl/profile/zz-csc-env.sh
module load pytorch/2.9
source $NR_ROOT/venv/py312-neural-router/bin/activate

set -eo pipefail

export OLLAMA_MODELS=$NR_ROOT/weights/ollama
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_API_BASE=http://127.0.0.1:11434
export HF_HOME=$NR_ROOT/weights/hf
export TOKENIZERS_PARALLELISM=false

echo "=== q7b sequential job $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi -L
date

$OLLAMA serve > $NR_ROOT/logs/ollama-seq-$SLURM_JOB_ID.log 2>&1 &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT
for i in $(seq 1 30); do
    curl -sS --max-time 2 http://$OLLAMA_HOST/api/tags >/dev/null 2>&1 && { echo "ollama ready after ${i}s"; break; }
    sleep 1
done

cd $NR_ROOT/code

# Task list: (dataset, config, seed). Dataset-aware max_events: D1/D3=1000, D2=300.
declare -a TASKS=(
    "D1 A3 42"     "D1 A3 123"    "D1 A3 456"    "D1 A3 1024"
    "D2 A0 42"     "D2 A0 123"    "D2 A0 456"    "D2 A0 789"    "D2 A0 1024"
    "D2 A1 42"     "D2 A1 123"    "D2 A1 456"    "D2 A1 789"    "D2 A1 1024"
    "D2 A2 42"     "D2 A2 123"    "D2 A2 456"    "D2 A2 789"    "D2 A2 1024"
    "D2 A3 42"     "D2 A3 123"    "D2 A3 456"    "D2 A3 789"    "D2 A3 1024"
    "D2 A4 42"     "D2 A4 123"    "D2 A4 456"    "D2 A4 789"    "D2 A4 1024"
    "D2 A5 42"     "D2 A5 123"    "D2 A5 456"    "D2 A5 789"    "D2 A5 1024"
    "D2 A6 42"     "D2 A6 123"    "D2 A6 456"    "D2 A6 789"    "D2 A6 1024"
    "D3 A6 42"     "D3 A6 123"    "D3 A6 456"    "D3 A6 789"    "D3 A6 1024"
)

for TASK in "${TASKS[@]}"; do
    read -r DATASET CONFIG SEED <<< "$TASK"
    TAG="qwen7b_${DATASET}_${CONFIG}_s${SEED}"
    TASK_OUT=$NR_ROOT/code/results/full/ablation/by_task/$TAG
    mkdir -p $TASK_OUT

    MAX_EVENTS=1000
    [ "$DATASET" = "D2" ] && MAX_EVENTS=300

    echo
    echo "=== [$TAG] starting at $(date -Iseconds) ==="
    python scripts/run_experiment.py \
        --dataset $DATASET \
        --configs $CONFIG \
        --seeds $SEED \
        --llm-model ollama/qwen2.5:7b \
        --mode full \
        --baselines none \
        --max-events $MAX_EVENTS \
        --output-tag $TAG \
        --output-dir $TASK_OUT \
        --resume \
        || { echo "TASK $TAG FAILED, continuing"; continue; }
    echo "=== [$TAG] done at $(date -Iseconds) ==="
done

echo
echo "=== ALL TASKS DONE ==="
date
