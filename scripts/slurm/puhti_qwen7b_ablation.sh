#!/bin/bash
#SBATCH --job-name=q7b-abl
#SBATCH --account=project_2018951
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1,nvme:100
#SBATCH --time=03:00:00
#SBATCH --array=0-104%40
#SBATCH --output=/scratch/project_2018951/neural-router/logs/q7b-%A_%a.out
#SBATCH --error=/scratch/project_2018951/neural-router/logs/q7b-%A_%a.err

# Array index → (dataset, config, seed). Layout:
#   idx = dataset_idx * 35 + config_idx * 5 + seed_idx
#   dataset ∈ {D1, D2, D3}, config ∈ {A0..A6}, seed ∈ {42, 123, 456, 789, 1024}
DATASETS=(D1 D2 D3)
CONFIGS=(A0 A1 A2 A3 A4 A5 A6)
SEEDS=(42 123 456 789 1024)

idx=$SLURM_ARRAY_TASK_ID
ds_idx=$((idx / 35))
cfg_idx=$(((idx % 35) / 5))
seed_idx=$((idx % 5))
DATASET=${DATASETS[$ds_idx]}
CONFIG=${CONFIGS[$cfg_idx]}
SEED=${SEEDS[$seed_idx]}
TAG="qwen7b_${DATASET}_${CONFIG}_s${SEED}"

NR_ROOT=/scratch/project_2018951/neural-router
OLLAMA=$NR_ROOT/bin/ollama-install/bin/ollama

source /etc/profile.d/lmod.sh 2>/dev/null || source /usr/share/lmod/lmod/init/bash
source /appl/profile/zz-csc-env.sh
module load pytorch/2.9
source $NR_ROOT/venv/py312-neural-router/bin/activate

set -eo pipefail

# Per-task Ollama: bind to a port derived from the array task to avoid cross-contamination
# if SLURM ever packs multiple tasks on one node (shouldn't, but belt+braces).
PORT=$((21000 + SLURM_ARRAY_TASK_ID + 1))
export OLLAMA_MODELS=$NR_ROOT/weights/ollama
export OLLAMA_HOST=127.0.0.1:$PORT
export OLLAMA_API_BASE=http://127.0.0.1:$PORT
export HF_HOME=$NR_ROOT/weights/hf
export TOKENIZERS_PARALLELISM=false

echo "=== q7b ablation task $SLURM_ARRAY_TASK_ID: $DATASET / $CONFIG / seed $SEED on $(hostname) ==="
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
# Per-task isolated output dir: each array task gets its own .checkpoints
# subdirectory so concurrent tasks don't race on Lustre metadata. Results will
# be merged after all tasks complete.
TASK_OUT=$NR_ROOT/code/results/full/ablation/by_task/$TAG
mkdir -p $TASK_OUT

# Per-dataset event caps: D1/D3 at 1000 events, D2 at 300 events.
# D2 (EUR-Lex) has much longer documents and more subscriptions; A6 at 1000
# events doesn't fit in the 3h wall limit. 300 keeps the D2 column uniform
# across all configs and finishes in time.
MAX_EVENTS=1000
[ "$DATASET" = "D2" ] && MAX_EVENTS=300

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
    --resume

echo "=== DONE $TAG ==="
date
