#!/bin/bash
#SBATCH --job-name=nr-r5-bf
#SBATCH --account=project_2018951
#SBATCH --partition=gpusmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1,nvme:50
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/project_2018951/neural-router/logs/r5-bf-%A_%a.out
#SBATCH --error=/scratch/project_2018951/neural-router/logs/r5-bf-%A_%a.err
# =============================================================================
# R5 backfill — TAAS round-1 VAL-1 mandate (n=5 across all D1 Qwen-7B
# ablation configs). Investigation found tab:cross-dataset cells already at
# n=5; only the underlying D1_ablation_qwen7b_results.csv has two configs
# below n=5 (A4 missing seed 789; A6 missing seeds 456, 789, 1024).
# This array tops both up to n=5.
#
# Each task runs a single (config, seed) cell on Mahti A100 (canonical
# template: scripts/slurm/puhti_qwen7b_ablation.sh, retargeted to gpusmall).
# Estimated runtime ~30 min/task (matches Puhti V100 baseline within 1.5x);
# 2h walltime per task is comfortable.
#
# Per-task isolated output dir (results/full/ablation/by_task/$TAG);
# does NOT touch existing s42/s123/s456/s1024 cells outside reseed scope.
# Flag-propagation: --seeds value passed as int-string; written verbatim
# to CSV via the canonical _append_row path in run_experiment.py.
# Verified post-run.
# =============================================================================

set -eo pipefail

# Backfill matrix: 4 cells = D1/A4/s789 + D1/A6/{s456,s789,s1024}
CFG_ARR=(A4 A6 A6 A6)
SEED_ARR=(789 456 789 1024)

idx=$SLURM_ARRAY_TASK_ID
CONFIG=${CFG_ARR[$idx]}
SEED=${SEED_ARR[$idx]}
DATASET=D1
TAG="qwen7b_${DATASET}_${CONFIG}_s${SEED}"

NR_ROOT=/scratch/project_2018951/neural-router
OLLAMA=$NR_ROOT/bin/ollama-install/bin/ollama

source /etc/profile.d/lmod.sh 2>/dev/null || source /usr/share/lmod/lmod/init/bash
source /appl/profile/zz-csc-env.sh
module load pytorch/2.9
source $NR_ROOT/venv/py312-neural-router/bin/activate

PORT=$((24000 + SLURM_ARRAY_TASK_ID + 1))
export OLLAMA_MODELS=$NR_ROOT/weights/ollama
export OLLAMA_HOST=127.0.0.1:$PORT
export OLLAMA_API_BASE=http://127.0.0.1:$PORT
export HF_HOME=$NR_ROOT/weights/hf
export TOKENIZERS_PARALLELISM=false

echo "=== R5-BACKFILL idx=$idx: $DATASET / $CONFIG / seed $SEED on $(hostname) ==="
nvidia-smi -L
date

$OLLAMA serve > $NR_ROOT/logs/ollama-r5bf-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log 2>&1 &
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
TASK_OUT=$NR_ROOT/code/results/full/ablation/by_task/$TAG
# Only mkdir; do NOT rm existing cells.
mkdir -p $TASK_OUT

python scripts/run_experiment.py \
    --dataset $DATASET \
    --configs $CONFIG \
    --seeds $SEED \
    --llm-model ollama/qwen2.5:7b \
    --mode full \
    --baselines none \
    --max-events 1000 \
    --output-tag $TAG \
    --output-dir $TASK_OUT \
    --resume

# Verify the new row landed with expected seed value.
RESULTS_CSV=$(ls $TASK_OUT/*.csv 2>/dev/null | head -1)
if [ ! -s "$RESULTS_CSV" ]; then
    echo "FAIL: no CSV output in $TASK_OUT"
    exit 1
fi
if ! awk -F, -v s="$SEED" 'NR>1 && $3==s {found=1} END{exit !found}' "$RESULTS_CSV"; then
    echo "FAIL: seed=$SEED not present in $RESULTS_CSV (flag round-trip violated)"
    exit 1
fi
echo "PASS $TAG: seed=$SEED present"

echo "=== DONE R5-BACKFILL $TAG ==="
date
