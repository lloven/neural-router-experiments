#!/bin/bash
#SBATCH --job-name=q3-8b-abl
#SBATCH --account=project_2018951
#SBATCH --partition=gpusmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1,nvme:100
#SBATCH --time=04:00:00
# NB: 4 h is generous, not a CSC limit — Mahti gpusmall caps at 1-12:00:00
# (1.5 days). Lift if the discrimination-capacity sweep needs more events.
#SBATCH --array=0-9%5
#SBATCH --output=/scratch/project_2018951/neural-router/logs/q3-8b-%A_%a.out
#SBATCH --error=/scratch/project_2018951/neural-router/logs/q3-8b-%A_%a.err
# =============================================================================
# Tier 1c: Qwen-3-8B replication of the Qwen-2.5-7B discrimination-capacity
# experiment. Newer-generation open-weight model at *matched compute* (8B vs
# 7B) — tests whether the |S|=201 collapse observed on D2 is a generic
# property of LLM-as-matching-engine or a Qwen-2.5-era artefact.
#
# Scope: D2 only (the dataset where the collapse happens), A0 + A1 (the two
# strongest configs for D2 in the existing data per Discussion.tex), 5 seeds.
#   array layout: idx = config_idx * 5 + seed_idx
#                 config ∈ {A0, A1}    seed ∈ {42, 123, 456, 789, 1024}
# Total: 10 array tasks. Walltime budgeted at 4 h/task on A100 — generous
# given Qwen-3-8B is faster than 2.5-7B and D2 cap is 300 events.
# Throttled to 5 concurrent (%5) to avoid pulling more than one ollama image
# concurrently and to leave A100 headroom for shared tenants.
#
# Comparability with existing Qwen-2.5-7B Puhti results:
#   * Same ablation configs (A0, A1)
#   * Same dataset (D2 = MultiEURLEX, |S|=201)
#   * Same MAX_EVENTS=300 (D2 cap, matches puhti_qwen7b_ablation.sh)
#   * Same seeds (42, 123, 456, 789, 1024)
#   * Same prompt templates, same evaluation harness
# Differences are confined to: (1) backend = ollama/qwen3:8b instead of
# ollama/qwen2.5:7b, (2) GPU = A100 instead of V100. (2) is irrelevant for
# F1 (it only affects wall-clock latency).
#
# Cost estimate: ~30-60 GPU-h on A100, well under 1% of the 250k BU envelope.
#
# Usage:
#   # 1. Run the smoke first (REQUIRED, per L32):
#   sbatch scripts/slurm/mahti_qwen3_8b_smoke.sh
#   # 2. Verify the smoke output before proceeding.
#   # 3. Submit the array:
#   sbatch scripts/slurm/mahti_qwen3_8b_ablation.sh
#   # 4. Watch:  squeue -u $USER -j <jobid>
#   # 5. Reconcile when done:
#   #    From laptop: python scripts/reconcile_puhti.py \
#   #        --ssh-host mahti --pull
#   #    (then --apply)
# =============================================================================

DATASETS=(D2)
CONFIGS=(A0 A1)
SEEDS=(42 123 456 789 1024)

idx=$SLURM_ARRAY_TASK_ID
ds_idx=0
cfg_idx=$((idx / 5))
seed_idx=$((idx % 5))
DATASET=${DATASETS[$ds_idx]}
CONFIG=${CONFIGS[$cfg_idx]}
SEED=${SEEDS[$seed_idx]}
TAG="qwen3-8b_${DATASET}_${CONFIG}_s${SEED}"

NR_ROOT=/scratch/project_2018951/neural-router
OLLAMA=$NR_ROOT/bin/ollama-install/bin/ollama

source /etc/profile.d/lmod.sh 2>/dev/null || source /usr/share/lmod/lmod/init/bash
source /appl/profile/zz-csc-env.sh
module load pytorch/2.9
source $NR_ROOT/venv/py312-neural-router/bin/activate

set -eo pipefail

# Per-task ollama port avoids cross-contamination if SLURM ever packs.
PORT=$((22000 + SLURM_ARRAY_TASK_ID + 1))
export OLLAMA_MODELS=$NR_ROOT/weights/ollama
export OLLAMA_HOST=127.0.0.1:$PORT
export OLLAMA_API_BASE=http://127.0.0.1:$PORT
export HF_HOME=$NR_ROOT/weights/hf
export TOKENIZERS_PARALLELISM=false

echo "=== q3-8b ablation task $SLURM_ARRAY_TASK_ID: $DATASET / $CONFIG / seed $SEED on $(hostname) ==="
nvidia-smi -L
date

$OLLAMA serve > $NR_ROOT/logs/ollama-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log 2>&1 &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT
for i in $(seq 1 30); do
    curl -sS --max-time 2 http://$OLLAMA_HOST/api/tags >/dev/null 2>&1 && { echo "ollama ready after ${i}s"; break; }
    sleep 1
done

# Defensive pull. Idempotent — no-op if already cached on $OLLAMA_MODELS.
if ! $OLLAMA list | grep -q "^qwen3:8b"; then
    echo "Pulling qwen3:8b ..."
    $OLLAMA pull qwen3:8b
fi

cd $NR_ROOT/code
TASK_OUT=$NR_ROOT/code/results/full/ablation/by_task/$TAG
mkdir -p $TASK_OUT

# Match the Puhti Qwen-2.5-7B cap so the F1 comparison is apples-to-apples.
MAX_EVENTS=300

python scripts/run_experiment.py \
    --dataset $DATASET \
    --configs $CONFIG \
    --seeds $SEED \
    --llm-model ollama/qwen3:8b \
    --mode full \
    --baselines none \
    --max-events $MAX_EVENTS \
    --output-tag $TAG \
    --output-dir $TASK_OUT \
    --resume

echo "=== DONE $TAG ==="
date
