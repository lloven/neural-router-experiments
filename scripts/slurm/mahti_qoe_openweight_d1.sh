#!/bin/bash
#SBATCH --job-name=nrouter-qoe
#SBATCH --account=project_2018951
#SBATCH --partition=gpusmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1,nvme:100
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/project_2018951/neural-router/logs/qoe-%j.out
#SBATCH --error=/scratch/project_2018951/neural-router/logs/qoe-%j.err
# =============================================================================
# QoE heterogeneous backend assignment on open-weight Qwen 2.5 tiers —
# manuscript Contribution #4 / §5.8 / Experiment 3 of the 2026-05-04
# validation plan.
#
# Two backends share one Ollama instance via hot-swap on the A100:
#   tier_mid    = qwen2.5:7b    (mid-cost)
#   tier_large  = qwen2.5:32b   (expensive-accurate)
#
# tier_small (qwen2.5:1.5b) DROPPED: 6620213 (cancelled) showed 38+ JSON
# parse failures in 17 min — the 1.5B model is too small to reliably emit
# the structured output our matcher requires (consistent with the open-
# weight LLM quirks pattern: small models don't follow JSON instructions).
# Including it in the QoE sweep only burns compute on F1=0 noise.
#
# Strategies: homogeneous, round_robin, qoe_optimised
# Weight presets: accuracy_first, balanced, cost_first
# Dataset: D1, --max-events 1000 (matches the existing D1 ablation cap;
#          full corpus would push wall-time past 6h).
# Seeds: 42, 123, 456 (3 seeds, down from 5; sufficient for CI; deepens
#        wall-time margin).
#
# run_qoe.py iterates the full grid internally in one process. A100 40GB
# can hold 7B + 32B with hot-swap.
#
# Estimated cell wall-times (post-cancellation evidence + 1000-event cap):
#   homogeneous(7B):    3 seeds × ~10 min  = 30 min
#   homogeneous(32B):   3 seeds × ~25 min  = 75 min
#   round_robin:        3 seeds × ~18 min  = 54 min
#   qoe_optimised:      3 seeds × 3 presets × ~18 min = 162 min
#   calibration phase:  ~15 min total
# Total: ~336 min ≈ 5.6h (fits the 6h cap with headroom).
# BU estimate: 5.6h × 16 BU/h × 1.5 safety ≈ 135 BU.
# =============================================================================

set -eo pipefail
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

echo "=== qoe open-weight $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi -L
date

$OLLAMA serve > $NR_ROOT/logs/ollama-qoe-$SLURM_JOB_ID.log 2>&1 &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT
for i in $(seq 1 30); do
    curl -sS --max-time 2 http://$OLLAMA_HOST/api/tags >/dev/null 2>&1 && { echo "ollama ready after ${i}s"; break; }
    sleep 1
done

# Defensive pulls (both should be cached from prior Tier 1c campaign).
for m in "qwen2.5:7b" "qwen2.5:32b"; do
    if ! $OLLAMA list | grep -q "^${m}"; then
        echo "Pulling ${m} ..."
        $OLLAMA pull "${m}"
    fi
done

cd $NR_ROOT/code
TASK_OUT=$NR_ROOT/code/results/full/qoe/by_task/qwen-tiers_D1
mkdir -p $TASK_OUT

# All CLI args are comma-separated strings (verified against scripts/run_qoe.py).
python scripts/run_qoe.py \
    --dataset D1 \
    --strategies homogeneous,round_robin,qoe_optimised \
    --weight-presets accuracy_first,balanced,cost_first \
    --seeds 42,123,456 \
    --backends "tier_mid:ollama/qwen2.5:7b,tier_large:ollama/qwen2.5:32b" \
    --max-events 1000 \
    --output-dir $TASK_OUT

echo "=== DONE qoe ==="
date
