#!/bin/bash
#SBATCH --job-name=qoe-pt-tail
#SBATCH --account=project_2018951
#SBATCH --partition=gpusmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1,nvme:50
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project_2018951/neural-router/logs/qoe-pert-topic-tail-%j.out
#SBATCH --error=/scratch/project_2018951/neural-router/logs/qoe-pert-topic-tail-%j.err
# =============================================================================
# Tail-job for topic_restricted_cal: 9 missing cells from 6628942 walltime kill.
# Missing seeds {0, 789} for the qoe_optimised strategy and seed=0 for hom/rr.
# Runs --seeds 0,789 (some hom/rr runs at seed=789 will be redundant
# duplicates of existing rows but cheap; post-job merge dedups by tuple).
# Output to a SEPARATE subdir (topic_restricted_tail) so the main 21-row
# CSV is preserved and the merge happens explicitly post-job.
# =============================================================================

set -eo pipefail
NR_ROOT=/scratch/project_2018951/neural-router
OLLAMA=$NR_ROOT/bin/ollama-install/bin/ollama
source /etc/profile.d/lmod.sh 2>/dev/null || source /usr/share/lmod/lmod/init/bash
source /appl/profile/zz-csc-env.sh
module load pytorch/2.9
source $NR_ROOT/venv/py312-neural-router/bin/activate
export OLLAMA_MODELS=$NR_ROOT/weights/ollama
PORT=$((20000 + SLURM_JOB_ID % 10000))
export OLLAMA_HOST=127.0.0.1:$PORT
export OLLAMA_API_BASE=http://127.0.0.1:$PORT
export HF_HOME=$NR_ROOT/weights/hf
export TOKENIZERS_PARALLELISM=false

echo "=== qoe-pert-topic-tail $SLURM_JOB_ID on $(hostname), port $PORT ==="
nvidia-smi -L
date
$OLLAMA serve > $NR_ROOT/logs/ollama-qoe-pert-topic-tail-$SLURM_JOB_ID.log 2>&1 &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT
for i in $(seq 1 30); do
    if curl -sS --max-time 2 http://$OLLAMA_HOST/api/tags >/dev/null 2>&1; then
        echo "ollama ready at $OLLAMA_HOST after ${i}s"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "FATAL: ollama failed to bind to $OLLAMA_HOST in 30s" >&2
        tail -20 $NR_ROOT/logs/ollama-qoe-pert-topic-tail-$SLURM_JOB_ID.log >&2
        exit 1
    fi
    sleep 1
done

if ! $OLLAMA list | grep -q "^qwen2.5:7b"; then $OLLAMA pull qwen2.5:7b; fi
if ! $OLLAMA list | grep -q "^qwen2.5:32b"; then $OLLAMA pull qwen2.5:32b; fi

cd $NR_ROOT/code
FULL_OUT=$NR_ROOT/code/results/full/qoe_perturbation/by_task
rm -rf "$FULL_OUT/topic_restricted_tail"
mkdir -p "$FULL_OUT/topic_restricted_tail"

echo "=== topic_restricted_cal tail (seeds 0,789) ==="
python scripts/run_qoe.py \
    --dataset D1 \
    --strategies homogeneous,round_robin,qoe_optimised \
    --weight-presets accuracy_first,balanced,cost_first \
    --seeds 0,789 \
    --backends "tier_mid:ollama/qwen2.5:7b,tier_large:ollama/qwen2.5:32b" \
    --max-events 1000 --calibration-fraction 0.10 \
    --perturbation topic_restricted_cal \
    --topic-mask "sports,business_&_entrepreneurs,arts_&_culture" \
    --output-dir "$FULL_OUT/topic_restricted_tail"

CSV="$FULL_OUT/topic_restricted_tail/qoe_D1.csv"
if [ -s "$CSV" ] && [ "$(wc -l < "$CSV")" -gt 1 ]; then
    echo "PASS topic_restricted_tail: $(wc -l < "$CSV") lines"
else
    echo "FAIL topic_restricted_tail: $CSV empty/missing"
    exit 1
fi

echo "=== DONE qoe-pert-topic-tail ==="
date
