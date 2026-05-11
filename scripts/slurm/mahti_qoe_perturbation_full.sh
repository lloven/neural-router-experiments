#!/bin/bash
#SBATCH --job-name=qoe-pert-full
#SBATCH --account=project_2018951
#SBATCH --partition=gpusmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1,nvme:50
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/project_2018951/neural-router/logs/qoe-pert-full-%j.out
#SBATCH --error=/scratch/project_2018951/neural-router/logs/qoe-pert-full-%j.err
# =============================================================================
# C9 perturbation full run (gpusmall, 4h): TAAS round-1 reviewer mandate.
#
# Grid: 3 perturbation cells x 3 strategies x 5 seeds x 2 backends.
#   Perturbations: baseline, topic_restricted_cal, latency_injection.
#   Strategies:    homogeneous-7B, homogeneous-32B, round_robin, qoe_optimised.
#   Seeds:         42, 123, 456, 789, 0.
#   Backends:      Qwen 2.5 7B, Qwen 2.5 32B.
#
# Output: results/full/qoe_perturbation/by_task/{cell}/qoe_D1.csv
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

echo "=== qoe-perturbation full $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi -L
date
$OLLAMA serve > $NR_ROOT/logs/ollama-qoe-pert-full-$SLURM_JOB_ID.log 2>&1 &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT
for i in $(seq 1 30); do
    curl -sS --max-time 2 http://$OLLAMA_HOST/api/tags >/dev/null 2>&1 && break
    sleep 1
done

if ! $OLLAMA list | grep -q "^qwen2.5:7b"; then
    $OLLAMA pull qwen2.5:7b
fi
if ! $OLLAMA list | grep -q "^qwen2.5:32b"; then
    $OLLAMA pull qwen2.5:32b
fi

cd $NR_ROOT/code
FULL_OUT=$NR_ROOT/code/results/full/qoe_perturbation/by_task
# Clean state per run for the canonical (non-smoke) cells we're about
# to write — avoid schema drift across runs. Smoke output (subdir 'smoke')
# is preserved for reference.
for sub in baseline topic_restricted latency_injection; do
    rm -rf "$FULL_OUT/$sub"
done
mkdir -p $FULL_OUT
SHARED_FLAGS=(--dataset D1 \
    --strategies homogeneous,round_robin,qoe_optimised \
    --weight-presets accuracy_first,balanced,cost_first \
    --seeds 42,123,456,789,0 \
    --backends "tier_mid:ollama/qwen2.5:7b,tier_large:ollama/qwen2.5:32b" \
    --max-events 1000 --calibration-fraction 0.10)

# Cell 1: baseline (no perturbation)
echo "=== Cell 1/3: baseline ==="
python scripts/run_qoe.py "${SHARED_FLAGS[@]}" --output-dir "$FULL_OUT/baseline"

# Cell 2: topic_restricted_cal (calibration sample restricted)
echo "=== Cell 2/3: topic_restricted_cal ==="
python scripts/run_qoe.py "${SHARED_FLAGS[@]}" \
    --perturbation topic_restricted_cal \
    --topic-mask "sports,business_&_entrepreneurs,arts_&_culture" \
    --output-dir "$FULL_OUT/topic_restricted"

# Cell 3: latency_injection
echo "=== Cell 3/3: latency_injection ==="
python scripts/run_qoe.py "${SHARED_FLAGS[@]}" \
    --perturbation latency_injection \
    --injection-event-index 500 \
    --injected-latency-s 0.05 \
    --output-dir "$FULL_OUT/latency_injection"

# Validate every cell wrote a non-empty CSV
for sub in baseline topic_restricted latency_injection; do
    CSV="$FULL_OUT/$sub/qoe_D1.csv"
    if [ -s "$CSV" ] && [ "$(wc -l < "$CSV")" -gt 1 ]; then
        echo "PASS $sub: $(wc -l < "$CSV") lines"
    else
        echo "FAIL $sub: $CSV empty/missing"
        exit 1
    fi
done

echo "=== DONE qoe-perturbation full ==="
date
