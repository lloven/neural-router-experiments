#!/bin/bash
#SBATCH --job-name=qoe-pert-lat
#SBATCH --account=project_2018951
#SBATCH --partition=gpusmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1,nvme:50
#SBATCH --time=03:30:00
#SBATCH --output=/scratch/project_2018951/neural-router/logs/qoe-pert-lat-%j.out
#SBATCH --error=/scratch/project_2018951/neural-router/logs/qoe-pert-lat-%j.err
# =============================================================================
# C9 perturbation split — latency_injection cell only (gpusmall, 3.5h).
# Split from mahti_qoe_perturbation_full.sh; baseline preserved.
# Estimated runtime: 30 cells x ~5.3 min/cell = ~2:40h.
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

echo "=== qoe-pert-lat $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi -L
date
$OLLAMA serve > $NR_ROOT/logs/ollama-qoe-pert-lat-$SLURM_JOB_ID.log 2>&1 &
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
# L51: clean ONLY this cell's subdir; preserve baseline (already complete).
rm -rf "$FULL_OUT/latency_injection"
mkdir -p "$FULL_OUT/latency_injection"
SHARED_FLAGS=(--dataset D1 \
    --strategies homogeneous,round_robin,qoe_optimised \
    --weight-presets accuracy_first,balanced,cost_first \
    --seeds 42,123,456,789,0 \
    --backends "tier_mid:ollama/qwen2.5:7b,tier_large:ollama/qwen2.5:32b" \
    --max-events 1000 --calibration-fraction 0.10)

echo "=== latency_injection ==="
python scripts/run_qoe.py "${SHARED_FLAGS[@]}" \
    --perturbation latency_injection \
    --injection-event-index 500 \
    --injected-latency-s 0.05 \
    --output-dir "$FULL_OUT/latency_injection"

CSV="$FULL_OUT/latency_injection/qoe_D1.csv"
if [ -s "$CSV" ] && [ "$(wc -l < "$CSV")" -gt 1 ]; then
    echo "PASS latency_injection: $(wc -l < "$CSV") lines"
else
    echo "FAIL latency_injection: $CSV empty/missing"
    exit 1
fi

echo "=== DONE qoe-pert-lat ==="
date
