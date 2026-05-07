#!/bin/bash
#SBATCH --job-name=qoe-c50-rt
#SBATCH --account=project_2018951
#SBATCH --partition=gpusmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1,nvme:50
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project_2018951/neural-router/logs/qoe-calfrac-050-retail-%j.out
#SBATCH --error=/scratch/project_2018951/neural-router/logs/qoe-calfrac-050-retail-%j.err
# =============================================================================
# Re-tail for frac_0.50: 5 cells still missing after 6631533 TIMEOUT.
# Missing: hom_tier_mid s789, hom_tier_large s789, round_robin s789,
#          qoe_optimised(balanced) s456, qoe_optimised(balanced) s789.
# Run --seeds 456,789 covers all (some redundant runs at seed=456 for hom/rr
# but those are deduped at merge time).
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

echo "=== qoe-calfrac-050-retail $SLURM_JOB_ID on $(hostname), port $PORT ==="
nvidia-smi -L
date
$OLLAMA serve > $NR_ROOT/logs/ollama-qoe-calfrac-050-retail-$SLURM_JOB_ID.log 2>&1 &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT
for i in $(seq 1 30); do
    if curl -sS --max-time 2 http://$OLLAMA_HOST/api/tags >/dev/null 2>&1; then
        echo "ollama ready at $OLLAMA_HOST after ${i}s"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "FATAL: ollama failed to bind to $OLLAMA_HOST in 30s" >&2
        tail -20 $NR_ROOT/logs/ollama-qoe-calfrac-050-retail-$SLURM_JOB_ID.log >&2
        exit 1
    fi
    sleep 1
done
if ! $OLLAMA list | grep -q "^qwen2.5:7b"; then $OLLAMA pull qwen2.5:7b; fi
if ! $OLLAMA list | grep -q "^qwen2.5:32b"; then $OLLAMA pull qwen2.5:32b; fi

cd $NR_ROOT/code
FULL_OUT=$NR_ROOT/code/results/full/qoe_calfrac/by_task
rm -rf "$FULL_OUT/frac_0.50_retail"
mkdir -p "$FULL_OUT/frac_0.50_retail"

echo "=== frac_0.50 re-tail (seeds 456,789) ==="
python scripts/run_qoe.py \
    --dataset D1 \
    --strategies homogeneous,round_robin,qoe_optimised \
    --weight-presets balanced \
    --seeds 456,789 \
    --backends "tier_mid:ollama/qwen2.5:7b,tier_large:ollama/qwen2.5:32b" \
    --max-events 1000 \
    --calibration-fraction 0.50 \
    --output-dir "$FULL_OUT/frac_0.50_retail"

CSV="$FULL_OUT/frac_0.50_retail/qoe_D1.csv"
if [ -s "$CSV" ] && [ "$(wc -l < "$CSV")" -gt 1 ]; then
    echo "PASS frac_0.50_retail: $(wc -l < "$CSV") lines"
else
    echo "FAIL frac_0.50_retail: $CSV empty/missing"
    exit 1
fi

echo "=== DONE qoe-calfrac-050-retail ==="
date
