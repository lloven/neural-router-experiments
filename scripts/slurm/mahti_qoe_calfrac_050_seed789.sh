#!/bin/bash
#SBATCH --job-name=qoe-c50-789
#SBATCH --account=project_2018951
#SBATCH --partition=gpusmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1,nvme:50
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/project_2018951/neural-router/logs/qoe-calfrac-050-789-%j.out
#SBATCH --error=/scratch/project_2018951/neural-router/logs/qoe-calfrac-050-789-%j.err
# =============================================================================
# Final 3 missing cells in frac_0.50: seed=789 only
# (homogeneous tier_large, qoe_optimised(balanced), round_robin).
# Uses --llm-cache so it can warm-load from the matched-pair cache if useful.
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

echo "=== qoe-c50-789 $SLURM_JOB_ID on $(hostname), port $PORT ==="
nvidia-smi -L
date
$OLLAMA serve > $NR_ROOT/logs/ollama-qoe-c50-789-$SLURM_JOB_ID.log 2>&1 &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT
for i in $(seq 1 30); do
    if curl -sS --max-time 2 http://$OLLAMA_HOST/api/tags >/dev/null 2>&1; then
        echo "ollama ready at $OLLAMA_HOST after ${i}s"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "FATAL: ollama failed to bind to $OLLAMA_HOST in 30s" >&2
        exit 1
    fi
    sleep 1
done
if ! $OLLAMA list | grep -q "^qwen2.5:7b"; then $OLLAMA pull qwen2.5:7b; fi
if ! $OLLAMA list | grep -q "^qwen2.5:32b"; then $OLLAMA pull qwen2.5:32b; fi

cd $NR_ROOT/code
FULL_OUT=$NR_ROOT/code/results/full/qoe_calfrac/by_task
rm -rf "$FULL_OUT/frac_0.50_seed789"
mkdir -p "$FULL_OUT/frac_0.50_seed789"

CACHE=$FULL_OUT/llm_cache_calfrac.jsonl

echo "=== frac_0.50 seed=789 ==="
python scripts/run_qoe.py \
    --dataset D1 \
    --strategies homogeneous,round_robin,qoe_optimised \
    --weight-presets balanced \
    --seeds 789 \
    --backends "tier_mid:ollama/qwen2.5:7b,tier_large:ollama/qwen2.5:32b" \
    --max-events 1000 \
    --calibration-fraction 0.50 \
    --llm-cache $CACHE \
    --output-dir "$FULL_OUT/frac_0.50_seed789"

CSV="$FULL_OUT/frac_0.50_seed789/qoe_D1.csv"
if [ -s "$CSV" ] && [ "$(wc -l < "$CSV")" -gt 1 ]; then
    echo "PASS frac_0.50_seed789: $(wc -l < "$CSV") lines"
else
    echo "FAIL frac_0.50_seed789: $CSV empty/missing"
    exit 1
fi
echo "=== DONE qoe-c50-789 ==="
date
