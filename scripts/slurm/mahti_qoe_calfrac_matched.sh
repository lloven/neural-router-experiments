#!/bin/bash
#SBATCH --job-name=qoe-calf-mp
#SBATCH --account=project_2018951
#SBATCH --partition=gpusmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1,nvme:50
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/project_2018951/neural-router/logs/qoe-calfrac-matched-%j.out
#SBATCH --error=/scratch/project_2018951/neural-router/logs/qoe-calfrac-matched-%j.err
# =============================================================================
# H4 calibration-fraction sweep matched-pair re-run (post-L65 fix).
# Runs all 4 fractions {0.05, 0.10, 0.20, 0.50} sequentially in one job,
# sharing a single LLM-call cache. With L65 seed-threading fix, the 5 seeds
# now produce 5 different calibration samples per fraction; with the cache,
# eval-side LLM nondeterminism is neutralised across fractions and across
# strategies (since hom/rr eval doesn't depend on frac).
#
# Pre-fix calfrac data (6628944-6628947, 6629336-6629337) is corrupted by
# the L65 seed bug and discarded. Cleanly fill 6632888 (frac=0.50/seed=789,
# 3 cells from the fix-time fill) merges in afterward via dedup.
#
# Output: results/full/qoe_calfrac_matched/by_task/frac_{0.05,0.10,0.20,0.50}/
# Cache : results/full/qoe_calfrac_matched/llm_cache.jsonl
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

echo "=== qoe-calfrac-matched $SLURM_JOB_ID on $(hostname), port $PORT ==="
nvidia-smi -L
date
$OLLAMA serve > $NR_ROOT/logs/ollama-qoe-calfrac-matched-$SLURM_JOB_ID.log 2>&1 &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT
for i in $(seq 1 30); do
    if curl -sS --max-time 2 http://$OLLAMA_HOST/api/tags >/dev/null 2>&1; then
        echo "ollama ready at $OLLAMA_HOST after ${i}s"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "FATAL: ollama failed to bind in 30s" >&2
        exit 1
    fi
    sleep 1
done
if ! $OLLAMA list | grep -q "^qwen2.5:7b"; then $OLLAMA pull qwen2.5:7b; fi
if ! $OLLAMA list | grep -q "^qwen2.5:32b"; then $OLLAMA pull qwen2.5:32b; fi

cd $NR_ROOT/code
OUT_BASE=$NR_ROOT/code/results/full/qoe_calfrac_matched
LLM_CACHE=$OUT_BASE/llm_cache.jsonl
rm -rf $OUT_BASE
mkdir -p $OUT_BASE/by_task

SHARED_FLAGS=(--dataset D1 \
    --strategies homogeneous,round_robin,qoe_optimised \
    --weight-presets balanced \
    --seeds 42,123,456,789,0 \
    --backends "tier_mid:ollama/qwen2.5:7b,tier_large:ollama/qwen2.5:32b" \
    --max-events 1000 \
    --llm-cache $LLM_CACHE)

for FRAC in 0.05 0.10 0.20 0.50; do
    echo "=== calibration_fraction=$FRAC ==="
    date
    python scripts/run_qoe.py "${SHARED_FLAGS[@]}" \
        --calibration-fraction $FRAC \
        --output-dir "$OUT_BASE/by_task/frac_${FRAC}"
    echo "  cache size after frac=$FRAC: $(wc -l < $LLM_CACHE 2>/dev/null || echo 0) entries"
done

# L30 + L53 validate.
for FRAC in 0.05 0.10 0.20 0.50; do
    CSV="$OUT_BASE/by_task/frac_${FRAC}/qoe_D1.csv"
    if [ -s "$CSV" ] && [ "$(wc -l < "$CSV")" -gt 1 ]; then
        echo "PASS frac=$FRAC: $(wc -l < "$CSV") lines"
    else
        echo "FAIL frac=$FRAC: $CSV empty/missing"
        exit 1
    fi
done

echo "=== DONE qoe-calfrac-matched ==="
date
