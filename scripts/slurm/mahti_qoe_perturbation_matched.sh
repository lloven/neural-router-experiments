#!/bin/bash
#SBATCH --job-name=qoe-pert-mp
#SBATCH --account=project_2018951
#SBATCH --partition=gpusmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1,nvme:50
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/project_2018951/neural-router/logs/qoe-pert-matched-%j.out
#SBATCH --error=/scratch/project_2018951/neural-router/logs/qoe-pert-matched-%j.err
# =============================================================================
# C9 perturbation matched-pair re-run (post-L65 fix).
# Runs baseline + topic_restricted_cal + latency_injection sequentially in
# ONE SLURM job, sharing a single LLM-call cache. With the L65 seed-threading
# fix in src/qoe.py, the 5 seeds now actually produce 5 different calibration
# samples; the shared cache neutralises LLM nondeterminism between baseline
# and perturbed cells (matched-cell semantics).
#
# Output: results/full/qoe_perturbation_matched/by_task/{baseline,
#         topic_restricted, latency_injection}/qoe_D1.csv
# Cache : results/full/qoe_perturbation_matched/llm_cache.jsonl (~1-2 GB
#         for 90 cells × ~30 LLM calls × ~3 KB/entry).
#
# Estimated runtime: 90 cells × ~5.3 min/cell = ~8h IF every call hits the
# API. With the cache, baseline populates the cache, and the two perturbed
# cells reuse cache for unchanged prompts (only topic_restricted changes
# the calibration step's prompts; latency_injection adds wall-clock sleep
# but reuses LLM responses verbatim). Expected: baseline ~3h, topic ~2.5h,
# latency ~1.5h ≈ 7h total. 6h walltime is tight; if it walltime-kills,
# split per-perturbation in tail-job style.
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

echo "=== qoe-pert-matched $SLURM_JOB_ID on $(hostname), port $PORT ==="
nvidia-smi -L
date
$OLLAMA serve > $NR_ROOT/logs/ollama-qoe-pert-matched-$SLURM_JOB_ID.log 2>&1 &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT
for i in $(seq 1 30); do
    if curl -sS --max-time 2 http://$OLLAMA_HOST/api/tags >/dev/null 2>&1; then
        echo "ollama ready at $OLLAMA_HOST after ${i}s"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "FATAL: ollama failed to bind to $OLLAMA_HOST in 30s" >&2
        tail -20 $NR_ROOT/logs/ollama-qoe-pert-matched-$SLURM_JOB_ID.log >&2
        exit 1
    fi
    sleep 1
done

if ! $OLLAMA list | grep -q "^qwen2.5:7b"; then $OLLAMA pull qwen2.5:7b; fi
if ! $OLLAMA list | grep -q "^qwen2.5:32b"; then $OLLAMA pull qwen2.5:32b; fi

cd $NR_ROOT/code
OUT_BASE=$NR_ROOT/code/results/full/qoe_perturbation_matched
LLM_CACHE=$OUT_BASE/llm_cache.jsonl
# L51: clean state per run.
rm -rf $OUT_BASE
mkdir -p $OUT_BASE/by_task

SHARED_FLAGS=(--dataset D1 \
    --strategies homogeneous,round_robin,qoe_optimised \
    --weight-presets accuracy_first,balanced,cost_first \
    --seeds 42,123,456,789,0 \
    --backends "tier_mid:ollama/qwen2.5:7b,tier_large:ollama/qwen2.5:32b" \
    --max-events 1000 --calibration-fraction 0.10 \
    --llm-cache $LLM_CACHE)

# Cell 1: baseline (no perturbation) — populates the cache
echo "=== Cell 1/3: baseline (populates LLM cache) ==="
date
python scripts/run_qoe.py "${SHARED_FLAGS[@]}" --output-dir "$OUT_BASE/by_task/baseline"
echo "  cache size: $(wc -l < $LLM_CACHE 2>/dev/null || echo 0) entries"

# Cell 2: topic_restricted_cal — the calibration sample changes, so calibration
# prompts will MISS cache; eval prompts (drawn from full event pool) will HIT.
echo "=== Cell 2/3: topic_restricted_cal ==="
date
python scripts/run_qoe.py "${SHARED_FLAGS[@]}" \
    --perturbation topic_restricted_cal \
    --topic-mask "sports,business_&_entrepreneurs,arts_&_culture" \
    --output-dir "$OUT_BASE/by_task/topic_restricted"
echo "  cache size: $(wc -l < $LLM_CACHE 2>/dev/null || echo 0) entries"

# Cell 3: latency_injection — injection only adds wall-clock sleep, prompts
# are unchanged, so this should be cache-hit-dominant.
echo "=== Cell 3/3: latency_injection ==="
date
python scripts/run_qoe.py "${SHARED_FLAGS[@]}" \
    --perturbation latency_injection \
    --injection-event-index 500 \
    --injected-latency-s 0.05 \
    --output-dir "$OUT_BASE/by_task/latency_injection"
echo "  cache size: $(wc -l < $LLM_CACHE 2>/dev/null || echo 0) entries"

# L30 validate.
for sub in baseline topic_restricted latency_injection; do
    CSV="$OUT_BASE/by_task/$sub/qoe_D1.csv"
    if [ -s "$CSV" ] && [ "$(wc -l < "$CSV")" -gt 1 ]; then
        echo "PASS $sub: $(wc -l < "$CSV") lines"
    else
        echo "FAIL $sub: $CSV empty/missing"
        exit 1
    fi
done

echo "=== DONE qoe-pert-matched ==="
date
