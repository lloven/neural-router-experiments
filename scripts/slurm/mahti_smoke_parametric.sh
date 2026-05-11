#!/bin/bash
#SBATCH --job-name=nrouter-smoke
#SBATCH --account=project_2018951
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1,nvme:50
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project_2018951/neural-router/logs/smoke-%x-%j.out
#SBATCH --error=/scratch/project_2018951/neural-router/logs/smoke-%x-%j.err
# =============================================================================
# Parametric smoke test for any new Mahti model. Use BEFORE submitting the
# corresponding Tier 1c array (mahti_d2_ablation.sh) to validate that the
# model loads, Ollama serves it, and a 50-event run completes end-to-end.
#
# REQUIRED env vars:
#   MODEL_ID     -- e.g. "ollama/llama3.1:8b"
#   TAG_PREFIX   -- e.g. "llama-8b"
#   OLLAMA_PULL  -- e.g. "llama3.1:8b"
#
# Submission:
#   sbatch --job-name=smoke-llama \
#       --export=ALL,MODEL_ID=ollama/llama3.1:8b,TAG_PREFIX=llama-8b,OLLAMA_PULL=llama3.1:8b \
#       scripts/slurm/mahti_smoke_parametric.sh
# =============================================================================

set -eo pipefail

: "${MODEL_ID:?Set MODEL_ID via sbatch --export}"
: "${TAG_PREFIX:?Set TAG_PREFIX via sbatch --export}"
: "${OLLAMA_PULL:?Set OLLAMA_PULL via sbatch --export}"

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

echo "=== smoke[$TAG_PREFIX] $SLURM_JOB_ID on $(hostname) ==="
echo "MODEL_ID=$MODEL_ID  OLLAMA_PULL=$OLLAMA_PULL"
nvidia-smi -L
date

$OLLAMA serve > $NR_ROOT/logs/ollama-smoke-$SLURM_JOB_ID.log 2>&1 &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT
for i in $(seq 1 30); do
    curl -sS --max-time 2 http://$OLLAMA_HOST/api/tags >/dev/null 2>&1 && { echo "ollama ready after ${i}s"; break; }
    sleep 1
done

# Pull the model. Idempotent.
if ! $OLLAMA list | grep -q "^${OLLAMA_PULL}"; then
    echo "Pulling ${OLLAMA_PULL} (this is the first smoke for this model) ..."
    time $OLLAMA pull "${OLLAMA_PULL}"
fi

echo
echo "=== ollama list ==="
$OLLAMA list
echo
echo "=== ${OLLAMA_PULL} sanity ==="
$OLLAMA run "${OLLAMA_PULL}" "Respond with exactly one word: hello." 2>&1 | tail -3
echo

# End-to-end smoke: 50 events on D1 with A3 (clustering+C&M)
cd $NR_ROOT/code
SMOKE_TAG="smoke_${TAG_PREFIX}"
SMOKE_OUT=$NR_ROOT/code/results/full/ablation/by_task/$SMOKE_TAG
mkdir -p $SMOKE_OUT

echo "=== Neural Router smoke: D1 A3 seed42, 50 events, ${MODEL_ID} ==="
python scripts/run_experiment.py \
    --dataset D1 \
    --configs A3 \
    --seeds 42 \
    --max-events 50 \
    --llm-model "$MODEL_ID" \
    --output-tag "$SMOKE_TAG" \
    --output-dir "$SMOKE_OUT"

# CSV content check.
echo
echo "=== CSV content check ==="
SMOKE_CSV=$SMOKE_OUT/${SMOKE_TAG}_results.csv
if [ -s "$SMOKE_CSV" ] && [ "$(wc -l < "$SMOKE_CSV")" -gt 1 ]; then
    echo "  CSV has data rows:"
    head -2 "$SMOKE_CSV"
    echo
    echo "PASS: ${TAG_PREFIX} smoke produced a valid CSV row."
else
    echo "FAIL: ${TAG_PREFIX} smoke CSV is missing or empty (header-only)."
    exit 1
fi

# Realised billing rate dump for calibration (per the 2026-04-29 OPS_LOG entry).
echo
echo "=== sacct billing for this smoke task ==="
sacct -j $SLURM_JOB_ID -X -P --format=JobID,Partition,Elapsed,ElapsedRaw,AllocTRES,State 2>&1 | tail -3

echo
echo "=== DONE smoke[$TAG_PREFIX] ==="
date
