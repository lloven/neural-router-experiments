#!/bin/bash
#SBATCH --job-name=qoe-smoke
#SBATCH --account=project_2018951
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1,nvme:50
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project_2018951/neural-router/logs/qoe-smoke-%j.out
#SBATCH --error=/scratch/project_2018951/neural-router/logs/qoe-smoke-%j.err
# =============================================================================
# QoE smoke (gputest, 15 min): minimal QoE run with one backend tier and one
# strategy + weight preset. Validates Ollama hot-swap, run_qoe.py CLI shape,
# and CSV output before launching the full 8-hour job.
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

echo "=== qoe smoke $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi -L
date
$OLLAMA serve > $NR_ROOT/logs/ollama-qoe-smoke-$SLURM_JOB_ID.log 2>&1 &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT
for i in $(seq 1 30); do
    curl -sS --max-time 2 http://$OLLAMA_HOST/api/tags >/dev/null 2>&1 && break
    sleep 1
done

# Defensive pull. Even if the model was cached on scratch from a prior
# campaign, the local Ollama server needs to register it on first start
# on this compute node. Idempotent: no-op if already known.
echo "=== ollama list before pull ==="
$OLLAMA list
if ! $OLLAMA list | grep -q "^qwen2.5:7b"; then
    echo "Pulling qwen2.5:7b ..."
    $OLLAMA pull qwen2.5:7b
fi

# L23 multi-level smoke: integration (50 events, ~1 min) → full (full corpus, ~8 min).
# Default integration so accidental re-submission is cheap.
SMOKE_LEVEL=${SMOKE_LEVEL:-integration}
case "$SMOKE_LEVEL" in
    integration) MAX_E=50;    STRATS=homogeneous                   ;;  # ~1 min, schema validation
    full)        MAX_E=200;   STRATS=homogeneous,qoe_optimised     ;;  # ~5 min, exercises calibration + qoe_optimised path
    *) echo "unknown SMOKE_LEVEL=$SMOKE_LEVEL (use integration|full)" >&2; exit 2 ;;
esac
echo "=== SMOKE_LEVEL=$SMOKE_LEVEL  strategies=$STRATS  max_events=$MAX_E ==="

cd $NR_ROOT/code
SMOKE_OUT=$NR_ROOT/code/results/full/qoe/by_task/smoke_$SMOKE_LEVEL
mkdir -p $SMOKE_OUT

python scripts/run_qoe.py \
    --dataset D1 \
    --strategies "$STRATS" \
    --weight-presets balanced \
    --seeds 42 \
    --backends "tier_mid:ollama/qwen2.5:7b" \
    --max-events $MAX_E \
    --output-dir $SMOKE_OUT

# L30: validate result CSV (data row present).
SMOKE_CSV=$SMOKE_OUT/qoe_D1.csv
if [ -s "$SMOKE_CSV" ] && [ "$(wc -l < "$SMOKE_CSV")" -gt 1 ]; then
    echo "PASS smoke: CSV has data"
    head -3 "$SMOKE_CSV"
else
    echo "FAIL smoke: empty/missing CSV at $SMOKE_CSV"
    exit 1
fi
echo "=== DONE smoke ==="
date
