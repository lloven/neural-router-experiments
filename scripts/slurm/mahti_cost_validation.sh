#!/bin/bash
#SBATCH --job-name=nrouter-cost-val
#SBATCH --account=project_2018951
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1,nvme:50
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project_2018951/neural-router/logs/cost-val-%j.out
#SBATCH --error=/scratch/project_2018951/neural-router/logs/cost-val-%j.err
# =============================================================================
# Cost-model validation re-run with per-cluster logging (gputest, 15 min).
#
# SMOKE_LEVEL parametric (per L23 multi-level smoke testing):
#   integration: A0 × k=1 × seed=42 × --max-events 50, ~1 min on gputest.
#                Validates Ollama + per-cluster CSV path on actual GPU.
#   full       : A0,A3 × k∈{1,5,19} × seed=42 × --max-events 1000.
#                ~12 min wall-clock × 8 BU/GPU-h ≈ 1.6 BU.
# Default SMOKE_LEVEL=integration so accidental re-submission is cheap.
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

echo "=== cost-validation $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi -L
date
$OLLAMA serve > $NR_ROOT/logs/ollama-cost-val-$SLURM_JOB_ID.log 2>&1 &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT
for i in $(seq 1 30); do
    curl -sS --max-time 2 http://$OLLAMA_HOST/api/tags >/dev/null 2>&1 && { echo "ollama ready after ${i}s"; break; }
    sleep 1
done

echo "=== ollama list before pull ==="
$OLLAMA list
if ! $OLLAMA list | grep -q "^qwen2.5:7b"; then
    echo "Pulling qwen2.5:7b ..."
    $OLLAMA pull qwen2.5:7b
fi

SMOKE_LEVEL=${SMOKE_LEVEL:-integration}
case "$SMOKE_LEVEL" in
    integration) CFGS=A0;       KS=1;        MAX_E=50   ;;
    full)        CFGS=A0,A3;    KS=1,5,19;   MAX_E=1000 ;;
    *) echo "unknown SMOKE_LEVEL=$SMOKE_LEVEL (use integration|full)" >&2; exit 2 ;;
esac
echo "=== SMOKE_LEVEL=$SMOKE_LEVEL  configs=$CFGS  k=$KS  max_events=$MAX_E ==="

cd $NR_ROOT/code
OUT_DIR=$NR_ROOT/code/results/full/cost_validation/$SMOKE_LEVEL
mkdir -p $OUT_DIR

python scripts/run_cost_validation.py \
    --dataset D1 \
    --configs "$CFGS" \
    --k-values "$KS" \
    --seed 42 \
    --max-events $MAX_E \
    --llm-model ollama/qwen2.5:7b \
    --output-dir $OUT_DIR

# L30 content check.
OUT_CSV=$OUT_DIR/cost_validation_D1.csv
if [ -s "$OUT_CSV" ] && [ "$(wc -l < "$OUT_CSV")" -gt 1 ]; then
    echo "PASS: cost-validation CSV has data"
    head -2 "$OUT_CSV"
    echo "Row count: $(($(wc -l < "$OUT_CSV") - 1))"
else
    echo "FAIL: empty/missing CSV at $OUT_CSV"
    exit 1
fi
echo "=== DONE cost-validation ==="
date
