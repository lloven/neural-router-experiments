#!/bin/bash
#SBATCH --job-name=nrouter-cross-d2-smoke
#SBATCH --account=project_2018951
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1,nvme:50
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project_2018951/neural-router/logs/cross-d2-smoke-%j.out
#SBATCH --error=/scratch/project_2018951/neural-router/logs/cross-d2-smoke-%j.err
# =============================================================================
# D2-native crossover smoke (gputest, 15 min). Validates the parametric
# script flow + crossover signal direction at the most informative cell:
# W=1024, |S|=127 (forces ~70% of native subs to be truncated by A0).
#
# Multi-level smoke testing: integration smoke runs at 50 events for
# ~3 min, full smoke at 100 events for ~8 min. Default integration to
# keep accidental re-submissions cheap.
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

echo "=== d2 crossover smoke $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi -L
date
$OLLAMA serve > $NR_ROOT/logs/ollama-cross-d2-smoke-$SLURM_JOB_ID.log 2>&1 &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT
for i in $(seq 1 30); do
    curl -sS --max-time 2 http://$OLLAMA_HOST/api/tags >/dev/null 2>&1 && { echo "ollama ready after ${i}s"; break; }
    sleep 1
done

echo "=== ollama list ==="
$OLLAMA list
if ! $OLLAMA list | grep -q "^qwen2.5:7b"; then
    echo "Pulling qwen2.5:7b ..."
    $OLLAMA pull qwen2.5:7b
fi

SMOKE_LEVEL=${SMOKE_LEVEL:-integration}
# D2 events are long (~1150 tokens median); batch-size 5 keeps prompt
# safely under qwen2.5:7b's 32K context cap. (At batch=50, prompt
# overflows and the LLM returns garbage / empty matches → F1=0 even when
# the cost-model logic is fine. Diagnosed in B58, 2026-05-05.)
case "$SMOKE_LEVEL" in
    integration) MAX_E=30;  W=1024; N_SUBS=127; CFGS=A0,A4; BATCH=5 ;;  # ~5 min
    full)        MAX_E=100; W=1024; N_SUBS=127; CFGS=A0,A4; BATCH=5 ;;  # ~12 min
    *) echo "unknown SMOKE_LEVEL=$SMOKE_LEVEL (use integration|full)" >&2; exit 2 ;;
esac
echo "=== SMOKE_LEVEL=$SMOKE_LEVEL  W=$W  |S|=$N_SUBS  configs=$CFGS  max_events=$MAX_E batch=$BATCH ==="

cd $NR_ROOT/code
SMOKE_OUT=$NR_ROOT/code/results/full/crossover/by_task/d2_smoke_$SMOKE_LEVEL
mkdir -p $SMOKE_OUT

python scripts/run_crossover.py \
    --dataset D2 \
    --configs "$CFGS" \
    --sub-volumes "$N_SUBS" \
    --seeds 42 \
    --max-context-tokens $W \
    --llm-model ollama/qwen2.5:7b \
    --output-dir $SMOKE_OUT \
    --batch-size $BATCH \
    --llm-timeout 600 \
    --max-events $MAX_E

SMOKE_CSV=$SMOKE_OUT/crossover_D2.csv
if [ -s "$SMOKE_CSV" ] && [ "$(wc -l < "$SMOKE_CSV")" -gt 1 ]; then
    echo "PASS smoke: CSV has data"
    head -3 "$SMOKE_CSV"
    echo "=== A0 vs A4 rows ==="
    awk -F',' 'NR==1 || $1=="A0" || $1=="A4"' "$SMOKE_CSV"
else
    echo "FAIL smoke: empty/missing CSV at $SMOKE_CSV"
    exit 1
fi
echo "=== DONE smoke ==="
date
