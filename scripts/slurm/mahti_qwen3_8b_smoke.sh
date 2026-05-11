#!/bin/bash
#SBATCH --job-name=nrouter-q3-smoke
#SBATCH --account=project_2018951
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1,nvme:50
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project_2018951/neural-router/logs/q3-smoke-%j.out
#SBATCH --error=/scratch/project_2018951/neural-router/logs/q3-smoke-%j.err
# =============================================================================
# Smoke test for the Qwen-3-8B / Mahti A100 path.
#
# Run BEFORE submitting mahti_qwen3_8b_ablation.sh so we catch:
#   * whether `ollama pull qwen3:8b` actually retrieved the model on Mahti
#   * whether the model loads on a single A100 40GB
#   * whether NeuralRouter end-to-end runs without crashing on the new prompt
#     formatting / instruction-following Qwen-3 produces.
#
# Multi-level smoke testing: smoke first, full second.
#
# Usage:
#   sbatch scripts/slurm/mahti_qwen3_8b_smoke.sh
# =============================================================================

NR_ROOT=/scratch/project_2018951/neural-router
OLLAMA=$NR_ROOT/bin/ollama-install/bin/ollama

source /etc/profile.d/lmod.sh 2>/dev/null || source /usr/share/lmod/lmod/init/bash
source /appl/profile/zz-csc-env.sh
module load pytorch/2.9
source $NR_ROOT/venv/py312-neural-router/bin/activate

set -eo pipefail

export OLLAMA_MODELS=$NR_ROOT/weights/ollama
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_API_BASE=http://127.0.0.1:11434
export HF_HOME=$NR_ROOT/weights/hf
export TOKENIZERS_PARALLELISM=false

echo "=== Mahti Qwen-3-8B smoke $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi -L
date

$OLLAMA serve > $NR_ROOT/logs/ollama-$SLURM_JOB_ID.log 2>&1 &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT
for i in $(seq 1 30); do
    curl -sS --max-time 2 http://$OLLAMA_HOST/api/tags >/dev/null 2>&1 && { echo "ollama ready after ${i}s"; break; }
    sleep 1
done

# Pull qwen3:8b if not already cached. Idempotent.
if ! $OLLAMA list | grep -q "^qwen3:8b"; then
    echo "Pulling qwen3:8b ..."
    $OLLAMA pull qwen3:8b
fi

echo
echo "=== ollama list ==="
$OLLAMA list
echo
echo "=== Qwen-3-8B inference sanity ==="
$OLLAMA run qwen3:8b "Respond with exactly one word: hello." 2>&1 | tail -3
echo

# End-to-end NeuralRouter smoke: 50 events on D1 with A3.
cd $NR_ROOT/code
echo "=== Neural Router smoke: D1 A3 seed42, 50 events, Qwen-3-8B ==="
python scripts/run_experiment.py \
    --dataset D1 \
    --configs A3 \
    --seeds 42 \
    --max-events 50 \
    --llm-model ollama/qwen3:8b \
    --output-tag mahti_smoke_qwen3-8b \
    --output-dir $NR_ROOT/code/results/full/ablation/by_task/mahti_smoke_qwen3-8b

# Validate the smoke result before declaring success.
echo
echo "=== CSV content check ==="
SMOKE_CSV=$NR_ROOT/code/results/full/ablation/by_task/mahti_smoke_qwen3-8b/mahti_smoke_qwen3-8b_results.csv
if [ -s "$SMOKE_CSV" ] && [ "$(wc -l < "$SMOKE_CSV")" -gt 1 ]; then
    echo "  CSV has data rows:"
    head -2 "$SMOKE_CSV"
else
    echo "  ERROR: smoke CSV is missing or empty (header-only)."
    exit 1
fi

echo
echo "=== DONE ==="
date
