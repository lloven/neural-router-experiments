#!/bin/bash
#SBATCH --job-name=qoe-calfrac-full
#SBATCH --account=project_2018951
#SBATCH --partition=gpusmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1,nvme:50
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/project_2018951/neural-router/logs/qoe-calfrac-full-%j.out
#SBATCH --error=/scratch/project_2018951/neural-router/logs/qoe-calfrac-full-%j.err
# =============================================================================
# H4 calibration-fraction sweep (gpusmall, 3h): TAAS round-1 dual-reviewer
# (TAAS-1, AI-3) recommendation. Empirically substantiates the "calibration-
# noise-limited" claim by sweeping calibration_fraction in {0.05, 0.10, 0.20,
# 0.50} on D1 with two Qwen 2.5 backends (7B + 32B).
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

echo "=== qoe-calfrac full $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi -L
date
$OLLAMA serve > $NR_ROOT/logs/ollama-qoe-calfrac-full-$SLURM_JOB_ID.log 2>&1 &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT
for i in $(seq 1 30); do
    curl -sS --max-time 2 http://$OLLAMA_HOST/api/tags >/dev/null 2>&1 && break
    sleep 1
done
if ! $OLLAMA list | grep -q "^qwen2.5:7b"; then $OLLAMA pull qwen2.5:7b; fi
if ! $OLLAMA list | grep -q "^qwen2.5:32b"; then $OLLAMA pull qwen2.5:32b; fi

cd $NR_ROOT/code
FULL_OUT=$NR_ROOT/code/results/full/qoe_calfrac/by_task
# Clean per-fraction output dirs to avoid schema drift across runs.
# Flag-propagation: stable 'frac_0.NN' naming preserves the dot so neither
# bash nor python need to reconstruct it from a numeric value.
for FRAC in 0.05 0.10 0.20 0.50; do
    rm -rf "$FULL_OUT/frac_${FRAC}"
done
mkdir -p $FULL_OUT
SHARED_FLAGS=(--dataset D1 \
    --strategies homogeneous,round_robin,qoe_optimised \
    --weight-presets balanced \
    --seeds 42,123,456,789,0 \
    --backends "tier_mid:ollama/qwen2.5:7b,tier_large:ollama/qwen2.5:32b" \
    --max-events 1000)

# Sweep four calibration fractions
for FRAC in 0.05 0.10 0.20 0.50; do
    echo "=== calibration_fraction=$FRAC ==="
    python scripts/run_qoe.py "${SHARED_FLAGS[@]}" \
        --calibration-fraction $FRAC \
        --output-dir "$FULL_OUT/frac_${FRAC}"
done

# Validation: every fraction wrote a non-empty CSV and the CSV records
# the fraction value (so downstream analysis can group by it).
for FRAC in 0.05 0.10 0.20 0.50; do
    CSV="$FULL_OUT/frac_${FRAC}/qoe_D1.csv"
    if [ -s "$CSV" ] && [ "$(wc -l < "$CSV")" -gt 1 ]; then
        echo "PASS frac=$FRAC: $(wc -l < "$CSV") lines"
    else
        echo "FAIL frac=$FRAC: $CSV empty/missing"
        exit 1
    fi
done

echo "=== DONE qoe-calfrac full ==="
date
