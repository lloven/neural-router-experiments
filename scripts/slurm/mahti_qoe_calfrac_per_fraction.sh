#!/bin/bash
#SBATCH --job-name=qoe-calfrac
#SBATCH --account=project_2018951
#SBATCH --partition=gpusmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1,nvme:50
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/project_2018951/neural-router/logs/qoe-calfrac-frac-%j.out
#SBATCH --error=/scratch/project_2018951/neural-router/logs/qoe-calfrac-frac-%j.err
# =============================================================================
# H4 calibration-fraction sweep, per-fraction split (gpusmall, 2.5h).
# Submit with: sbatch --export=ALL,FRAC=0.05 mahti_qoe_calfrac_per_fraction.sh
# Split from mahti_qoe_calfrac_full.sh after the original 3h job ran out of
# walltime. Each invocation runs ONE fraction (~20 cells, ~1:50h).
#
# Flag-propagation: the FRAC value is consumed verbatim — both as the directory
# suffix (frac_${FRAC} keeps the dot) and as the --calibration-fraction value
# passed to run_qoe.py. No bash/python normalization mismatch (str(0.10) → '0.1').
# =============================================================================

set -eo pipefail
if [ -z "${FRAC:-}" ]; then
    echo "FATAL: FRAC env var must be set (e.g. sbatch --export=ALL,FRAC=0.05 ...)" >&2
    exit 1
fi
echo "=== Running fraction FRAC=$FRAC ==="

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

# Per-job unique port to avoid collisions when SLURM packs multiple jobs
# on the same node (root cause of 6628944/6628945 F1 corruption: both bound
# 11434 on g3301; the losing ollama failed silently and the python script
# routed requests to the surviving ollama under unexpected concurrent state).
PORT=$((20000 + SLURM_JOB_ID % 10000))
export OLLAMA_HOST=127.0.0.1:$PORT
export OLLAMA_API_BASE=http://127.0.0.1:$PORT

echo "=== qoe-calfrac-${FRAC} $SLURM_JOB_ID on $(hostname), port $PORT ==="
nvidia-smi -L
date
$OLLAMA serve > $NR_ROOT/logs/ollama-qoe-calfrac-frac-$SLURM_JOB_ID.log 2>&1 &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT
# Verify ollama bound to OUR port within 30s; fail loud if not.
for i in $(seq 1 30); do
    if curl -sS --max-time 2 http://$OLLAMA_HOST/api/tags >/dev/null 2>&1; then
        echo "ollama ready at $OLLAMA_HOST after ${i}s"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "FATAL: ollama failed to bind to $OLLAMA_HOST in 30s" >&2
        tail -20 $NR_ROOT/logs/ollama-qoe-calfrac-frac-$SLURM_JOB_ID.log >&2
        exit 1
    fi
    sleep 1
done
if ! $OLLAMA list | grep -q "^qwen2.5:7b"; then $OLLAMA pull qwen2.5:7b; fi
if ! $OLLAMA list | grep -q "^qwen2.5:32b"; then $OLLAMA pull qwen2.5:32b; fi

cd $NR_ROOT/code
FULL_OUT=$NR_ROOT/code/results/full/qoe_calfrac/by_task
# Clean ONLY this fraction's subdir; preserve other fractions.
rm -rf "$FULL_OUT/frac_${FRAC}"
mkdir -p "$FULL_OUT/frac_${FRAC}"

python scripts/run_qoe.py \
    --dataset D1 \
    --strategies homogeneous,round_robin,qoe_optimised \
    --weight-presets balanced \
    --seeds 42,123,456,789,0 \
    --backends "tier_mid:ollama/qwen2.5:7b,tier_large:ollama/qwen2.5:32b" \
    --max-events 1000 \
    --calibration-fraction $FRAC \
    --output-dir "$FULL_OUT/frac_${FRAC}"

CSV="$FULL_OUT/frac_${FRAC}/qoe_D1.csv"
if [ -s "$CSV" ] && [ "$(wc -l < "$CSV")" -gt 1 ]; then
    echo "PASS frac=$FRAC: $(wc -l < "$CSV") lines"
else
    echo "FAIL frac=$FRAC: $CSV empty/missing"
    exit 1
fi

echo "=== DONE qoe-calfrac-${FRAC} ==="
date
