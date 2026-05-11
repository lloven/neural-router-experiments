#!/bin/bash
#SBATCH --job-name=qoe-calfrac-smoke
#SBATCH --account=project_2018951
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1,nvme:50
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project_2018951/neural-router/logs/qoe-calfrac-smoke-%j.out
#SBATCH --error=/scratch/project_2018951/neural-router/logs/qoe-calfrac-smoke-%j.err
# =============================================================================
# H4 calibration-fraction smoke (gputest, 15 min): TAAS round-1 dual-reviewer
# (TAAS-1, AI-3) recommendation. Validates the --calibration-fraction CLI
# wiring with two distinct fraction values and confirms the fraction
# propagates to the CSV (flag set must equal flag consumed).
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

echo "=== qoe-calfrac smoke $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi -L
date
$OLLAMA serve > $NR_ROOT/logs/ollama-qoe-calfrac-smoke-$SLURM_JOB_ID.log 2>&1 &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT
for i in $(seq 1 30); do
    curl -sS --max-time 2 http://$OLLAMA_HOST/api/tags >/dev/null 2>&1 && break
    sleep 1
done
if ! $OLLAMA list | grep -q "^qwen2.5:7b"; then $OLLAMA pull qwen2.5:7b; fi

cd $NR_ROOT/code
SMOKE_OUT=$NR_ROOT/code/results/full/qoe_calfrac/by_task/smoke
# Clean state per run.
rm -rf $SMOKE_OUT
mkdir -p $SMOKE_OUT
SHARED_FLAGS=(--dataset D1 --strategies homogeneous,qoe_optimised --weight-presets balanced \
    --seeds 42 --backends "tier_mid:ollama/qwen2.5:7b" --max-events 50)

# Two cal-frac points exercise the flag plumbing.
# Flag-propagation: pass the directory label from bash, never reconstruct
# from a float in python (str(0.10) normalises to '0.1', mismatch).
declare -A CELL_DIRS
for FRAC in 0.10 0.20; do
    DIRNAME="frac_${FRAC}"  # stable label: 'frac_0.10', 'frac_0.20'
    echo "=== calibration_fraction=$FRAC -> $DIRNAME ==="
    python scripts/run_qoe.py "${SHARED_FLAGS[@]}" \
        --calibration-fraction $FRAC \
        --output-dir "$SMOKE_OUT/$DIRNAME"
    CELL_DIRS[$FRAC]="$SMOKE_OUT/$DIRNAME"
done

# Flag-propagation check: verify the recorded calibration_fraction in
# each CSV matches the value passed on the CLI. Read the column directly
# from the CSV — do not reconstruct paths in python.
python -c "
import pandas as pd, sys
checks = [
    (0.10, '${CELL_DIRS[0.10]}/qoe_D1.csv'),
    (0.20, '${CELL_DIRS[0.20]}/qoe_D1.csv'),
]
for frac, p in checks:
    df = pd.read_csv(p)
    if 'calibration_fraction' not in df.columns:
        print(f'FAIL flag-propagation: {p} has no calibration_fraction column (run_qoe.py did not record the flag)')
        sys.exit(1)
    seen = df['calibration_fraction'].unique()
    if not (len(seen) == 1 and abs(seen[0] - frac) < 1e-9):
        print(f'FAIL flag-propagation: {p} has calibration_fraction={seen}, expected {frac}')
        sys.exit(1)
    print(f'PASS flag-propagation: {p} carries calibration_fraction={frac}')
"

echo "=== DONE qoe-calfrac smoke ==="
date
