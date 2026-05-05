#!/bin/bash
#SBATCH --job-name=nrouter-bart
#SBATCH --account=project_2018951
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1,nvme:50
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project_2018951/neural-router/logs/bart-%j.out
#SBATCH --error=/scratch/project_2018951/neural-router/logs/bart-%j.err
# =============================================================================
# DistilBART-MNLI zero-shot baseline runner (gputest, 15 min cap).
#
# SMOKE_LEVEL parametric (per L23):
#   integration: D2 with --max-events 50, ~30 s. Schema validation.
#   d2         : D2 full corpus, ~10 min on A100, ~1.5 BU.
#   d3         : D3 full corpus, ~3 min on A100, ~0.5 BU.
# Defaults to integration. The D1 row already exists from laptop CPU
# (latency 10323s = 2h 52m; F1=0.434).
# =============================================================================

set -eo pipefail
NR_ROOT=/scratch/project_2018951/neural-router
source /etc/profile.d/lmod.sh 2>/dev/null || source /usr/share/lmod/lmod/init/bash
source /appl/profile/zz-csc-env.sh
module load pytorch/2.9
source $NR_ROOT/venv/py312-neural-router/bin/activate

export HF_HOME=$NR_ROOT/weights/hf
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=0

echo "=== bart $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi -L
date

SMOKE_LEVEL=${SMOKE_LEVEL:-integration}
case "$SMOKE_LEVEL" in
    integration) DATASET=D2; MAX_E=50;    OUT_TAG="d2_smoke" ;;
    d2)          DATASET=D2; MAX_E=200;   OUT_TAG="d2"       ;;  # ~12 min, gputest-feasible
    d3)          DATASET=D3; MAX_E=1000;  OUT_TAG="d3"       ;;  # ~9 min, matches qwen-7b cap
    *) echo "unknown SMOKE_LEVEL=$SMOKE_LEVEL (use integration|d2|d3)" >&2; exit 2 ;;
esac
echo "=== SMOKE_LEVEL=$SMOKE_LEVEL  dataset=$DATASET  max_events=$MAX_E ==="

cd $NR_ROOT/code
OUT_DIR=$NR_ROOT/code/results/full/ablation
mkdir -p $OUT_DIR
OUT_CSV=$OUT_DIR/${DATASET}_baseline_zero_shot_results.csv
[ "$OUT_TAG" = "d2_smoke" ] && OUT_CSV=$NR_ROOT/code/results/full/ablation/${DATASET}_baseline_zero_shot_smoke.csv

python scripts/run_baseline_zero_shot.py \
    --dataset $DATASET \
    --max-events $MAX_E \
    --output $OUT_CSV \
    --model valhalla/distilbart-mnli-12-1

# L30 content check.
if [ -s "$OUT_CSV" ] && [ "$(wc -l < "$OUT_CSV")" -gt 1 ]; then
    echo "PASS: BART CSV has data"
    head -2 "$OUT_CSV"
else
    echo "FAIL: empty/missing CSV at $OUT_CSV"
    exit 1
fi
echo "=== DONE bart-$SMOKE_LEVEL ==="
date
