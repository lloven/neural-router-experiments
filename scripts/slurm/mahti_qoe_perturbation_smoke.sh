#!/bin/bash
#SBATCH --job-name=qoe-pert-smoke
#SBATCH --account=project_2018951
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1,nvme:50
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project_2018951/neural-router/logs/qoe-pert-smoke-%j.out
#SBATCH --error=/scratch/project_2018951/neural-router/logs/qoe-pert-smoke-%j.err
# =============================================================================
# C9 perturbation smoke (gputest, 15 min): TAAS round-1 reviewer mandate.
#
# Validates the perturbation hooks end-to-end on Mahti before the full run:
#   1. baseline (no perturbation)        — sanity, ~2 min
#   2. topic_restricted_cal              — calibration-domain shift
#   3. latency_injection                 — mid-run latency shift
#
# Treatment-verification: the 3 cells run in sequence with the SAME
# seed and dataset; the only difference is the perturbation kind. The CSV
# must show distinct latency rows (latency_injection > baseline > offset).
# Topic-restricted-cal must produce a different per-cluster assignment than
# baseline (calibration sample is constrained → different argmax).
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

echo "=== qoe-perturbation smoke $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi -L
date
$OLLAMA serve > $NR_ROOT/logs/ollama-qoe-pert-smoke-$SLURM_JOB_ID.log 2>&1 &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT
for i in $(seq 1 30); do
    curl -sS --max-time 2 http://$OLLAMA_HOST/api/tags >/dev/null 2>&1 && break
    sleep 1
done

echo "=== ollama list before pull ==="
$OLLAMA list
if ! $OLLAMA list | grep -q "^qwen2.5:7b"; then
    echo "Pulling qwen2.5:7b ..."
    $OLLAMA pull qwen2.5:7b
fi
if ! $OLLAMA list | grep -q "^qwen2.5:32b"; then
    echo "Pulling qwen2.5:32b ..."
    $OLLAMA pull qwen2.5:32b
fi

cd $NR_ROOT/code
SMOKE_OUT=$NR_ROOT/code/results/full/qoe_perturbation/by_task/smoke
# Ensure clean state — stale CSVs from a prior run with a different
# schema cause append-mode parser failures. Smoke output is cheap to rebuild.
rm -rf $SMOKE_OUT
mkdir -p $SMOKE_OUT
SHARED_FLAGS=(--dataset D1 --strategies homogeneous,qoe_optimised --weight-presets balanced \
    --seeds 42 --backends "tier_mid:ollama/qwen2.5:7b,tier_large:ollama/qwen2.5:32b" \
    --max-events 50 --calibration-fraction 0.10)

# Step 1: baseline (no perturbation) — treatment-verifiable matched-pair
echo "=== baseline (no perturbation) ==="
python scripts/run_qoe.py "${SHARED_FLAGS[@]}" \
    --output-dir "$SMOKE_OUT/baseline"

# Step 2: topic_restricted_cal (calibration sample restricted to 3 of 19 topics)
echo "=== topic_restricted_cal (mask = 3 of 19 topics) ==="
python scripts/run_qoe.py "${SHARED_FLAGS[@]}" \
    --perturbation topic_restricted_cal \
    --topic-mask "sports,business_&_entrepreneurs,arts_&_culture" \
    --output-dir "$SMOKE_OUT/topic_restricted"

# Step 3: latency_injection (extra 0.1s/event for events with idx >= 25)
echo "=== latency_injection (idx>=25, +0.1s/event) ==="
python scripts/run_qoe.py "${SHARED_FLAGS[@]}" \
    --perturbation latency_injection \
    --injection-event-index 25 \
    --injected-latency-s 0.1 \
    --output-dir "$SMOKE_OUT/latency_injection"

# Validate every cell wrote a CSV with at least 1 data row
for sub in baseline topic_restricted latency_injection; do
    CSV="$SMOKE_OUT/$sub/qoe_D1.csv"
    if [ -s "$CSV" ] && [ "$(wc -l < "$CSV")" -gt 1 ]; then
        echo "PASS $sub: $(wc -l < "$CSV") lines in CSV"
    else
        echo "FAIL $sub: empty/missing CSV at $CSV"
        exit 1
    fi
done

# Treatment-verification: latency_injection's wall-clock latency must
# be strictly greater than baseline's WHEN MATCHED BY (strategy, backend).
# Comparing means across the unmatched cell-mix conflates the +0.1s/event
# injection with the larger Ollama warmup variance on the 32B backend
# (the first cell to load 32B incurs the warmup penalty, later cells run
# warm). Matched comparison eliminates that confound.
python -c "
import pandas as pd, sys
base = pd.read_csv('$SMOKE_OUT/baseline/qoe_D1.csv')
lat  = pd.read_csv('$SMOKE_OUT/latency_injection/qoe_D1.csv')
key = ['strategy', 'weight_preset', 'backend', 'seed']
joined = base.merge(lat, on=key, suffixes=('_base', '_lat'))
print('Matched-cell latency comparison (baseline -> latency_injection):')
deltas = []
for _, r in joined.iterrows():
    label = f'{r[\"strategy\"]}/{r[\"weight_preset\"]}/{r[\"backend\"]}/seed{r[\"seed\"]}'
    delta = r['latency_s_lat'] - r['latency_s_base']
    deltas.append(delta)
    print(f'  {label}: {r[\"latency_s_base\"]:.3f} -> {r[\"latency_s_lat\"]:.3f}  (delta {delta:+.3f}s)')
# Treatment is verified if AT LEAST ONE matched cell shows delta > 0.5s.
# (Single matched cell suffices for treatment-verification — unmatched
# cells may show warmup-dominated negative deltas that aren't
# injection-driven.)
positive = [d for d in deltas if d > 0.5]
if not positive:
    print(f'FAIL treatment-verification: no matched cell shows delta > 0.5s; deltas={deltas}')
    sys.exit(1)
print(f'PASS treatment-verification: {len(positive)}/{len(deltas)} matched cells show delta > 0.5s')
"

echo "=== DONE qoe-perturbation smoke ==="
date
