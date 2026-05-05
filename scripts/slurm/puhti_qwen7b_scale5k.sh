#!/bin/bash
#SBATCH --job-name=q7b-scale5k
#SBATCH --account=project_2018951
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1,nvme:100
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/project_2018951/neural-router/logs/q7b-scale5k-%j.out
#SBATCH --error=/scratch/project_2018951/neural-router/logs/q7b-scale5k-%j.err
# =============================================================================
# Fill the missing 5000-event scaling point on D3 that the 6-h-walltime
# scaling job (34225741) timed out on.
#
# run_scaling.py iterates an internally-fixed list. We override via an
# inline Python wrapper that calls scale_events(event_counts=[5000])
# directly, so only the missing point runs (no need to re-do 50..2000).
#
# Estimated cost: ~6 h × 10 BU/h = 60 BU. Same magnitude as the original
# job but only the tail point.
#
# Submit:
#   sbatch scripts/slurm/puhti_qwen7b_scale5k.sh
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

DATASET=D3
TAG="qwen7b_scale_evt5000_${DATASET}"

echo "=== ${TAG} on $(hostname) ==="
nvidia-smi -L
date

$OLLAMA serve > $NR_ROOT/logs/ollama-$SLURM_JOB_ID.log 2>&1 &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT
for i in $(seq 1 30); do
    curl -sS --max-time 2 http://$OLLAMA_HOST/api/tags >/dev/null 2>&1 && { echo "ollama ready after ${i}s"; break; }
    sleep 1
done

cd $NR_ROOT/code
TASK_OUT=$NR_ROOT/code/results/full/scaling/by_task/$TAG
mkdir -p $TASK_OUT

# Call scale_events directly with a single-element event_counts to avoid
# repeating points already covered by the original 6 h job.
python -c "
import sys
sys.path.insert(0, '.')
from pathlib import Path
from src.data import load_dataset_by_name
from src.embeddings import EmbeddingModel
from src.llm import LLMClient
from scripts.run_scaling import scale_events

ds = load_dataset_by_name('$DATASET', cache_dir='data')
embedder = EmbeddingModel('all-MiniLM-L6-v2', cache_dir='data')
llm = LLMClient(model='ollama/qwen2.5:7b', timeout=900, max_tokens=16384)
out_dir = Path('$TASK_OUT')

print(f'Running 5000-event scaling point on {ds.name} ({ds.num_events} events available)')
scale_events(
    dataset=ds,
    embedder=embedder,
    llm_client=llm,
    seed=42,
    output_dir=out_dir,
    batch_size=10,        # small batch for ollama
    llm_timeout=900,
    event_counts=[5000],  # ONLY this point
)
"

echo "=== DONE $TAG ==="
date
