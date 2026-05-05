#!/bin/bash
#SBATCH --job-name=nrouter-int
#SBATCH --account=project_2018951
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1,nvme:100
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/project_2018951/neural-router/logs/int-%j.out
#SBATCH --error=/scratch/project_2018951/neural-router/logs/int-%j.err

NR_ROOT=/scratch/project_2018951/neural-router
OLLAMA=$NR_ROOT/bin/ollama-install/bin/ollama

# --- environment (CSC init scripts reference undefined vars; source with -u off) ---
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

echo "=== integration smoke $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi -L
date
echo

# --- start ollama serve ---
$OLLAMA serve > $NR_ROOT/logs/ollama-$SLURM_JOB_ID.log 2>&1 &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT
for i in $(seq 1 30); do
    curl -sS --max-time 2 http://$OLLAMA_HOST/api/tags >/dev/null 2>&1 && { echo "ollama ready after ${i}s"; break; }
    sleep 1
done

# --- integration smoke: D1-D3, 3 configs, 3 baselines, 100 events, seed 42 ---
cd $NR_ROOT/code
echo "=== run_experiment --mode integration_smoke ==="
python scripts/run_experiment.py \
    --dataset all \
    --llm-model ollama/qwen2.5:7b \
    --mode integration_smoke \
    --baselines bm25,sbert \
    --output-tag puhti_int2 \
    --resume
echo
echo "=== DONE ==="
date
