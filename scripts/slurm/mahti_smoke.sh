#!/bin/bash
#SBATCH --job-name=nrouter-m-smoke
#SBATCH --account=project_2018951
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1,nvme:50
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project_2018951/neural-router/logs/m-smoke-%j.out
#SBATCH --error=/scratch/project_2018951/neural-router/logs/m-smoke-%j.err

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

echo "=== Mahti smoke $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi -L
date

$OLLAMA serve > $NR_ROOT/logs/ollama-$SLURM_JOB_ID.log 2>&1 &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT
for i in $(seq 1 30); do
    curl -sS --max-time 2 http://$OLLAMA_HOST/api/tags >/dev/null 2>&1 && { echo "ollama ready after ${i}s"; break; }
    sleep 1
done

echo "=== ollama list ==="
$OLLAMA list
echo
echo "=== Qwen-2.5-32B inference sanity ==="
echo "hi" | $OLLAMA run qwen2.5:32b "Respond with exactly one word: hello." 2>&1 | tail -3
echo

# Neural Router smoke: single ablation run at 100 events on D1 with 32B
cd $NR_ROOT/code
echo "=== Neural Router smoke: D1 A3 seed42, 100 events, Qwen-2.5-32B ==="
python scripts/run_experiment.py \
    --dataset D1 \
    --configs A3 \
    --seeds 42 \
    --max-events 100 \
    --llm-model ollama/qwen2.5:32b \
    --output-tag mahti_smoke_32b \
    --output-dir $NR_ROOT/code/results/full/ablation/by_task/mahti_smoke_32b
echo
echo "=== DONE ==="
date
