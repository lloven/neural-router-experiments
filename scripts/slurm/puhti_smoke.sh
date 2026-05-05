#!/bin/bash
#SBATCH --job-name=nrouter-smoke
#SBATCH --account=project_2018951
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1,nvme:50
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project_2018951/neural-router/logs/smoke-%j.out
#SBATCH --error=/scratch/project_2018951/neural-router/logs/smoke-%j.err

NR_ROOT=/scratch/project_2018951/neural-router
OLLAMA=$NR_ROOT/bin/ollama-install/bin/ollama

# --- environment (CSC init scripts reference undefined vars; source with -u off) ---
source /etc/profile.d/lmod.sh 2>/dev/null || source /usr/share/lmod/lmod/init/bash
source /appl/profile/zz-csc-env.sh
module load pytorch/2.9
source $NR_ROOT/venv/py312-neural-router/bin/activate

set -eo pipefail

export OLLAMA_MODELS=$NR_ROOT/weights/ollama
# Force explicit IPv4: the compute node rejects AF_INET6 sockets, so litellm
# resolving 'localhost' → ::1 crashes with [Errno 97]. Use 127.0.0.1 everywhere.
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_API_BASE=http://127.0.0.1:11434
export HF_HOME=$NR_ROOT/weights/hf
export TOKENIZERS_PARALLELISM=false

echo "=== smoke job $SLURM_JOB_ID on $(hostname) ==="
echo "OLLAMA_HOST=$OLLAMA_HOST"
nvidia-smi -L
echo

# --- start ollama serve in background ---
$OLLAMA serve > $NR_ROOT/logs/ollama-$SLURM_JOB_ID.log 2>&1 &
OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT

# wait for ollama to be ready
for i in $(seq 1 30); do
    if curl -sS --max-time 2 http://$OLLAMA_HOST/api/tags >/dev/null 2>&1; then
        echo "ollama ready after ${i}s"
        break
    fi
    sleep 1
done

# --- sanity: ollama can see qwen2.5:7b and run it ---
echo "=== ollama list ==="
$OLLAMA list
echo
echo "=== ollama inference sanity ==="
echo "hi" | $OLLAMA run qwen2.5:7b "Respond with exactly one word: hello." 2>&1 | head -3
echo

# --- neural router smoke: 1 config, 1 seed, 20 events on D1 ---
cd $NR_ROOT/code
echo "=== neural router smoke ==="
python scripts/run_experiment.py \
    --dataset D1 \
    --configs A3 \
    --seeds 42 \
    --max-events 20 \
    --llm-model ollama/qwen2.5:7b \
    --output-tag puhti_smoke \
    --mode unit_smoke
echo
echo "=== DONE ==="
