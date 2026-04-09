#!/usr/bin/env bash
# =============================================================================
# setup-vm.sh — First-time VM setup for Neural Router experiments.
#
# Installs Python 3.10+, Ollama, pulls the model, sets up git bare repo,
# creates venv, and pre-downloads sentence-transformers model.
#
# Usage (run from LOCAL machine):
#   ./scripts/remote/setup-vm.sh
#
# Prerequisites:
#   - SSH access to the VM via the nrouter-vm alias
#   - VM has Ubuntu 22.04+ with NVIDIA GPU
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --- Configuration -----------------------------------------------------------

SSH_HOST="${NROUTER_SSH_HOST:-nrouter-vm}"
REMOTE_DIR="${NROUTER_REMOTE_DIR:-~/neural-router}"
MODEL="${NROUTER_MODEL:-qwen2.5:7b}"

# --- Helpers -----------------------------------------------------------------

info()  { printf "\033[1;34m[setup]\033[0m %s\n" "$*"; }
ok()    { printf "\033[1;32m[setup]\033[0m %s\n" "$*"; }
warn()  { printf "\033[1;33m[setup]\033[0m %s\n" "$*"; }
err()   { printf "\033[1;31m[setup]\033[0m %s\n" "$*" >&2; }

ssh_vm() { ssh "$SSH_HOST" "$@"; }

# =============================================================================
# Step 1: Test SSH connectivity
# =============================================================================

info "Testing SSH connection to ${SSH_HOST} ..."
if ! ssh_vm "echo ok" >/dev/null 2>&1; then
    err "Cannot connect to ${SSH_HOST}. Check your SSH config."
    err "Expected SSH config entry:"
    err ""
    err "  Host nrouter-vm"
    err "      HostName <vm-ip-or-hostname>"
    err "      User <username>"
    err "      IdentityFile ~/.ssh/<key>"
    exit 1
fi
ok "SSH connection OK."

# =============================================================================
# Step 2: Install system dependencies
# =============================================================================

info "Installing system dependencies ..."
ssh_vm "bash -s" <<'REMOTE_SCRIPT'
set -euo pipefail

# Python 3.10+, pip, venv, tmux
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-pip python3-venv tmux git curl

# Verify Python version
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "[setup] Python version: ${PY_VER}"

MAJOR=$(echo "$PY_VER" | cut -d. -f1)
MINOR=$(echo "$PY_VER" | cut -d. -f2)
if [ "$MAJOR" -lt 3 ] || { [ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 10 ]; }; then
    echo "[setup] ERROR: Python 3.10+ required, found ${PY_VER}" >&2
    exit 1
fi
REMOTE_SCRIPT
ok "System dependencies installed."

# =============================================================================
# Step 3: Install Ollama
# =============================================================================

info "Installing Ollama ..."
ssh_vm "bash -s" <<'REMOTE_SCRIPT'
set -euo pipefail

if command -v ollama &>/dev/null; then
    echo "[setup] Ollama already installed: $(ollama --version)"
else
    curl -fsSL https://ollama.ai/install.sh | sh
    echo "[setup] Ollama installed: $(ollama --version)"
fi
REMOTE_SCRIPT
ok "Ollama installed."

# =============================================================================
# Step 4: Pull model
# =============================================================================

info "Pulling model: ${MODEL} ..."
ssh_vm "ollama pull ${MODEL}"
ok "Model ${MODEL} pulled."

# =============================================================================
# Step 5: Verify GPU
# =============================================================================

info "Checking GPU ..."
ssh_vm "nvidia-smi" || warn "nvidia-smi not found or no GPU available."

# =============================================================================
# Step 6: Initialize bare repo + working copy
# =============================================================================

info "Setting up git repositories ..."
ssh_vm "bash -s" <<REMOTE_SCRIPT
set -euo pipefail

REMOTE_DIR="${REMOTE_DIR}"

if [ ! -d "\${REMOTE_DIR}.git" ]; then
    git init --bare "\${REMOTE_DIR}.git"
    echo "[setup] Bare repo created at \${REMOTE_DIR}.git"
else
    echo "[setup] Bare repo already exists at \${REMOTE_DIR}.git"
fi

if [ ! -d "\${REMOTE_DIR}" ]; then
    git clone "\${REMOTE_DIR}.git" "\${REMOTE_DIR}"
    echo "[setup] Working copy cloned to \${REMOTE_DIR}"
else
    echo "[setup] Working copy already exists at \${REMOTE_DIR}"
fi
REMOTE_SCRIPT
ok "Git repositories ready."

# =============================================================================
# Step 7: Create venv + install dependencies
# =============================================================================

info "Creating Python venv and installing dependencies ..."
ssh_vm "bash -s" <<REMOTE_SCRIPT
set -euo pipefail

cd ${REMOTE_DIR}

if [ ! -d .venv ]; then
    python3 -m venv .venv
    echo "[setup] Virtual environment created."
fi

.venv/bin/pip install --upgrade pip -q
if [ -f requirements.txt ]; then
    .venv/bin/pip install -r requirements.txt -q
    echo "[setup] Python dependencies installed."
else
    echo "[setup] No requirements.txt found (will install after first deploy)."
fi
REMOTE_SCRIPT
ok "Python environment ready."

# =============================================================================
# Step 8: Pre-download sentence-transformers model
# =============================================================================

info "Pre-downloading sentence-transformers model ..."
ssh_vm "bash -s" <<REMOTE_SCRIPT
set -euo pipefail

cd ${REMOTE_DIR}
.venv/bin/python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
print('[setup] Sentence-transformers model cached.')
" 2>/dev/null || echo "[setup] sentence-transformers not installed yet (will download after deploy)."
REMOTE_SCRIPT
ok "Model pre-download attempted."

# =============================================================================
# Step 9: Create working directories
# =============================================================================

info "Creating working directories ..."
ssh_vm "mkdir -p ${REMOTE_DIR}/{results,logs,data}"
ok "Directories created."

# =============================================================================
# Step 10: Print system summary
# =============================================================================

info "=== System Info Summary ==="
ssh_vm "bash -s" <<REMOTE_SCRIPT
set -euo pipefail

echo "  Hostname:  \$(hostname)"
echo "  OS:        \$(lsb_release -ds 2>/dev/null || cat /etc/os-release | head -1)"
echo "  Python:    \$(python3 --version)"
echo "  Ollama:    \$(ollama --version 2>/dev/null || echo 'not found')"
echo "  GPU:       \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "  VRAM:      \$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Disk:      \$(df -h ~ | tail -1 | awk '{print \$4 \" available\"}')"
echo "  Remote:    ${REMOTE_DIR}"
REMOTE_SCRIPT

ok "=== VM setup complete ==="
echo ""
echo "  Next steps:"
echo "    1. Add git remote: git remote add vm ${SSH_HOST}:${REMOTE_DIR}.git"
echo "    2. Deploy code:    ./scripts/remote/deploy.sh"
echo "    3. Run smoke test: ./scripts/remote/run-experiments.sh smoke"
echo ""
