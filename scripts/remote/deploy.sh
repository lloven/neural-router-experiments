#!/usr/bin/env bash
# =============================================================================
# deploy.sh — Sync code and environment to the Neural Router VM.
#
# Usage:
#   ./scripts/remote/deploy.sh                 # git push + pull
#   ./scripts/remote/deploy.sh --setup         # run setup-vm.sh first
#   ./scripts/remote/deploy.sh --env           # also transfer .env
#   ./scripts/remote/deploy.sh --data          # also rsync data/
#   ./scripts/remote/deploy.sh --setup --env   # combine flags
#
# Prerequisites:
#   - SSH access to VM via nrouter-vm alias
#   - Git remote 'vm' configured: git remote add vm nrouter-vm:~/neural-router.git
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --- Configuration -----------------------------------------------------------

SSH_HOST="${NROUTER_SSH_HOST:-nrouter-vm}"
REMOTE_DIR="${NROUTER_REMOTE_DIR:-~/neural-router}"

# --- Helpers -----------------------------------------------------------------

info()  { printf "\033[1;34m[deploy]\033[0m %s\n" "$*"; }
ok()    { printf "\033[1;32m[deploy]\033[0m %s\n" "$*"; }
warn()  { printf "\033[1;33m[deploy]\033[0m %s\n" "$*"; }
err()   { printf "\033[1;31m[deploy]\033[0m %s\n" "$*" >&2; }

ssh_vm() { ssh "$SSH_HOST" "$@"; }

# --- Parse arguments ---------------------------------------------------------

DO_SETUP=false
DO_ENV=false
DO_DATA=false

for arg in "$@"; do
    case "$arg" in
        --setup)  DO_SETUP=true ;;
        --env)    DO_ENV=true ;;
        --data)   DO_DATA=true ;;
        --help|-h)
            echo "Usage: $0 [--setup] [--env] [--data]"
            echo ""
            echo "  --setup    Run setup-vm.sh before deploying"
            echo "  --env      Transfer .env file to VM"
            echo "  --data     Rsync data/ directory to VM"
            exit 0
            ;;
        *)
            err "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

# =============================================================================
# Step 0 (optional): Run setup
# =============================================================================

if $DO_SETUP; then
    info "Running VM setup first ..."
    "$SCRIPT_DIR/setup-vm.sh"
    ok "Setup complete, continuing with deploy."
fi

# =============================================================================
# Step 1: Test SSH connection
# =============================================================================

info "Testing SSH connection to ${SSH_HOST} ..."
if ! ssh_vm "echo ok" >/dev/null 2>&1; then
    err "Cannot connect to ${SSH_HOST}. Check your SSH config."
    exit 1
fi
ok "SSH connection OK."

# =============================================================================
# Step 2: Git push + pull
# =============================================================================

info "Pushing code to VM ..."
cd "$REPO_DIR"

# Ensure the 'vm' remote exists
if ! git remote get-url vm &>/dev/null; then
    info "Adding git remote 'vm' -> ${SSH_HOST}:${REMOTE_DIR}.git"
    git remote add vm "${SSH_HOST}:${REMOTE_DIR}.git"
fi

BRANCH=$(git rev-parse --abbrev-ref HEAD)
git push vm "${BRANCH}" 2>&1 || {
    warn "Git push failed. Trying with --force for first deploy ..."
    git push vm "${BRANCH}" --force
}
ok "Code pushed (branch: ${BRANCH})."

info "Pulling on VM ..."
ssh_vm "cd ${REMOTE_DIR} && git checkout ${BRANCH} 2>/dev/null || git checkout -b ${BRANCH} origin/${BRANCH} && git pull --ff-only"
ok "Code synced on VM."

# =============================================================================
# Step 3 (optional): Transfer .env
# =============================================================================

if $DO_ENV; then
    if [ -f "$REPO_DIR/.env" ]; then
        info "Transferring .env to VM ..."
        scp "$REPO_DIR/.env" "${SSH_HOST}:${REMOTE_DIR}/.env"
        ok ".env transferred."
    else
        warn "No .env file found at ${REPO_DIR}/.env"
    fi
fi

# =============================================================================
# Step 4 (optional): Rsync data/
# =============================================================================

if $DO_DATA; then
    if [ -d "$REPO_DIR/data" ]; then
        info "Syncing data/ to VM ..."
        rsync -avz --progress \
            -e "ssh" \
            "$REPO_DIR/data/" \
            "${SSH_HOST}:${REMOTE_DIR}/data/"
        ok "Data synced."
    else
        warn "No data/ directory found."
    fi
fi

# =============================================================================
# Step 5: Install dependencies if requirements.txt changed
# =============================================================================

info "Checking Python dependencies ..."
ssh_vm "bash -s" <<REMOTE_SCRIPT
set -euo pipefail

cd ${REMOTE_DIR}

if [ ! -d .venv ]; then
    python3 -m venv .venv
    echo "[deploy] Virtual environment created."
fi

# Check if requirements.txt has changed since last install
MARKER=".venv/.requirements_hash"
CURRENT_HASH=\$(md5sum requirements.txt 2>/dev/null | cut -d' ' -f1 || echo "none")
LAST_HASH=\$(cat "\$MARKER" 2>/dev/null || echo "")

if [ "\$CURRENT_HASH" != "\$LAST_HASH" ]; then
    echo "[deploy] requirements.txt changed, installing ..."
    .venv/bin/pip install --upgrade pip -q
    .venv/bin/pip install -r requirements.txt -q
    echo "\$CURRENT_HASH" > "\$MARKER"
    echo "[deploy] Dependencies installed."
else
    echo "[deploy] Dependencies up to date."
fi
REMOTE_SCRIPT
ok "Dependencies checked."

# =============================================================================
# Step 6: Verify Ollama health
# =============================================================================

info "Checking Ollama status ..."
ssh_vm "bash -s" <<'REMOTE_SCRIPT'
set -euo pipefail

if ! command -v ollama &>/dev/null; then
    echo "[deploy] WARNING: Ollama not installed."
    exit 0
fi

# Check if Ollama server is running
if ollama list &>/dev/null; then
    echo "[deploy] Ollama OK. Models:"
    ollama list
else
    echo "[deploy] WARNING: Ollama server not responding. Start with: ollama serve &"
fi
REMOTE_SCRIPT
ok "Ollama check done."

# =============================================================================
# Summary
# =============================================================================

ok "=== Deploy complete ==="
echo ""
echo "  Host:    ${SSH_HOST}"
echo "  Remote:  ${REMOTE_DIR}"
echo "  Branch:  ${BRANCH}"
echo ""
echo "  Next: ./scripts/remote/run-experiments.sh smoke"
echo ""
