#!/usr/bin/env bash
# =============================================================================
# check-vm.sh — Health check for the Neural Router VM.
#
# Usage (run from LOCAL machine):
#   ./scripts/remote/check-vm.sh
#
# Checks: SSH, GPU, Ollama, disk, running experiments, manifest state.
# =============================================================================

set -euo pipefail

SSH_HOST="${NROUTER_SSH_HOST:-remote-host}"
REMOTE_DIR="${NROUTER_REMOTE_DIR:-~/neural-router}"

# --- Helpers -----------------------------------------------------------------

info()  { printf "\033[1;34m[check]\033[0m %s\n" "$*"; }
ok()    { printf "\033[1;32m[check]\033[0m %s\n" "$*"; }
warn()  { printf "\033[1;33m[check]\033[0m %s\n" "$*"; }
err()   { printf "\033[1;31m[check]\033[0m %s\n" "$*" >&2; }

ssh_vm() { ssh "$SSH_HOST" "$@"; }

# =============================================================================
# Check 1: SSH connectivity
# =============================================================================

info "=== Neural Router VM Health Check ==="
echo ""

info "SSH connectivity ..."
if ! ssh_vm "echo ok" >/dev/null 2>&1; then
    err "Cannot connect to ${SSH_HOST}."
    exit 1
fi
ok "SSH: connected to ${SSH_HOST}"

# =============================================================================
# Check 2: GPU
# =============================================================================

info "GPU status ..."
ssh_vm "bash -s" <<'REMOTE_SCRIPT'
if command -v nvidia-smi &>/dev/null; then
    echo "  GPU:"
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu \
        --format=csv,noheader 2>/dev/null | while read -r line; do
        echo "    $line"
    done
else
    echo "  GPU: nvidia-smi not found"
fi
REMOTE_SCRIPT

# =============================================================================
# Check 3: Ollama
# =============================================================================

info "Ollama status ..."
ssh_vm "bash -s" <<'REMOTE_SCRIPT'
if command -v ollama &>/dev/null; then
    echo "  Version: $(ollama --version 2>/dev/null || echo 'unknown')"
    echo "  Models:"
    ollama list 2>/dev/null | sed 's/^/    /' || echo "    (server not responding)"
else
    echo "  Ollama: not installed"
fi
REMOTE_SCRIPT

# =============================================================================
# Check 4: Disk space
# =============================================================================

info "Disk space ..."
ssh_vm "df -h ~ | tail -1 | awk '{printf \"  Used: %s / %s (%s), Available: %s\n\", \$3, \$2, \$5, \$4}'"

# =============================================================================
# Check 5: Running processes
# =============================================================================

info "Running experiments ..."
ssh_vm "bash -s" <<REMOTE_SCRIPT
echo "  Processes:"
pgrep -af "run_all.py|run_one.py|run_experiment.py" 2>/dev/null | sed 's/^/    /' || echo "    (none)"

echo "  tmux sessions:"
tmux list-sessions 2>/dev/null | sed 's/^/    /' || echo "    (none)"
REMOTE_SCRIPT

# =============================================================================
# Check 6: Manifest state
# =============================================================================

info "Manifest state ..."
ssh_vm "bash -s" <<REMOTE_SCRIPT
MANIFEST="${REMOTE_DIR}/results/manifest.json"
if [ -f "\$MANIFEST" ]; then
    cd ${REMOTE_DIR}
    if [ -f scripts/monitor.py ] && [ -d .venv ]; then
        .venv/bin/python scripts/monitor.py --once 2>/dev/null | sed 's/^/  /' || echo "  (monitor error)"
    else
        echo "  Manifest exists: \$MANIFEST"
        echo "  Size: \$(wc -c < "\$MANIFEST") bytes"
    fi
else
    echo "  No manifest found."
fi
REMOTE_SCRIPT

echo ""
ok "=== Health check complete ==="
