#!/usr/bin/env bash
# =============================================================================
# run-experiments.sh — Run Neural Router experiments on the remote VM.
#
# Usage (run from LOCAL machine):
#   ./scripts/remote/run-experiments.sh smoke                # quick smoke test
#   ./scripts/remote/run-experiments.sh full                 # full experiment sweep
#   ./scripts/remote/run-experiments.sh full --resume        # resume interrupted
#   ./scripts/remote/run-experiments.sh crossover            # crossover experiments
#   ./scripts/remote/run-experiments.sh crossover --dry-run  # show plan, don't run
#   ./scripts/remote/run-experiments.sh stop                 # stop running experiments
#   ./scripts/remote/run-experiments.sh status               # check experiment status
#
# Flags:
#   --resume    Resume from last checkpoint (skip completed runs)
#   --dry-run   Show what would run without executing
#   --no-sync   Skip git push/pull before launching
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --- Configuration -----------------------------------------------------------

SSH_HOST="${NROUTER_SSH_HOST:-remote-host}"
REMOTE_DIR="${NROUTER_REMOTE_DIR:-~/neural-router}"

# --- Helpers -----------------------------------------------------------------

info()  { printf "\033[1;34m[run]\033[0m %s\n" "$*"; }
ok()    { printf "\033[1;32m[run]\033[0m %s\n" "$*"; }
warn()  { printf "\033[1;33m[run]\033[0m %s\n" "$*"; }
err()   { printf "\033[1;31m[run]\033[0m %s\n" "$*" >&2; }

ssh_vm() { ssh "$SSH_HOST" "$@"; }

# --- Parse action + flags ----------------------------------------------------

ACTION="${1:-help}"
shift || true

RESUME_FLAG=""
DRY_RUN=""
NO_SYNC=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)  RESUME_FLAG="--resume"; shift ;;
        --dry-run) DRY_RUN="1"; shift ;;
        --no-sync) NO_SYNC="1"; shift ;;
        --help|-h)
            echo "Usage: $0 {smoke|full|crossover|stop|status} [--resume] [--dry-run] [--no-sync]"
            exit 0
            ;;
        *)
            err "Unknown flag: $1"
            exit 1
            ;;
    esac
done

# --- Auto-sync: git push + pull before launch --------------------------------

sync_code() {
    if [[ -n "$NO_SYNC" ]]; then
        info "Skipping code sync (--no-sync)."
        return
    fi

    info "Syncing code to VM ..."
    cd "$REPO_DIR"
    BRANCH=$(git rev-parse --abbrev-ref HEAD)

    if ! git remote get-url vm &>/dev/null; then
        git remote add vm "${SSH_HOST}:${REMOTE_DIR}.git"
    fi

    git push vm "${BRANCH}" 2>&1 || warn "Git push failed (might be first deploy)."
    ssh_vm "cd ${REMOTE_DIR} && git pull --ff-only" 2>&1 || warn "Git pull failed on VM."
    ok "Code synced."
}

# --- tmux launcher -----------------------------------------------------------
# Launches a command inside a named tmux session on the VM.
# If the session already exists, warns and exits.

launch_tmux() {
    local session_name="$1"
    shift
    local remote_cmd="$*"

    info "Checking for existing tmux session '${session_name}' ..."
    if ssh_vm "tmux has-session -t ${session_name}" 2>/dev/null; then
        warn "tmux session '${session_name}' already running."
        warn "Attach with: ssh ${SSH_HOST} -t 'tmux attach -t ${session_name}'"
        warn "Or stop it:  $0 stop"
        exit 1
    fi

    info "Launching in tmux session '${session_name}' ..."
    ssh_vm "tmux new-session -d -s ${session_name} '${remote_cmd}'"
    ok "Experiment launched in tmux session '${session_name}'."
    echo ""
    echo "  Monitor:  ssh ${SSH_HOST} -t 'tmux attach -t ${session_name}'"
    echo "  Status:   $0 status"
    echo "  Stop:     $0 stop"
    echo ""
}

# =============================================================================
# Actions
# =============================================================================

case "$ACTION" in

# -----------------------------------------------------------------------------
# smoke — quick sanity check
# -----------------------------------------------------------------------------
smoke)
    sync_code
    info "=== Smoke Test ==="

    SESSION="nrouter-smoke"
    REMOTE_CMD="cd ${REMOTE_DIR} && .venv/bin/python scripts/run_all.py --mode unit_smoke"

    if [[ -n "$DRY_RUN" ]]; then
        info "[dry-run] Would run: ${REMOTE_CMD}"
        exit 0
    fi

    launch_tmux "$SESSION" "$REMOTE_CMD"
    ;;

# -----------------------------------------------------------------------------
# full — complete experiment sweep
# -----------------------------------------------------------------------------
full)
    sync_code
    info "=== Full Experiment Sweep ==="

    SESSION="nrouter-full"
    REMOTE_CMD="cd ${REMOTE_DIR} && .venv/bin/python scripts/run_all.py --mode full --ollama-slots 1 --api-slots 0"

    [[ -n "$RESUME_FLAG" ]] && REMOTE_CMD="${REMOTE_CMD} ${RESUME_FLAG}"

    if [[ -n "$DRY_RUN" ]]; then
        info "[dry-run] Would run: ${REMOTE_CMD}"
        exit 0
    fi

    launch_tmux "$SESSION" "$REMOTE_CMD"
    ;;

# -----------------------------------------------------------------------------
# crossover — crossover experiments
# -----------------------------------------------------------------------------
crossover)
    sync_code
    info "=== Crossover Experiments ==="

    SESSION="nrouter-crossover"
    REMOTE_CMD="cd ${REMOTE_DIR} && .venv/bin/python scripts/run_all.py --mode crossover --ollama-slots 1 --api-slots 0"

    [[ -n "$RESUME_FLAG" ]] && REMOTE_CMD="${REMOTE_CMD} ${RESUME_FLAG}"

    if [[ -n "$DRY_RUN" ]]; then
        info "[dry-run] Would run: ${REMOTE_CMD}"
        exit 0
    fi

    launch_tmux "$SESSION" "$REMOTE_CMD"
    ;;

# -----------------------------------------------------------------------------
# stop — kill running experiments
# -----------------------------------------------------------------------------
stop)
    info "Stopping experiments on VM ..."
    ssh_vm "bash -s" <<'REMOTE_SCRIPT'
set -euo pipefail

# Kill experiment processes
pkill -f "run_all.py" 2>/dev/null && echo "[run] Killed run_all.py" || echo "[run] No run_all.py running."
pkill -f "run_one.py" 2>/dev/null && echo "[run] Killed run_one.py" || echo "[run] No run_one.py running."

# Kill tmux sessions
for session in nrouter-smoke nrouter-full nrouter-crossover; do
    tmux kill-session -t "$session" 2>/dev/null && echo "[run] Killed tmux: $session" || true
done

echo "[run] All experiments stopped."
REMOTE_SCRIPT
    ok "Stop complete."
    ;;

# -----------------------------------------------------------------------------
# status — check what's running
# -----------------------------------------------------------------------------
status)
    info "=== VM Status ==="
    ssh_vm "bash -s" <<REMOTE_SCRIPT
set -euo pipefail

echo "--- tmux sessions ---"
tmux list-sessions 2>/dev/null || echo "  (none)"

echo ""
echo "--- experiment processes ---"
pgrep -af "run_all.py|run_one.py" 2>/dev/null || echo "  (none running)"

echo ""
echo "--- Ollama ---"
ollama list 2>/dev/null || echo "  (not available)"

echo ""
echo "--- GPU ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "  (no GPU)"

echo ""
echo "--- results count ---"
if [ -d "${REMOTE_DIR}/results" ]; then
    find "${REMOTE_DIR}/results" -name "*.csv" | wc -l | xargs echo "  CSVs:"
else
    echo "  (no results directory)"
fi

echo ""
echo "--- monitor ---"
cd ${REMOTE_DIR}
if [ -f scripts/monitor.py ] && [ -d .venv ]; then
    .venv/bin/python scripts/monitor.py --once 2>/dev/null || echo "  (monitor failed)"
else
    echo "  (monitor not available)"
fi
REMOTE_SCRIPT
    ;;

# -----------------------------------------------------------------------------
# help
# -----------------------------------------------------------------------------
help|--help|-h)
    echo "Usage: $0 {smoke|full|crossover|stop|status} [--resume] [--dry-run] [--no-sync]"
    echo ""
    echo "Actions:"
    echo "  smoke      Quick smoke test (~2 min)"
    echo "  full       Full experiment sweep"
    echo "  crossover  Crossover experiments"
    echo "  stop       Stop all running experiments"
    echo "  status     Check experiment and VM status"
    echo ""
    echo "Flags:"
    echo "  --resume   Resume from last checkpoint"
    echo "  --dry-run  Show plan without executing"
    echo "  --no-sync  Skip git push/pull before launch"
    exit 0
    ;;

*)
    err "Unknown action: $ACTION"
    echo "Run '$0 help' for usage."
    exit 1
    ;;
esac
