#!/usr/bin/env bash
# =============================================================================
# collect-results.sh — Pull experiment results from Neural Router VM.
#
# Usage (run from LOCAL machine):
#   ./scripts/remote/collect-results.sh                    # pull all results
#   ./scripts/remote/collect-results.sh --logs             # also pull logs/
#   ./scripts/remote/collect-results.sh --mode crossover   # filter by mode
#   ./scripts/remote/collect-results.sh --mode full --logs # combine flags
#
# Results are merged into the local results/ directory with --backup.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --- Configuration -----------------------------------------------------------

SSH_HOST="${NROUTER_SSH_HOST:-remote-host}"
REMOTE_DIR="${NROUTER_REMOTE_DIR:-~/neural-router}"
LOCAL_RESULTS="${REPO_DIR}/results"

# --- Helpers -----------------------------------------------------------------

info()  { printf "\033[1;34m[collect]\033[0m %s\n" "$*"; }
ok()    { printf "\033[1;32m[collect]\033[0m %s\n" "$*"; }
warn()  { printf "\033[1;33m[collect]\033[0m %s\n" "$*"; }
err()   { printf "\033[1;31m[collect]\033[0m %s\n" "$*" >&2; }

ssh_vm() { ssh "$SSH_HOST" "$@"; }

# --- Parse arguments ---------------------------------------------------------

PULL_LOGS=false
MODE_FILTER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --logs)  PULL_LOGS=true; shift ;;
        --mode)  MODE_FILTER="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--mode <mode>] [--logs]"
            echo ""
            echo "  --mode <mode>  Filter results by mode (e.g., crossover, full, smoke)"
            echo "  --logs         Also pull logs/ directory"
            exit 0
            ;;
        *)
            err "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Step 1: Test SSH connection
# =============================================================================

info "Testing SSH connection to ${SSH_HOST} ..."
if ! ssh_vm "echo ok" >/dev/null 2>&1; then
    err "Cannot connect to ${SSH_HOST}."
    exit 1
fi
ok "SSH connection OK."

# =============================================================================
# Step 2: Pull results
# =============================================================================

if [ -n "$MODE_FILTER" ]; then
    REMOTE_RESULTS="${REMOTE_DIR}/results/${MODE_FILTER}"
    LOCAL_TARGET="${LOCAL_RESULTS}/${MODE_FILTER}"
    info "Pulling results for mode: ${MODE_FILTER} ..."
else
    REMOTE_RESULTS="${REMOTE_DIR}/results"
    LOCAL_TARGET="${LOCAL_RESULTS}"
    info "Pulling all results ..."
fi

# Check that remote directory exists
if ! ssh_vm "test -d ${REMOTE_RESULTS}"; then
    warn "Remote results directory does not exist: ${REMOTE_RESULTS}"
    warn "No results to collect."
    exit 0
fi

mkdir -p "$LOCAL_TARGET"

if command -v rsync &>/dev/null; then
    info "Syncing results via rsync ..."
    rsync -avz --progress \
        --backup --suffix=".vm-backup" \
        -e "ssh" \
        "${SSH_HOST}:${REMOTE_RESULTS}/" \
        "$LOCAL_TARGET/"
    ok "Results synced."
else
    info "rsync not found, using scp ..."
    TMPDIR=$(mktemp -d)
    scp -r "${SSH_HOST}:${REMOTE_RESULTS}/" "$TMPDIR/results"

    # Merge into local results
    find "$TMPDIR/results" -name "*.csv" -o -name "*.json" | while read -r f; do
        RELPATH="${f#$TMPDIR/results/}"
        DEST="${LOCAL_TARGET}/${RELPATH}"
        DEST_DIR=$(dirname "$DEST")
        mkdir -p "$DEST_DIR"

        if [ -f "$DEST" ]; then
            BASENAME=$(basename "$DEST")
            DEST="${DEST_DIR}/vm-${BASENAME}"
            info "Conflict: saving as ${DEST}"
        fi
        cp "$f" "$DEST"
    done
    rm -rf "$TMPDIR"
    ok "Results collected."
fi

# =============================================================================
# Step 3: Summary
# =============================================================================

echo ""
info "=== Local results summary ==="

# Count CSVs per stage/subdirectory
if [ -d "$LOCAL_RESULTS" ]; then
    TOTAL=$(find "$LOCAL_RESULTS" -name "*.csv" 2>/dev/null | wc -l | tr -d ' ')
    echo "  Total CSVs: ${TOTAL}"

    # Per-subdirectory breakdown
    for subdir in "$LOCAL_RESULTS"/*/; do
        [ -d "$subdir" ] || continue
        STAGE=$(basename "$subdir")
        COUNT=$(find "$subdir" -name "*.csv" 2>/dev/null | wc -l | tr -d ' ')
        echo "  ${STAGE}: ${COUNT} CSVs"
    done
else
    echo "  (no results directory)"
fi

# =============================================================================
# Step 4 (optional): Pull logs
# =============================================================================

if $PULL_LOGS; then
    info "Pulling logs ..."
    mkdir -p "${REPO_DIR}/logs/vm"
    rsync -avz --progress \
        -e "ssh" \
        "${SSH_HOST}:${REMOTE_DIR}/logs/" \
        "${REPO_DIR}/logs/vm/" \
        2>/dev/null || scp -r "${SSH_HOST}:${REMOTE_DIR}/logs/" "${REPO_DIR}/logs/vm/"
    ok "Logs saved to logs/vm/"
fi

echo ""
ok "=== Collection complete ==="
