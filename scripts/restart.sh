#!/usr/bin/env bash
# restart.sh — Automated restart procedure for the neural-router orchestrator.
#
# Steps:
#   1. Kill stale orchestrator processes (pgrep -f run_all.py)
#   2. Reset failed and stale-running runs in manifest
#   3. Append timestamped entry to OPERATIONS_LOG.md
#   4. Launch orchestrator with nohup (unless --dry-run)
#   5. Print PID and current manifest state
#
# Usage:
#   ./scripts/restart.sh --mode full
#   ./scripts/restart.sh --mode full --api-slots 2 --ollama-slots 1
#   ./scripts/restart.sh --mode full --dry-run

set -euo pipefail

# Defaults
API_SLOTS=1
OLLAMA_SLOTS=1
DRY_RUN=false
MODE=""
RESULTS_ROOT=""
PROJECT_ROOT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --api-slots)   API_SLOTS="$2"; shift 2 ;;
        --ollama-slots) OLLAMA_SLOTS="$2"; shift 2 ;;
        --dry-run)     DRY_RUN=true; shift ;;
        --mode)        MODE="$2"; shift 2 ;;
        --results-root) RESULTS_ROOT="$2"; shift 2 ;;
        --project-root) PROJECT_ROOT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$MODE" ]]; then
    echo "Error: --mode is required" >&2
    exit 1
fi

# Resolve project root (default: directory containing this script's parent)
if [[ -z "$PROJECT_ROOT" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

if [[ -z "$RESULTS_ROOT" ]]; then
    RESULTS_ROOT="$PROJECT_ROOT/results"
fi

MANIFEST_PATH="$RESULTS_ROOT/$MODE/manifest.json"
OPS_LOG="$PROJECT_ROOT/OPERATIONS_LOG.md"
LOG_DIR="$PROJECT_ROOT/logs"

# Determine Python interpreter
if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    PYTHON="$PROJECT_ROOT/.venv/bin/python"
else
    PYTHON="python3"
fi

echo "=== Neural Router Restart ==="
echo "Mode:         $MODE"
echo "API slots:    $API_SLOTS"
echo "Ollama slots: $OLLAMA_SLOTS"
echo "Manifest:     $MANIFEST_PATH"
echo "Ops log:      $OPS_LOG"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Kill stale orchestrator processes
# ---------------------------------------------------------------------------
echo "--- Step 1: Checking for stale orchestrator processes ---"
STALE_PIDS=$(pgrep -f "run_all.py" 2>/dev/null || true)
if [[ -n "$STALE_PIDS" ]]; then
    echo "Found stale PIDs: $STALE_PIDS"
    if [[ "$DRY_RUN" == "false" ]]; then
        for pid in $STALE_PIDS; do
            echo "  Killing PID $pid"
            kill "$pid" 2>/dev/null || true
        done
        sleep 1
    else
        echo "  (dry-run: would kill these PIDs)"
    fi
else
    echo "  No stale processes found."
fi

# ---------------------------------------------------------------------------
# Step 2: Reset failed and stale-running runs in manifest
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 2: Resetting manifest ---"

if [[ ! -f "$MANIFEST_PATH" ]]; then
    echo "  Manifest not found at $MANIFEST_PATH"
    echo "  Will be created on orchestrator launch."
else
    # Capture state before reset
    STATE_BEFORE=$("$PYTHON" -c "
import sys, json
sys.path.insert(0, '$PROJECT_ROOT')
from src.manifest import Manifest
m = Manifest.load('$MANIFEST_PATH')
runs = m.runs.values()
done = sum(1 for r in runs if r.status == 'done')
failed = sum(1 for r in runs if r.status == 'failed')
pending = sum(1 for r in runs if r.status == 'pending')
running = sum(1 for r in runs if r.status == 'running')
total = len(m.runs)
print(f'{done} done, {pending} pending, {failed} failed, {running} running (total {total})')
")
    echo "  Before: $STATE_BEFORE"

    # Reset failed -> pending and running -> pending
    "$PYTHON" -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
from src.manifest import Manifest
m = Manifest.load('$MANIFEST_PATH')
reset_count = m.reset_failed_to_pending()
stale_count = 0
for r in m.runs.values():
    if r.status == 'running':
        r.status = 'pending'
        stale_count += 1
m.save('$MANIFEST_PATH')
print(f'  Reset {reset_count} failed + {stale_count} stale running -> pending')
"

    # Capture state after reset
    STATE_AFTER=$("$PYTHON" -c "
import sys, json
sys.path.insert(0, '$PROJECT_ROOT')
from src.manifest import Manifest
m = Manifest.load('$MANIFEST_PATH')
runs = m.runs.values()
done = sum(1 for r in runs if r.status == 'done')
failed = sum(1 for r in runs if r.status == 'failed')
pending = sum(1 for r in runs if r.status == 'pending')
running = sum(1 for r in runs if r.status == 'running')
total = len(m.runs)
print(f'{done} done, {pending} pending, {failed} failed, {running} running (total {total})')
")
    echo "  After:  $STATE_AFTER"
fi

# ---------------------------------------------------------------------------
# Step 3: Append entry to OPERATIONS_LOG.md
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 3: Updating operations log ---"

TIMESTAMP=$(date "+%Y-%m-%d %H:%M")

"$PYTHON" -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
from src.ops_log import _ensure_ops_log, _append_status_row, _append_operations_entry
from src.manifest import Manifest
from pathlib import Path

ops_log = Path('$OPS_LOG')
_ensure_ops_log(ops_log)

# Get current manifest state
try:
    m = Manifest.load('$MANIFEST_PATH')
    runs = m.runs.values()
    state = {
        'done': sum(1 for r in runs if r.status == 'done'),
        'pending': sum(1 for r in runs if r.status == 'pending'),
        'failed': sum(1 for r in runs if r.status == 'failed'),
        'running': sum(1 for r in runs if r.status == 'running'),
        'total': len(m.runs),
    }
except Exception:
    state = {'done': 0, 'pending': 0, 'failed': 0, 'running': 0, 'total': 0}

notes = 'Automated restart (api_slots=$API_SLOTS, ollama_slots=$OLLAMA_SLOTS)'
_append_status_row(ops_log, state, notes)

body = (
    '- **Action:** Automated restart via restart.sh\n'
    '- **Before reset:** $STATE_BEFORE\n'
    '- **After reset:** $STATE_AFTER\n'
    '- **Slots:** api_slots=$API_SLOTS, ollama_slots=$OLLAMA_SLOTS'
)
_append_operations_entry(ops_log, 'Automated restart', body)

print('  Ops log updated.')
"

# ---------------------------------------------------------------------------
# Step 4: Launch orchestrator (unless dry-run)
# ---------------------------------------------------------------------------
echo ""
if [[ "$DRY_RUN" == "true" ]]; then
    echo "--- Step 4: Skipped (dry-run mode) ---"
    echo ""
    echo "=== Dry run complete ==="
    echo "State: $STATE_AFTER"
    exit 0
fi

echo "--- Step 4: Launching orchestrator ---"
mkdir -p "$LOG_DIR"

nohup "$PYTHON" scripts/run_all.py \
    --mode "$MODE" \
    --api-slots "$API_SLOTS" \
    --ollama-slots "$OLLAMA_SLOTS" \
    --results-root "$RESULTS_ROOT" \
    >> "$LOG_DIR/orchestrator.log" 2>&1 &

NEW_PID=$!
echo "  Orchestrator launched with PID $NEW_PID"
echo "  Log: $LOG_DIR/orchestrator.log"

# ---------------------------------------------------------------------------
# Step 5: Print summary
# ---------------------------------------------------------------------------
echo ""
echo "=== Restart complete ==="
echo "PID:    $NEW_PID"
echo "State:  $STATE_AFTER"
echo "Slots:  api=$API_SLOTS, ollama=$OLLAMA_SLOTS"
