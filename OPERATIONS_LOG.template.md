# Neural Router Experiment Operations Log

Track all manual interventions: starts, stops, restarts, cleanups, budget changes, configuration changes, and notable events.

## Status Summary

| Date | Done | Running | Pending | Failed | Total | Notes |
|---|---|---|---|---|---|---|
<!-- Auto-populated by src/ops_log.py at milestones (every 25 runs) and manual entries -->

## Operations

<!-- Entries are appended automatically by scripts/restart.sh and src/ops_log.py -->
<!-- Add manual notes (budget changes, config decisions, investigation results) by editing directly -->

## Configuration History

| Date | Change | Reason |
|---|---|---|
<!-- Document experiment config changes here -->

## Budget Tracking

| Date | Event | Cumulative spend (approx) |
|---|---|---|
<!-- Track API spend here (not committed to git) -->

## Known Issues

1. **Silent failure on "usage limit":** The LLM client treats "usage limit" as a fatal error (no retry). This is correct behavior (retrying won't help until budget is raised). The orchestrator now auto-logs API limit hits to this file via `src/ops_log.py`.

2. **Empty result files:** Some runs may have written result CSV headers but no data rows before the API limit hit. The manifest system detects these (requires data rows), but verify no empty-result runs slipped through as "done."

## Restart Procedure

**Use the wrapper script** (preferred):
```bash
scripts/restart.sh                          # default: --api-slots 1 --ollama-slots 1
scripts/restart.sh --api-slots 2            # increase API parallelism
scripts/restart.sh --api-slots 2 --ollama-slots 1 --dry-run   # preview without launching
```

The script automatically:
1. Kills any stale orchestrator processes
2. Resets failed and stale-running manifest entries to pending
3. Appends a timestamped restart entry to this file
4. Launches the orchestrator with nohup (logs to `logs/orchestrator.log`)
5. Prints the new PID and manifest state

**Manual procedure** (if wrapper script unavailable):
1. Verify API budget has headroom: check Anthropic console
2. Kill stale processes: `pkill -f run_all.py`
3. Reset manifest: `.venv/bin/python -c "from pathlib import Path; from src.manifest import Manifest; m = Manifest.load(Path('results/full/manifest.json')); print(m.reset_failed_to_pending(), 'reset'); m.save(Path('results/full/manifest.json'))"`
4. Launch: `nohup .venv/bin/python scripts/run_all.py --mode full --api-slots 1 --ollama-slots 1 >> logs/orchestrator.log 2>&1 &`
5. Log the restart in this file manually

## Auto-Logging

The orchestrator (`scripts/run_all.py`) automatically appends entries to this file via `src/ops_log.py`:
- **Startup:** timestamp, manifest state, slot configuration
- **Milestones:** every 25 completed runs, a status summary row
- **API limit hits:** when a run fails with "usage limit" error
- **Shutdown:** timestamp, final state, reason (clean exit, signal, error)

Manual notes (budget changes, config decisions, investigation results) should still be added by editing this file directly.

## Setup

On first run, if `OPERATIONS_LOG.md` does not exist, `src/ops_log.py` creates it from this template. The actual log (`OPERATIONS_LOG.md`) is gitignored; this template is tracked.
