# Neural Router Experiment Operations Log

Track all manual interventions: starts, stops, restarts, cleanups, budget changes, configuration changes, and notable events.

## Status Summary

| Date | Done | Running | Pending | Failed | Total | Notes |
|---|---|---|---|---|---|---|
| 2026-03-20 ~01:00 | ~93 | 0 | ~189 | ~90 | 372 | API limit hit, orchestrator stopped |
| 2026-03-22 ~14:00 | 100 | 3 | 186 | 76 | 372 | After restart + budget raise; 76 failed (API limit) |

## Operations

### 2026-03-22 — Budget raised, experiments restarted

- **Action:** Anthropic API monthly budget limit raised by user
- **Trigger:** Haiku and Sonnet experiments both hitting "usage limit" fatal error
- **Result:** Orchestrator restarted, resumed from checkpoints. Progress: 93 -> 100 done before hitting limit again.
- **Failed runs (76):** Need to be reset from "failed" to "pending" in manifest before next restart, OR the orchestrator needs to auto-retry failed runs on restart.

### 2026-03-22 16:26 — Manifest reset + orchestrator restarted (PID 33322)

- **Action:** Added `reset_failed_to_pending()` method to manifest.py (with test). Reset 76 failed + 10 stale running -> pending. Killed stale orchestrator processes (PID 95417, 33270). Launched new orchestrator with `--api-slots 1 --ollama-slots 1` (reduced parallelism to avoid burning budget too fast).
- **State after reset:** 101 done, 271 pending, 0 failed, 0 running
- **State after launch:** 101 done, 268 pending, 2 running, 0 failed
- **Logging:** Output appended to `logs/orchestrator.log`

### 2026-03-22 — Budget raised again

- **Action:** User raised API budget limit again (second time today)
- **Status:** Experiments restarted (see entry above).

### 2026-03-20 ~13:30 UTC — First API limit hit

- **Action:** None (discovered retroactively)
- **Trigger:** Anthropic billing limit reached after ~$170 spend
- **Result:** All subsequent LLM calls returned "usage limit" fatal error. Runs after this point produced empty results or failed silently.
- **Impact:** Some runs between the limit hit and orchestrator stop may have written empty result files. These were detected by the manifest system (requires result_file to exist AND contain data rows).

### 2026-03-19 — Experiments started

- **Action:** Full experiment matrix launched via orchestrator
- **Config:** 372 total runs (ablation D1/D3 x 7 configs x 3 seeds x 2 models + baselines + cross-dataset)
- **Models:** claude-3-haiku-20240307 (Haiku), claude-3-5-sonnet-20241022 (Sonnet), qwen2.5-7b (local Ollama)
- **Parallelism:** max_parallel=1 for Haiku, max_parallel=2 for Sonnet, max_parallel=1 for Qwen
- **Environment:** Local laptop (MacBook)

## Configuration History

| Date | Change | Reason |
|---|---|---|
| 2026-03-19 | Initial config: 372 runs, max_parallel per model defaults | Start of experiment campaign |
| 2026-03-22 | BART-large-MNLI zero-shot baseline added to manuscript (not yet in experiment code) | IoTJ reviewer feedback |

## Budget Tracking

| Date | Event | Cumulative spend (approx) |
|---|---|---|
| 2026-03-19 | Experiments started | $0 |
| 2026-03-20 | First limit hit | ~$170 |
| 2026-03-22 | Budget raised (first time) | ~$170 |
| 2026-03-22 | Second limit hit | ~$180 (est.) |
| 2026-03-22 | Budget raised (second time) | ~$180 |

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
