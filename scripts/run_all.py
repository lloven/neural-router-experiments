#!/usr/bin/env python3
"""Central orchestrator replacing DVC for experiment management.

Generates (or loads) a manifest, then schedules runs in parallel
respecting slot limits for ollama and API backends.

Usage::

    python scripts/run_all.py --mode full
    python scripts/run_all.py --mode full --ollama-slots 1 --api-slots 2
    python scripts/run_all.py --mode full --dry-run
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_profile
from src.manifest import Manifest, SlotPool, generate_runs, sort_by_priority
from src.ops_log import get_manifest_state, log_api_limit, log_milestone, log_shutdown, log_startup


# ---------------------------------------------------------------------------
# Testable helper functions
# ---------------------------------------------------------------------------

def create_or_load_manifest(
    manifest_path: Path,
    mode: str,
    profile: dict,
) -> Manifest:
    """Load an existing manifest or create one from the profile.

    On load, any stale "running" entries (from a crashed previous run) are
    reset to "pending" so they get retried.
    """
    manifest_path = Path(manifest_path)

    if manifest_path.exists():
        manifest = Manifest.load(manifest_path)
        # Reset stale "running" entries (crash recovery)
        changed = False
        for run in manifest.runs.values():
            if run.status == "running":
                run.status = "pending"
                changed = True
        if changed:
            manifest.save(manifest_path)
    else:
        runs = generate_runs(profile)
        manifest = Manifest(mode=mode, runs={r.run_id: r for r in runs})
        manifest.save(manifest_path)

    return manifest


def print_dry_run(manifest: Manifest) -> None:
    """Print the run matrix without launching any processes."""
    pending = sort_by_priority(manifest.get_pending_runs())
    done = [r for r in manifest.runs.values() if r.status == "done"]
    failed = [r for r in manifest.runs.values() if r.status == "failed"]

    print(f"Mode: {manifest.mode}")
    print(f"Total runs: {len(manifest.runs)}")
    print(f"  pending: {len(pending)}")
    print(f"  done:    {len(done)}")
    print(f"  failed:  {len(failed)}")
    print()

    if pending:
        print("Pending runs (in priority order):")
        for r in pending:
            print(f"  {r.run_id}  [{r.slot_type}]  {r.stage}/{r.dataset}/{r.config}")
    else:
        print("No pending runs.")


def launch_pending_runs(
    manifest: Manifest,
    pool: SlotPool,
    active: dict[str, subprocess.Popen],
    manifest_path: Path,
    mode: str,
) -> dict[str, subprocess.Popen]:
    """Launch pending runs that fit within available slots.

    Returns the updated active-processes dict (including previously active).
    """
    # Work on a copy so caller sees updates
    active = dict(active)
    pending = sort_by_priority(manifest.get_pending_runs())

    for run in pending:
        # Skip if already tracked as active
        if run.run_id in active:
            continue

        if pool.acquire(run.slot_type, run.run_id):
            cmd = [
                sys.executable, "scripts/run_one.py",
                "--run-id", run.run_id,
                "--manifest", str(manifest_path),
                "--mode", mode,
            ]
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            active[run.run_id] = proc
            manifest.update_run(run.run_id, status="running")
            manifest.save(manifest_path)

    return active


def render_status(
    manifest: Manifest,
    pool: SlotPool,
    active: dict[str, subprocess.Popen],
) -> None:
    """Print a compact inline status summary."""
    total = len(manifest.runs)
    done = sum(1 for r in manifest.runs.values() if r.status == "done")
    failed = sum(1 for r in manifest.runs.values() if r.status == "failed")
    pending = sum(1 for r in manifest.runs.values() if r.status == "pending")
    running = len(active)

    print(
        f"\r  [{done}/{total} done, {running} running, "
        f"{pending} pending, {failed} failed]",
        end="", flush=True,
    )


# ---------------------------------------------------------------------------
# Main orchestration loop
# ---------------------------------------------------------------------------

def orchestrate(
    mode: str,
    ollama_slots: int = 1,
    api_slots: int = 2,
    dry_run: bool = False,
    results_root: str = "results",
    ops_log_path: Path | str | None = None,
) -> None:
    """Main orchestration loop."""
    manifest_path = Path(results_root) / mode / "manifest.json"
    if ops_log_path is None:
        ops_log_path = Path("OPERATIONS_LOG.md")
    else:
        ops_log_path = Path(ops_log_path)
    profile = load_profile(mode)

    manifest = create_or_load_manifest(manifest_path, mode, profile)

    if dry_run:
        print_dry_run(manifest)
        return

    pool = SlotPool(ollama_max=ollama_slots, api_max=api_slots)
    active_processes: dict[str, subprocess.Popen] = {}

    # -- Ops log: startup ----------------------------------------------------
    slot_config = {"api_slots": api_slots, "ollama_slots": ollama_slots}
    log_startup(ops_log_path, get_manifest_state(manifest), slot_config)

    # Track milestone thresholds (every 25 completions)
    last_milestone = (
        sum(1 for r in manifest.runs.values() if r.status == "done") // 25
    ) * 25

    # Graceful shutdown: stop launching new runs on SIGINT/SIGTERM
    shutting_down = False
    shutdown_reason = "clean exit"

    def _shutdown_handler(signum, frame):
        nonlocal shutting_down, shutdown_reason
        if shutting_down:
            # Second signal: hard exit
            print("\nForce quit.")
            sys.exit(1)
        shutting_down = True
        shutdown_reason = f"signal {signum}"
        print("\nShutting down gracefully (waiting for active runs to finish)...")

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    print(f"Orchestrator started: mode={mode}, "
          f"ollama_slots={ollama_slots}, api_slots={api_slots}")
    print(f"Total runs: {len(manifest.runs)}")

    while True:
        # 1. Check completed processes
        for run_id, proc in list(active_processes.items()):
            retcode = proc.poll()
            if retcode is not None:
                # Process finished; reload manifest (executor updated it)
                manifest = Manifest.load(manifest_path)
                entry = manifest.runs[run_id]
                pool.release(entry.slot_type, run_id)
                del active_processes[run_id]
                if entry.status == "done":
                    print(f"\n  [+] {run_id}: done")
                    # -- Ops log: milestone check ----------------------------
                    done_count = sum(
                        1 for r in manifest.runs.values() if r.status == "done"
                    )
                    if done_count >= last_milestone + 25:
                        last_milestone = (done_count // 25) * 25
                        log_milestone(
                            ops_log_path, get_manifest_state(manifest),
                        )
                else:
                    print(f"\n  [!] {run_id}: {entry.status} - {entry.error}")
                    # -- Ops log: API limit detection ------------------------
                    if entry.error and "usage limit" in entry.error.lower():
                        log_api_limit(ops_log_path, run_id, entry.error)

        # 2. Launch new runs if slots available (unless shutting down)
        if not shutting_down:
            active_processes = launch_pending_runs(
                manifest, pool, active_processes, manifest_path, mode,
            )

        # 3. Check if all done
        if not active_processes and not manifest.get_pending_runs():
            break
        if shutting_down and not active_processes:
            break

        # 4. Display progress
        render_status(manifest, pool, active_processes)
        time.sleep(5)

    # Final summary
    done = sum(1 for r in manifest.runs.values() if r.status == "done")
    failed = sum(1 for r in manifest.runs.values() if r.status == "failed")
    pending = sum(1 for r in manifest.runs.values() if r.status == "pending")
    print(f"\n\nOrchestrator finished: {done} done, {failed} failed, "
          f"{pending} pending out of {len(manifest.runs)} total.")

    # -- Ops log: shutdown ---------------------------------------------------
    log_shutdown(ops_log_path, get_manifest_state(manifest), shutdown_reason)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Central experiment orchestrator"
    )
    parser.add_argument(
        "--mode", required=True,
        help="Experiment mode (integration_smoke, full, etc.)",
    )
    parser.add_argument(
        "--ollama-slots", type=int, default=1,
        help="Max concurrent ollama runs (default: 1)",
    )
    parser.add_argument(
        "--api-slots", type=int, default=2,
        help="Max concurrent API runs (default: 2)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show run matrix without executing",
    )
    parser.add_argument(
        "--results-root", default="results",
        help="Results root directory (default: results)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    orchestrate(
        mode=args.mode,
        ollama_slots=args.ollama_slots,
        api_slots=args.api_slots,
        dry_run=args.dry_run,
        results_root=args.results_root,
    )


if __name__ == "__main__":
    main()
