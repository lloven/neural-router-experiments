"""Auto-logging to OPERATIONS_LOG.md.

Provides functions to append structured entries to the ops log during
orchestrator startup, milestones, API limit hits, and shutdown.

The ops log has two sections that receive entries:
- **Status Summary** table: compact rows with date + manifest counts
- **Operations** section: timestamped narrative entries
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

_OPS_LOG_TEMPLATE = """\
# Neural Router Experiment Operations Log

Track all manual interventions: starts, stops, restarts, cleanups, budget changes, configuration changes, and notable events.

## Status Summary

| Date | Done | Running | Pending | Failed | Total | Notes |
|---|---|---|---|---|---|---|

## Operations

"""


def _now() -> str:
    """Return a human-readable timestamp."""
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _ensure_ops_log(path: Path) -> str:
    """Read the ops log, creating it from template if it doesn't exist."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_OPS_LOG_TEMPLATE)
    return path.read_text()


def _append_status_row(path: Path, state: dict, notes: str) -> None:
    """Append a row to the Status Summary table.

    Inserts immediately after the last ``|`` line in the table (before
    the blank line that separates it from ## Operations).
    """
    path = Path(path)
    content = _ensure_ops_log(path)
    ts = _now()
    row = (
        f"| {ts} | {state['done']} | {state.get('running', 0)} "
        f"| {state['pending']} | {state['failed']} "
        f"| {state['total']} | {notes} |"
    )

    # Find the end of the status summary table (last line starting with |)
    lines = content.split("\n")
    insert_idx = None
    in_table = False
    for i, line in enumerate(lines):
        if line.startswith("| Date"):
            in_table = True
        elif in_table and line.startswith("|"):
            insert_idx = i  # keep updating to find last table row
        elif in_table and not line.startswith("|"):
            # End of table; insert after last row
            if insert_idx is None:
                # Table has only header + separator, insert here
                insert_idx = i - 1
            break

    if insert_idx is not None:
        lines.insert(insert_idx + 1, row)
    else:
        # Fallback: insert after the separator line
        for i, line in enumerate(lines):
            if line.startswith("|---"):
                lines.insert(i + 1, row)
                break

    path.write_text("\n".join(lines))


def _append_operations_entry(path: Path, title: str, body: str) -> None:
    """Append a timestamped entry to the Operations section."""
    path = Path(path)
    content = _ensure_ops_log(path)
    ts = _now()

    entry = f"\n### {ts} -- {title}\n\n{body}\n"

    # Find "## Operations" and append after it (before next ## or at end)
    lines = content.split("\n")
    ops_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "## Operations":
            ops_idx = i
            break

    if ops_idx is not None:
        # Find next ## section after Operations, or end of file
        insert_idx = len(lines)
        for i in range(ops_idx + 1, len(lines)):
            if lines[i].startswith("## ") and lines[i].strip() != "## Operations":
                insert_idx = i
                break

        # Insert the entry just before the next section (or at end)
        entry_lines = entry.split("\n")
        for j, el in enumerate(entry_lines):
            lines.insert(insert_idx + j, el)
    else:
        # No Operations section found; append at end
        lines.append(entry)

    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_manifest_state(manifest) -> dict:
    """Extract a state summary dict from a Manifest object."""
    runs = manifest.runs.values()
    done = sum(1 for r in runs if r.status == "done")
    pending = sum(1 for r in runs if r.status == "pending")
    failed = sum(1 for r in runs if r.status == "failed")
    running = sum(1 for r in runs if r.status == "running")
    total = len(manifest.runs)
    return {
        "done": done,
        "pending": pending,
        "failed": failed,
        "running": running,
        "total": total,
    }


def log_startup(
    ops_log_path: Path,
    manifest_state: dict,
    slot_config: dict,
) -> None:
    """Log orchestrator startup to the ops log."""
    _ensure_ops_log(ops_log_path)

    slot_str = ", ".join(f"{k}={v}" for k, v in sorted(slot_config.items()))
    notes = f"Startup ({slot_str})"

    _append_status_row(ops_log_path, manifest_state, notes)
    _append_operations_entry(
        ops_log_path,
        "Startup",
        f"- **Action:** Orchestrator started\n"
        f"- **Slots:** {slot_str}\n"
        f"- **State:** {manifest_state['done']} done, "
        f"{manifest_state['pending']} pending, "
        f"{manifest_state['failed']} failed, "
        f"{manifest_state.get('running', 0)} running "
        f"(total {manifest_state['total']})",
    )


def log_milestone(ops_log_path: Path, manifest_state: dict) -> None:
    """Log a milestone (e.g., every 25 completed runs)."""
    done = manifest_state["done"]
    notes = f"Milestone: {done} done"

    _append_status_row(ops_log_path, manifest_state, notes)


def log_api_limit(
    ops_log_path: Path,
    run_id: str,
    error_msg: str,
) -> None:
    """Log an API limit hit."""
    _ensure_ops_log(ops_log_path)

    _append_operations_entry(
        ops_log_path,
        "API limit hit",
        f"- **Run:** `{run_id}`\n"
        f"- **Error:** {error_msg}",
    )


def log_shutdown(
    ops_log_path: Path,
    manifest_state: dict,
    reason: str,
) -> None:
    """Log orchestrator shutdown."""
    _ensure_ops_log(ops_log_path)

    notes = f"Shutdown ({reason})"
    _append_status_row(ops_log_path, manifest_state, notes)
    _append_operations_entry(
        ops_log_path,
        "Shutdown",
        f"- **Action:** Orchestrator stopped\n"
        f"- **Reason:** {reason}\n"
        f"- **Final state:** {manifest_state['done']} done, "
        f"{manifest_state['pending']} pending, "
        f"{manifest_state['failed']} failed "
        f"(total {manifest_state['total']})",
    )
