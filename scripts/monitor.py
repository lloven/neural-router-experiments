"""Manifest-based experiment monitor.

Reads ONLY from manifest.json and progress sidecar files.
No log parsing, no file existence checks, no pgrep.

Usage::

    python -m scripts.monitor --mode full --refresh 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path for direct invocation (python scripts/monitor.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.manifest import Manifest


# ---------------------------------------------------------------------------
# Progress sidecar reader
# ---------------------------------------------------------------------------

def read_progress_sidecars(mode: str, results_root: str = "results") -> dict[str, dict]:
    """Read all progress sidecar files for live stage detail.

    Returns:
        Dict mapping run_id (file stem) -> sidecar data dict.
    """
    progress_dir = Path(results_root) / mode / ".progress"
    sidecars: dict[str, dict] = {}
    if not progress_dir.exists():
        return sidecars
    for p in progress_dir.glob("*.json"):
        try:
            data = json.loads(p.read_text())
            sidecars[p.stem] = data
        except (json.JSONDecodeError, OSError):
            pass
    return sidecars


# ---------------------------------------------------------------------------
# Dashboard renderer
# ---------------------------------------------------------------------------

def _format_eta(eta_seconds: float) -> str:
    """Format ETA in seconds to a human-readable string."""
    if eta_seconds < 0:
        return "?"
    if eta_seconds < 60:
        return f"{int(eta_seconds)}s"
    minutes = int(eta_seconds / 60)
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    remaining_min = minutes % 60
    if remaining_min == 0:
        return f"{hours}h"
    return f"{hours}h{remaining_min}m"


def _format_progress_bar(fraction: float, width: int = 10) -> str:
    """Render a fixed-width ASCII progress bar."""
    filled = int(fraction * width)
    filled = max(0, min(filled, width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _time_ago(iso_timestamp: str | None) -> str:
    """Format an ISO timestamp as 'Xm ago'."""
    if not iso_timestamp:
        return ""
    try:
        dt = datetime.fromisoformat(iso_timestamp)
        delta = datetime.now() - dt
        minutes = int(delta.total_seconds() / 60)
        if minutes < 1:
            return "just now"
        if minutes < 60:
            return f"{minutes}m ago"
        hours = minutes // 60
        return f"{hours}h ago"
    except (ValueError, TypeError):
        return ""


def render_dashboard(manifest: Manifest, sidecars: dict[str, dict]) -> str:
    """Render the dashboard as a string.

    Args:
        manifest: Loaded experiment manifest.
        sidecars: Dict of run_id -> progress sidecar data.

    Returns:
        Multi-line string suitable for terminal display.
    """
    lines: list[str] = []
    sep = "=" * 78

    lines.append(sep)
    lines.append("  Neural Router Experiment Pipeline")
    lines.append(sep)
    lines.append("")

    # -- Overall progress ---------------------------------------------------
    total = len(manifest.runs)
    done = sum(1 for r in manifest.runs.values() if r.status == "done")
    running = sum(1 for r in manifest.runs.values() if r.status == "running")
    failed = sum(1 for r in manifest.runs.values() if r.status == "failed")
    pending = total - done - running - failed

    lines.append(
        f"  Progress: {done}/{total} done, {running} running, "
        f"{pending} pending, {failed} failed"
    )
    lines.append("")

    # -- Running ------------------------------------------------------------
    running_entries = [r for r in manifest.runs.values() if r.status == "running"]
    if running_entries:
        lines.append("  Running:")
        for entry in running_entries:
            sc = sidecars.get(entry.run_id, {})
            items_done = sc.get("items_done", 0)
            items_total = sc.get("items_total", 0)
            eta_s = sc.get("eta_s", -1)

            frac = items_done / items_total if items_total > 0 else 0
            bar = _format_progress_bar(frac)
            pct = f"{int(frac * 100):3d}%"

            # Use short label: stage/dataset/config/model (drop seeds for compactness)
            label = f"{entry.stage}/{entry.dataset}/{entry.config}/s{entry.seed}/{entry.model_key}"
            detail = f"{bar} {pct}"
            if items_total > 0:
                detail += f"  {items_done:>5d}/{items_total} events"
            eta_str = _format_eta(eta_s)
            if eta_str != "?":
                detail += f"  ETA: {eta_str}"

            lines.append(f"    {label:<45s} {detail}")
        lines.append("")

    # -- Recent completions -------------------------------------------------
    done_entries = [r for r in manifest.runs.values() if r.status == "done"]
    if done_entries:
        # Sort by finished_at descending, show most recent first
        done_entries.sort(
            key=lambda r: r.finished_at or "", reverse=True
        )
        lines.append("  Recent completions:")
        for entry in done_entries[:5]:
            label = f"{entry.stage}/{entry.dataset}/{entry.config}/s{entry.seed}/{entry.model_key}"
            f1_str = f"F1={entry.metrics['f1']:.3f}" if entry.metrics and "f1" in entry.metrics else ""
            ago = _time_ago(entry.finished_at)
            lines.append(f"    {label:<45s} {f1_str:>10s}  {ago}")
        lines.append("")

    # -- Failed -------------------------------------------------------------
    failed_entries = [r for r in manifest.runs.values() if r.status == "failed"]
    if failed_entries:
        lines.append(f"  Failed ({len(failed_entries)}):")
        for entry in failed_entries[:10]:
            label = f"{entry.stage}/{entry.dataset}/{entry.config}/s{entry.seed}/{entry.model_key}"
            err = (entry.error or "unknown")[:50]
            ago = _time_ago(entry.finished_at)
            lines.append(f"    {label:<45s} {err}  {ago}")
        lines.append("")

    # -- Pending summary ----------------------------------------------------
    if pending > 0:
        lines.append(f"  Pending: {pending} runs")
        lines.append("")

    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manifest-based experiment monitor"
    )
    parser.add_argument("--mode", default="full")
    parser.add_argument(
        "--manifest", default=None,
        help="Path to manifest.json (auto-detected from --mode if omitted)",
    )
    parser.add_argument("--refresh", type=float, default=5.0)
    args = parser.parse_args()

    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else Path("results") / args.mode / "manifest.json"
    )

    while True:
        manifest = Manifest.load(manifest_path)
        sidecars = read_progress_sidecars(args.mode)
        output = render_dashboard(manifest, sidecars)
        print("\033[2J\033[H" + output)

        if all(
            r.status in ("done", "failed") for r in manifest.runs.values()
        ):
            print("  Pipeline complete!")
            break

        time.sleep(args.refresh)


if __name__ == "__main__":
    main()
