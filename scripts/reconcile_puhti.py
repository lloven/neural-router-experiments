#!/usr/bin/env python3
"""Reconcile Puhti SLURM-array results back into the local manifest.

Background
==========

The Neural Router campaign is split across two infrastructures:

  * Local manifest (`results/full/manifest.json`) — written by `run_all.py`
    on the laptop / FCG vGPU. Tracks every (stage, dataset, config, seed,
    model) run.

  * Puhti SLURM array (`scripts/slurm/puhti_qwen7b_ablation.sh`) — runs the
    Qwen-7B ablation on V100s, writing results to per-task directories on
    /scratch:

        /scratch/project_2018951/neural-router/code/results/full/
            ablation/by_task/qwen7b_<DATASET>_<CONFIG>_s<SEED>/
                <prefix>_results.csv

Without reconciliation, the local manifest still shows a Puhti-completed run
as "pending" and would re-launch it on the next `--resume`.

This tool walks a local mirror of the Puhti `by_task/` tree (rsynced first
via `rsync_puhti_results()`), validates each CSV against L30 (data row
present, F1 non-null, config/dataset/seed match the directory name), and
marks the matching local-manifest entry as `done` with metrics.

Usage
=====

    # 1. Pull Puhti CSVs locally (rsync, ~MB-scale).
    python scripts/reconcile_puhti.py --pull

    # 2. Dry-run: show what would change without mutating the manifest.
    python scripts/reconcile_puhti.py --dry-run

    # 3. Apply.
    python scripts/reconcile_puhti.py --apply

L30 enforced: a CSV without a data row, or with null F1, or with mismatched
config/dataset/seed in the row, is treated as a silent failure and left
pending.

L37: this is a thin reconciliation wrapper, not a duplication of the
orchestrator. The local manifest stays authoritative for "what ran where";
this tool only updates `done/metrics` based on Puhti CSV evidence.
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# Allow running as `python scripts/reconcile_puhti.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.manifest import Manifest

logger = logging.getLogger(__name__)


# --- Defaults -------------------------------------------------------------

DEFAULT_SSH_HOST = "puhti"
DEFAULT_REMOTE_BY_TASK = (
    "/scratch/project_2018951/neural-router/code/results/full/"
    "ablation/by_task/"
)
DEFAULT_LOCAL_MIRROR = Path("results/puhti_mirror/by_task")
DEFAULT_MANIFEST = Path("results/full/manifest.json")

# Format produced by puhti_qwen7b_ablation.sh: TAG="qwen7b_${DATASET}_${CONFIG}_s${SEED}".
# Only this prefix is recognised because only "qwen7b" maps to a local
# manifest model_key (params.yaml). Tier 1c results from other models
# (llama-8b, qwen2.5-32b, qwen2.5-1.5b, phi3-mini, …) live in
# results/full/ablation/by_task/<prefix>_*/  and are analysed separately —
# they are deliberately NOT folded into the local manifest, because the
# manifest is the paper's primary-experiment record. Reconciliation
# treats unrecognised prefixes as `unmatched_dirs` and leaves them alone.
_TASK_DIR_RE = re.compile(
    r"^(?P<model>qwen7b)"
    r"_(?P<dataset>D[123])"
    r"_(?P<config>A[0-6])"
    r"_s(?P<seed>\d+)$"
)


# ---------------------------------------------------------------------------
# Path → run_id mapping
# ---------------------------------------------------------------------------


def task_dir_to_run_id(name: str) -> str | None:
    """Map a Puhti by_task directory name to its local manifest run_id.

    Returns None if the directory name doesn't match the SLURM TAG format.
    """
    m = _TASK_DIR_RE.match(name)
    if m is None:
        return None
    return (
        f"ablation__{m['dataset']}__{m['config']}"
        f"__seed{int(m['seed'])}__{m['model']}"
    )


# ---------------------------------------------------------------------------
# CSV validation (L30: content, not just existence)
# ---------------------------------------------------------------------------


def validate_csv(
    path: Path,
    expected_config: str,
    expected_dataset: str,
    expected_seed: int,
) -> dict[str, Any] | None:
    """Validate a Puhti results CSV and return the metric row if usable.

    Per L30, a structurally valid CSV is not enough — the data row must be
    present, F1 must be non-null, and the row's config/dataset/seed must
    match the values implied by the directory name.

    Returns:
        Dict with ``f1, precision, recall, fpr, invocations,
        compression_ratio, latency_s`` if the CSV is valid, else None.
    """
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # malformed CSV
        logger.warning("Could not parse %s: %s", path, exc)
        return None
    if df.empty:
        return None

    # The CSV produced by run_experiment.flush_result holds one row per
    # (config, seed) pair. Per-task dirs only ever contain one such pair,
    # but be defensive: pick the row that matches the expected key.
    matches = df[
        (df["config"] == expected_config)
        & (df["dataset"] == expected_dataset)
        & (df["seed"].astype(int) == int(expected_seed))
    ]
    if matches.empty:
        return None

    row = matches.iloc[0]
    f1 = row.get("f1")
    if f1 is None or (isinstance(f1, float) and pd.isna(f1)):
        return None

    out: dict[str, Any] = {}
    for k in (
        "f1", "precision", "recall", "fpr",
        "invocations", "compression_ratio", "latency_s",
        "tokens_prompt", "tokens_response", "cost_per_1k",
    ):
        if k not in df.columns:
            continue
        v = row.get(k)
        if pd.isna(v):
            out[k] = None
        else:
            # Coerce numpy scalars to Python native so json.dump can serialise.
            out[k] = v.item() if hasattr(v, "item") else v
    return out


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


def reconcile_local_dir(
    manifest: Manifest,
    by_task_root: Path,
    dry_run: bool = False,
) -> dict[str, int]:
    """Walk a local mirror of Puhti's by_task/ tree and update manifest.

    For each subdirectory:
      * Map its name to a run_id; skip if the format is unfamiliar.
      * If the matching manifest entry is already ``done``, count it but
        don't touch.
      * Else, look for a ``*_results.csv`` inside; validate it; if valid,
        mark the entry done with metrics (unless ``dry_run``).

    Returns:
        Counts: ``marked_done``, ``already_done``, ``unmatched_dirs``,
        ``invalid_csvs``.
    """
    summary = {
        "marked_done": 0,
        "already_done": 0,
        "unmatched_dirs": 0,
        "invalid_csvs": 0,
    }

    if not by_task_root.exists():
        logger.warning("by_task root %s does not exist", by_task_root)
        return summary

    for task_dir in sorted(p for p in by_task_root.iterdir() if p.is_dir()):
        run_id = task_dir_to_run_id(task_dir.name)
        # Recover key parts directly from the regex (avoids re-parsing
        # the run_id that we just constructed).
        m = _TASK_DIR_RE.match(task_dir.name)

        if run_id is None or m is None or run_id not in manifest.runs:
            summary["unmatched_dirs"] += 1
            continue

        entry = manifest.runs[run_id]
        if entry.status == "done":
            summary["already_done"] += 1
            continue

        csvs = sorted(task_dir.glob("*_results.csv"))
        if not csvs:
            summary["invalid_csvs"] += 1
            continue

        metrics = validate_csv(
            csvs[0],
            expected_config=m["config"],
            expected_dataset=m["dataset"],
            expected_seed=int(m["seed"]),
        )
        if metrics is None:
            summary["invalid_csvs"] += 1
            continue

        if dry_run:
            summary["marked_done"] += 1
            continue

        manifest.update_run(
            run_id,
            status="done",
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=None,
            metrics=metrics,
        )
        summary["marked_done"] += 1

    return summary


# ---------------------------------------------------------------------------
# rsync from Puhti
# ---------------------------------------------------------------------------


def rsync_puhti_results(
    ssh_host: str,
    remote_by_task: str,
    local_mirror: Path,
) -> None:
    """Mirror Puhti's by_task/ tree locally via rsync. Read-only on remote."""
    local_mirror.parent.mkdir(parents=True, exist_ok=True)
    remote = f"{ssh_host}:{remote_by_task.rstrip('/')}/"
    cmd = [
        "rsync", "-azv",
        # Only result CSVs and the directories that hold them; skip log
        # files, ollama caches, and intermediate checkpoints.
        "--include=*/", "--include=*_results.csv", "--exclude=*",
        remote, str(local_mirror) + "/",
    ]
    logger.info("rsync %s → %s", remote, local_mirror)
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_MANIFEST,
        help="Path to local manifest.json",
    )
    parser.add_argument(
        "--ssh-host", default=DEFAULT_SSH_HOST,
        help="SSH alias for Puhti (default: puhti)",
    )
    parser.add_argument(
        "--remote-by-task", default=DEFAULT_REMOTE_BY_TASK,
        help="Path to by_task/ on Puhti scratch",
    )
    parser.add_argument(
        "--local-mirror", type=Path, default=DEFAULT_LOCAL_MIRROR,
        help="Where to mirror Puhti CSVs locally",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pull", action="store_true",
                      help="Only rsync Puhti CSVs locally")
    mode.add_argument("--dry-run", action="store_true",
                      help="Reconcile without saving manifest")
    mode.add_argument("--apply", action="store_true",
                      help="Reconcile and save manifest")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.pull:
        rsync_puhti_results(args.ssh_host, args.remote_by_task, args.local_mirror)
        return 0

    if not args.manifest.exists():
        logger.error("Manifest not found: %s", args.manifest)
        return 1

    manifest = Manifest.load(args.manifest)
    summary = reconcile_local_dir(
        manifest, args.local_mirror, dry_run=args.dry_run,
    )

    print(
        f"Reconciliation summary "
        f"(mode={'dry-run' if args.dry_run else 'apply'}): "
        f"marked_done={summary['marked_done']}, "
        f"already_done={summary['already_done']}, "
        f"unmatched_dirs={summary['unmatched_dirs']}, "
        f"invalid_csvs={summary['invalid_csvs']}"
    )

    if args.apply and summary["marked_done"] > 0:
        manifest.save(args.manifest)
        logger.info("Saved manifest with %d new done entries", summary["marked_done"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
