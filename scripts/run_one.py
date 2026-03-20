#!/usr/bin/env python3
"""Single-run executor for the Neural Router experiment pipeline.

Executes exactly ONE run from the manifest and exits.  Designed to be
launched as a subprocess by the orchestrator.

Usage::

    python scripts/run_one.py \
        --run-id "ablation__D1__A0__seed42__qwen7b" \
        --manifest results/full/manifest.json \
        --mode full
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Load .env before any LLM imports (API keys)
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_experiment import run_ablation_stage
from scripts.run_sensitivity import run_sensitivity_stage
from scripts.run_scaling import run_scaling_stage
from src.config import load_profile
from src.embeddings import EmbeddingModel
from src.llm import LLMClient
from src.manifest import Manifest
from src.progress import SidecarProgressCallback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def execute_run(
    run_id: str,
    manifest_path: Path | str,
    profile: dict,
) -> bool:
    """Execute a single run identified by *run_id* in the manifest.

    Dispatches to the appropriate stage wrapper based on ``entry.stage``,
    then updates the manifest with the outcome (done/failed).

    Args:
        run_id: The unique identifier of the run to execute.
        manifest_path: Path to the manifest JSON file.
        profile: Loaded mode profile dict (from ``load_profile``).

    Returns:
        True on success, False on failure.
    """
    manifest_path = Path(manifest_path)

    # Load manifest and locate the run entry
    manifest = Manifest.load(manifest_path)
    if run_id not in manifest.runs:
        raise KeyError(f"Run '{run_id}' not found in manifest {manifest_path}")
    entry = manifest.runs[run_id]

    # Mark as started
    now_iso = datetime.now(timezone.utc).isoformat()
    manifest.update_run(run_id, status="running", started_at=now_iso)
    manifest.save(manifest_path)

    # Create sidecar progress callback for the monitor
    progress = SidecarProgressCallback(
        run_id=run_id,
        mode=manifest.mode,
    )

    try:
        # Initialize LLM client and embedding model
        llm_client = None
        embedding_model = None
        if not entry.config.startswith("baseline_"):
            llm_client = LLMClient(model=entry.model_id)
            embed_name = profile.get("defaults", {}).get(
                "embedding_model", "all-MiniLM-L6-v2"
            )
            embedding_model = EmbeddingModel(model_name=embed_name)

        # Dispatch to the correct stage wrapper
        if entry.stage == "ablation":
            results = run_ablation_stage(
                entry=entry, profile=profile,
                llm_client=llm_client, embedding_model=embedding_model,
                progress_tracker=progress,
            )
        elif entry.stage == "sensitivity":
            run_sensitivity_stage(
                entry=entry, profile=profile,
                llm_client=llm_client, embedding_model=embedding_model,
                progress_tracker=progress,
            )
            results = None
        elif entry.stage == "scaling":
            run_scaling_stage(
                entry=entry, profile=profile,
                llm_client=llm_client, embedding_model=embedding_model,
                progress_tracker=progress,
            )
            results = None
        else:
            raise ValueError(f"Unknown stage: {entry.stage}")

        # Extract metrics from ablation results
        metrics = None
        if results and len(results) > 0:
            r = results[0]
            metrics = {
                "f1": r.f1.mean,
                "precision": r.precision.mean,
                "recall": r.recall.mean,
                "fpr": r.fpr.mean,
                "invocations": r.invocations,
                "compression_ratio": r.compression_ratio,
                "latency_s": r.latency_s,
            }

        # Mark progress sidecar as done
        progress.finish()

        # Mark as done
        finished_iso = datetime.now(timezone.utc).isoformat()
        manifest = Manifest.load(manifest_path)  # re-read for freshness
        manifest.update_run(
            run_id,
            status="done",
            finished_at=finished_iso,
            error=None,
            metrics=metrics,
        )
        manifest.save(manifest_path)
        logger.info(f"Run {run_id} completed successfully")
        return True

    except Exception as exc:
        logger.error(f"Run {run_id} failed: {exc}")
        # Mark as failed
        try:
            manifest = Manifest.load(manifest_path)
            manifest.update_run(
                run_id,
                status="failed",
                error=str(exc),
                metrics=None,
            )
            manifest.save(manifest_path)
        except Exception as save_exc:
            logger.error(f"Could not save failure status: {save_exc}")
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute a single run from the experiment manifest"
    )
    parser.add_argument(
        "--run-id", required=True,
        help="Run ID to execute (must exist in manifest)"
    )
    parser.add_argument(
        "--manifest", required=True,
        help="Path to manifest.json"
    )
    parser.add_argument(
        "--mode", required=True,
        help="Experiment mode (integration_smoke, full, etc.)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = load_profile(mode=args.mode)
    success = execute_run(args.run_id, args.manifest, profile)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
