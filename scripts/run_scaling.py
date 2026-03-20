#!/usr/bin/env python3
"""Scaling analysis for the Neural Router (Section 4.5, Fig. 5).

Measures how accuracy and cost metrics scale with corpus size by
subsampling along two dimensions:

  1. Event scaling   (|M|): 50, 100, 200, 500, 1000, 2000, 5000 events
     with fixed subscription set.
  2. Subscription scaling (|S|): 5, 10, 20, 50, 100, full -- with
     a fixed 500-event sample. Only meaningful for D2/D3 (many subscriptions).

All runs use the A3 (full pipeline) configuration. Outputs CSV files
suitable for plotting scaling curves.

Usage:
    python scripts/run_scaling.py --dataset D1 --dimension events
    python scripts/run_scaling.py --dataset D2 --dimension subscriptions
    python scripts/run_scaling.py --dataset D1 --dimension all
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.data import load_dataset_by_name, Dataset, Event, Subscription
from src.router import NeuralRouter, RouterConfig, ABLATION_CONFIGS
from src.embeddings import EmbeddingModel
from src.evaluation import evaluate_matches
from src.llm import LLMClient, DryRunLLMClient
from src.config import load_profile, output_dir_for_mode, get_model_overrides
from src.manifest import RunEntry
from src.progress import ProgressTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

EVENT_COUNTS_DEFAULT = [50, 100, 200, 500, 1000, 2000, 5000]


def _append_row(row: dict, path: Path):
    """Append a single result row to a CSV, creating it with a header if needed."""
    write_header = not path.exists() or path.stat().st_size == 0
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", header=write_header, index=False)


def _load_completed_values(path: Path, col: str) -> set[str]:
    """Load already-completed values from a partial CSV for resume."""
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path)
        return set(df[col].astype(str).tolist())
    except Exception:
        return set()


def scale_events(dataset, embedder, llm_client, seed, output_dir, batch_size=200, llm_timeout=600, event_counts=EVENT_COUNTS_DEFAULT, progress_callback=None):
    """Scale the number of events |M| while keeping subscriptions fixed.

    Subsamples events deterministically at each size and runs A3.
    Writes results incrementally (one row per scale point) for crash recovery.

    Args:
        dataset: Full dataset (events are subsampled from this).
        embedder: Embedding model instance.
        llm_client: LLM client (real or dry-run).
        seed: Random seed for subsampling and k-means.
        output_dir: Directory for CSV output.
        batch_size: Hard cap on events per LLM call (max_batch_events).
        llm_timeout: Per-request timeout in seconds.
        progress_callback: Optional callback(idx, total) for progress tracking.

    Returns:
        DataFrame with one row per event count.
    """
    path = output_dir / f"scaling_events_{dataset.short_name}.csv"
    done = _load_completed_values(path, "n_events")
    rows = []
    rng = np.random.RandomState(seed)

    for i, n_events in enumerate(event_counts):
        if n_events > dataset.num_events:
            continue
        if str(n_events) in done:
            logger.info(f"Events scaling: |M|={n_events} — already done, skipping")
            continue

        # Subsample events
        indices = rng.choice(dataset.num_events, n_events, replace=False)
        sub_events = [dataset.events[i] for i in sorted(indices)]
        sub_dataset = Dataset(
            name=dataset.name,
            short_name=dataset.short_name,
            events=sub_events,
            subscriptions=dataset.subscriptions,
            metadata=dataset.metadata,
        )

        logger.info(f"Events scaling: |M|={n_events}")
        llm_client.reset_stats() if hasattr(llm_client, 'reset_stats') else None

        model_overrides = {}
        if "ollama" in llm_client.model.lower():
            model_overrides["max_parallel"] = 1
            model_overrides["use_async"] = False
        if "claude-3-haiku" in llm_client.model.lower():
            model_overrides["llm_max_tokens"] = 4096
            model_overrides["max_parallel"] = 2

        config = RouterConfig(**{
            **ABLATION_CONFIGS["A3"].__dict__,
            "seed": seed,
            "k": min(19, dataset.num_subscriptions),
            "kappa": 3,
            "llm_model": llm_client.model,
            "max_batch_events": batch_size,
            "llm_timeout": llm_timeout,
            **model_overrides,
        })

        router = NeuralRouter(config=config, llm_client=llm_client, embedding_model=embedder)

        t0 = time.time()
        router.optimize_subscriptions(sub_dataset.subscriptions)
        matches = router.match_events(sub_dataset.events)
        wall_time = time.time() - t0

        result = evaluate_matches(
            matches=matches,
            dataset=sub_dataset,
            config_name=f"scale_M{n_events}",
            seed=seed,
            router_stats=router.stats,
        )

        row = {
            "n_events": n_events,
            "n_subscriptions": dataset.num_subscriptions,
            "f1": result.f1.mean,
            "precision": result.precision.mean,
            "recall": result.recall.mean,
            "invocations": result.invocations,
            "latency_s": wall_time,
            "tokens_prompt": result.tokens_prompt,
            "tokens_response": result.tokens_response,
            "cost_per_1k": result.cost_per_1k,
        }
        rows.append(row)
        _append_row(row, path)
        logger.info(f"  |M|={n_events}: F1={result.f1.mean:.4f}, {wall_time:.0f}s")
        if progress_callback:
            progress_callback(idx=i, total=len(event_counts))

    logger.info(f"Saved to {path} ({len(rows)} new rows)")
    return pd.DataFrame(rows) if rows else pd.read_csv(path)


def scale_subscriptions(dataset, embedder, llm_client, seed, output_dir, batch_size=200, llm_timeout=600, progress_callback=None):
    """Scale the number of subscriptions |S| while keeping events fixed.

    Uses a fixed 500-event sample and subsamples subscriptions at each size.
    Prioritises subscriptions that have at least one matching event to avoid
    degenerate evaluation scenarios.

    Only meaningful for datasets with many subscriptions (D2, D3); skips
    datasets with fewer than 20 subscriptions.

    Args:
        dataset: Full dataset (subscriptions are subsampled from this).
        embedder: Embedding model instance.
        llm_client: LLM client (real or dry-run).
        seed: Random seed for subsampling and k-means.
        output_dir: Directory for CSV output.
        batch_size: Hard cap on events per LLM call (max_batch_events).
        llm_timeout: Per-request timeout in seconds.

    Returns:
        DataFrame with one row per subscription count.
    """
    if dataset.num_subscriptions < 20:
        logger.info(f"Skipping subscription scaling for {dataset.short_name} (only {dataset.num_subscriptions} subs)")
        return pd.DataFrame()

    path = output_dir / f"scaling_subs_{dataset.short_name}.csv"
    done = _load_completed_values(path, "n_subscriptions")

    sub_counts = [5, 10, 20, 50, 100]
    sub_counts = [n for n in sub_counts if n <= dataset.num_subscriptions]
    sub_counts.append(dataset.num_subscriptions)  # always include full set

    rows = []
    rng = np.random.RandomState(seed)

    # Use a fixed subset of events
    n_events = min(500, dataset.num_events)
    event_indices = rng.choice(dataset.num_events, n_events, replace=False)
    sub_events = [dataset.events[i] for i in sorted(event_indices)]

    for i_sub, n_subs in enumerate(sub_counts):
        if str(n_subs) in done:
            logger.info(f"Subscription scaling: |S|={n_subs} — already done, skipping")
            continue
        # Subsample subscriptions (keeping those that have events)
        sub_ids_with_events = set()
        for e in sub_events:
            sub_ids_with_events.update(e.ground_truth)

        # Prefer subscriptions that have events
        subs_with_events = [s for s in dataset.subscriptions if s.id in sub_ids_with_events]
        subs_without = [s for s in dataset.subscriptions if s.id not in sub_ids_with_events]

        if n_subs <= len(subs_with_events):
            chosen_indices = rng.choice(len(subs_with_events), n_subs, replace=False)
            chosen_subs = [subs_with_events[i] for i in sorted(chosen_indices)]
        else:
            chosen_subs = list(subs_with_events)
            remaining = n_subs - len(chosen_subs)
            if remaining > 0 and subs_without:
                extra_indices = rng.choice(len(subs_without), min(remaining, len(subs_without)), replace=False)
                chosen_subs.extend([subs_without[i] for i in sorted(extra_indices)])

        sub_dataset = Dataset(
            name=dataset.name,
            short_name=dataset.short_name,
            events=sub_events,
            subscriptions=chosen_subs,
            metadata=dataset.metadata,
        )

        logger.info(f"Subscription scaling: |S|={n_subs} (actual={len(chosen_subs)})")
        llm_client.reset_stats() if hasattr(llm_client, 'reset_stats') else None

        k = min(max(1, n_subs // 2), 30)
        model_overrides = {}
        if "ollama" in llm_client.model.lower():
            model_overrides["max_parallel"] = 1
            model_overrides["use_async"] = False
        if "claude-3-haiku" in llm_client.model.lower():
            model_overrides["llm_max_tokens"] = 4096
            model_overrides["max_parallel"] = 2

        config = RouterConfig(**{
            **ABLATION_CONFIGS["A3"].__dict__,
            "seed": seed,
            "k": k,
            "kappa": 3,
            "llm_model": llm_client.model,
            "max_batch_events": batch_size,
            "llm_timeout": llm_timeout,
            **model_overrides,
        })

        router = NeuralRouter(config=config, llm_client=llm_client, embedding_model=embedder)

        t0 = time.time()
        router.optimize_subscriptions(sub_dataset.subscriptions)
        matches = router.match_events(sub_dataset.events)
        wall_time = time.time() - t0

        result = evaluate_matches(
            matches=matches,
            dataset=sub_dataset,
            config_name=f"scale_S{n_subs}",
            seed=seed,
            router_stats=router.stats,
        )

        row = {
            "n_events": n_events,
            "n_subscriptions": len(chosen_subs),
            "k": k,
            "f1": result.f1.mean,
            "precision": result.precision.mean,
            "recall": result.recall.mean,
            "invocations": result.invocations,
            "latency_s": wall_time,
            "tokens_prompt": result.tokens_prompt,
            "tokens_response": result.tokens_response,
            "cost_per_1k": result.cost_per_1k,
        }
        rows.append(row)
        _append_row(row, path)
        logger.info(f"  |S|={len(chosen_subs)}: F1={result.f1.mean:.4f}, {wall_time:.0f}s")
        if progress_callback:
            progress_callback(idx=i_sub, total=len(sub_counts))

    logger.info(f"Saved to {path} ({len(rows)} new rows)")
    return pd.DataFrame(rows) if rows else pd.read_csv(path)


def run_scaling_stage(
    entry: RunEntry,
    profile: dict,
    llm_client: LLMClient | None = None,
    embedding_model: EmbeddingModel | None = None,
    progress_tracker=None,
) -> None:
    """Run a single scaling entry from the manifest.

    Thin wrapper that resolves the scaling dimension from entry.config
    (e.g., "scale_events_100", "scale_subs_50") and calls scale_events()
    or scale_subscriptions().

    Args:
        entry: RunEntry describing the scaling run.
        profile: Loaded mode profile from params.yaml.
        llm_client: Pre-initialised LLM client.
        embedding_model: Pre-initialised embedding model.
        progress_tracker: Optional progress callback (unused for now).
    """
    overrides = get_model_overrides(entry.model_key)
    batch_size = overrides["batch_size"] or 200
    llm_timeout = overrides["llm_timeout"] or 600

    dataset = load_dataset_by_name(entry.dataset, cache_dir="data")

    output_dir = Path(entry.result_file).parent

    p_scale = profile.get("scaling", {})
    event_counts = p_scale.get("event_counts", EVENT_COUNTS_DEFAULT)

    config_str = entry.config  # e.g., "scale_events_100"

    if config_str.startswith("scale_events"):
        scale_events(
            dataset, embedding_model, llm_client, entry.seed, output_dir,
            batch_size=batch_size, llm_timeout=llm_timeout,
            event_counts=event_counts,
        )
    elif config_str.startswith("scale_subs"):
        scale_subscriptions(
            dataset, embedding_model, llm_client, entry.seed, output_dir,
            batch_size=batch_size, llm_timeout=llm_timeout,
        )
    else:
        raise ValueError(f"Unknown scaling config: {config_str}")


def parse_args():
    parser = argparse.ArgumentParser(description="Scaling analysis")
    parser.add_argument("--dataset", type=str, default="D1")
    parser.add_argument("--dimension", type=str, default="all",
                        help="events, subscriptions, or all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--cache-dir", type=str, default="data")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override max_batch_events in RouterConfig (default: 50 for ollama, 200 otherwise)")
    parser.add_argument("--llm-timeout", type=int, default=None,
                        help="Override llm_timeout in RouterConfig (default: 600 for ollama, 600 otherwise)")
    parser.add_argument("--mode", type=str, default=None,
                        help="Override test mode from params.yaml (unit_smoke, integration_smoke, full)")
    return parser.parse_args()


def main():
    args = parse_args()
    load_dotenv(override=True)

    # Load profile defaults
    profile = None
    try:
        profile = load_profile(mode=args.mode)
    except (FileNotFoundError, ValueError) as e:
        if args.mode is not None:
            raise
        logger.debug(f"No params.yaml profile loaded: {e}")

    # Resolve scaling parameters from profile
    if profile:
        p_scale = profile.get("scaling", {})
        event_counts = p_scale.get("event_counts", EVENT_COUNTS_DEFAULT)
        if args.output_dir == "results":  # argparse default
            args.output_dir = str(output_dir_for_mode("results", profile["mode"], "scaling"))
    else:
        event_counts = EVENT_COUNTS_DEFAULT

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect ollama and set defaults accordingly
    is_ollama = "ollama" in args.llm_model.lower()
    batch_size = args.batch_size if args.batch_size is not None else (50 if is_ollama else 200)
    llm_timeout = args.llm_timeout if args.llm_timeout is not None else (600 if is_ollama else 600)
    if is_ollama:
        logger.info(f"Ollama model detected: using batch_size={batch_size}, llm_timeout={llm_timeout}s")

    # Load full dataset (no max_events, we subsample ourselves)
    dataset = load_dataset_by_name(args.dataset, cache_dir=args.cache_dir)
    logger.info(dataset.summary())

    if args.dry_run:
        llm_client = DryRunLLMClient(model=args.llm_model)
    else:
        llm_client = LLMClient(model=args.llm_model, api_key=os.getenv("OPENAI_API_KEY"))

    embedder = EmbeddingModel("all-MiniLM-L6-v2", cache_dir=args.cache_dir)

    dims = args.dimension.lower().split(",") if args.dimension.lower() != "all" else ["events", "subscriptions"]

    # Progress tracking
    model_key = args.llm_model.split("/")[-1].split(":")[0]
    if "qwen" in model_key.lower():
        model_key = "qwen7b" if "7b" in args.llm_model else "qwen32b"
    elif "haiku" in model_key.lower():
        model_key = "haiku"
    elif "sonnet" in model_key.lower():
        model_key = "sonnet"
    mode = profile["mode"] if profile else "unknown"

    for dim in dims:
        if dim == "events":
            stage_name = f"scaling-events-{args.dataset}-{model_key}"
        elif dim == "subscriptions":
            stage_name = f"scaling-subs-{args.dataset}-{model_key}"
        else:
            logger.warning(f"Unknown dimension: {dim}")
            continue

        tracker = ProgressTracker(stage_name, mode=mode)

        def on_scale_progress(idx, total):
            tracker.update(
                current_task=f"{dim} scaling",
                items_done=idx + 1,
                items_total=total,
                detail=f"Scale point {idx+1}/{total}",
                force=True,
            )

        if dim == "events":
            scale_events(dataset, embedder, llm_client, args.seed, output_dir, batch_size=batch_size, llm_timeout=llm_timeout, event_counts=event_counts, progress_callback=on_scale_progress)
        elif dim == "subscriptions":
            scale_subscriptions(dataset, embedder, llm_client, args.seed, output_dir, batch_size=batch_size, llm_timeout=llm_timeout, progress_callback=on_scale_progress)

        tracker.finish()


if __name__ == "__main__":
    main()
