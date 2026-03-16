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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

EVENT_COUNTS = [50, 100, 200, 500, 1000, 2000, 5000]


def scale_events(dataset, embedder, llm_client, seed, output_dir):
    """Scale the number of events |M| while keeping subscriptions fixed.

    Subsamples events deterministically at each size and runs A3.

    Args:
        dataset: Full dataset (events are subsampled from this).
        embedder: Embedding model instance.
        llm_client: LLM client (real or dry-run).
        seed: Random seed for subsampling and k-means.
        output_dir: Directory for CSV output.

    Returns:
        DataFrame with one row per event count.
    """
    rows = []
    rng = np.random.RandomState(seed)

    for n_events in EVENT_COUNTS:
        if n_events > dataset.num_events:
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

        config = RouterConfig(**{
            **ABLATION_CONFIGS["A3"].__dict__,
            "seed": seed,
            "k": min(19, dataset.num_subscriptions),
            "kappa": 3,
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

        rows.append({
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
        })

    df = pd.DataFrame(rows)
    path = output_dir / f"scaling_events_{dataset.short_name}.csv"
    df.to_csv(path, index=False)
    logger.info(f"Saved to {path}")
    return df


def scale_subscriptions(dataset, embedder, llm_client, seed, output_dir):
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

    Returns:
        DataFrame with one row per subscription count.
    """
    if dataset.num_subscriptions < 20:
        logger.info(f"Skipping subscription scaling for {dataset.short_name} (only {dataset.num_subscriptions} subs)")
        return pd.DataFrame()

    sub_counts = [5, 10, 20, 50, 100]
    sub_counts = [n for n in sub_counts if n <= dataset.num_subscriptions]
    sub_counts.append(dataset.num_subscriptions)  # always include full set

    rows = []
    rng = np.random.RandomState(seed)

    # Use a fixed subset of events
    n_events = min(500, dataset.num_events)
    event_indices = rng.choice(dataset.num_events, n_events, replace=False)
    sub_events = [dataset.events[i] for i in sorted(event_indices)]

    for n_subs in sub_counts:
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
        config = RouterConfig(**{
            **ABLATION_CONFIGS["A3"].__dict__,
            "seed": seed,
            "k": k,
            "kappa": 3,
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

        rows.append({
            "n_events": n_events,
            "n_subscriptions": len(chosen_subs),
            "k": k,
            "f1": result.f1.mean,
            "precision": result.precision.mean,
            "recall": result.recall.mean,
            "invocations": result.invocations,
            "latency_s": wall_time,
            "cost_per_1k": result.cost_per_1k,
        })

    df = pd.DataFrame(rows)
    path = output_dir / f"scaling_subs_{dataset.short_name}.csv"
    df.to_csv(path, index=False)
    logger.info(f"Saved to {path}")
    return df


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
    return parser.parse_args()


def main():
    args = parse_args()
    load_dotenv()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load full dataset (no max_events, we subsample ourselves)
    dataset = load_dataset_by_name(args.dataset, cache_dir=args.cache_dir)
    logger.info(dataset.summary())

    if args.dry_run:
        llm_client = DryRunLLMClient(model=args.llm_model)
    else:
        llm_client = LLMClient(model=args.llm_model, api_key=os.getenv("OPENAI_API_KEY"))

    embedder = EmbeddingModel("all-MiniLM-L6-v2", cache_dir=args.cache_dir)

    dims = args.dimension.lower().split(",") if args.dimension.lower() != "all" else ["events", "subscriptions"]

    for dim in dims:
        if dim == "events":
            scale_events(dataset, embedder, llm_client, args.seed, output_dir)
        elif dim == "subscriptions":
            scale_subscriptions(dataset, embedder, llm_client, args.seed, output_dir)
        else:
            logger.warning(f"Unknown dimension: {dim}")


if __name__ == "__main__":
    main()
