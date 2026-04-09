#!/usr/bin/env python3
"""Crossover validation experiment (Phase 1c).

Demonstrates that A0 (raw LLM) dominates at low subscription counts but
A4 (full pipeline) wins when subscriptions exceed the context window.
This produces the paper's central figure.

Design:
  - Context-constrained model (simulated via --max-context-tokens, e.g., W=4096)
  - Subscription volumes: [50, 100, 200, 500, 1000, 2000]
  - Configs: A0 (raw) vs A4 (full pipeline)
  - Seeds: [42, 123, 456]
  - Metrics: F1, latency, token cost, effective subscription count

The A4 pipeline compresses subscriptions first (CoverAndMerge), so more
subscriptions fit within the budget.  A0 puts all subscriptions raw into
the prompt, hitting the window limit sooner.

Usage:
    python scripts/run_crossover.py --dataset D1 --max-context-tokens 4096
    python scripts/run_crossover.py --dataset D1 --max-context-tokens 4096 --dry-run
    python scripts/run_crossover.py --dataset D1 --max-context-tokens 4096 \\
        --configs A0,A4 --sub-volumes 50,100,200,500,1000,2000
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.data import load_dataset_by_name, Dataset, Subscription
from src.router import NeuralRouter, RouterConfig, ABLATION_CONFIGS
from src.embeddings import EmbeddingModel
from src.evaluation import evaluate_matches
from src.llm import LLMClient, DryRunLLMClient
from src.config import load_profile, output_dir_for_mode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_SUB_VOLUMES = [50, 100, 200, 500, 1000, 2000]
DEFAULT_CONFIGS = ["A0", "A4"]
DEFAULT_SEEDS = [42, 123, 456]


def _append_row(row: dict, path: Path):
    """Append a single result row to a CSV, creating it with header if needed."""
    write_header = not path.exists() or path.stat().st_size == 0
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", header=write_header, index=False)


def _load_completed_keys(path: Path) -> set[tuple[str, int, int]]:
    """Load already-completed (config, n_subscriptions, seed) triples from CSV."""
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path)
        return {
            (str(row["config"]), int(row["n_subscriptions"]), int(row["seed"]))
            for _, row in df.iterrows()
        }
    except Exception:
        return set()


def _subsample_subscriptions(
    subscriptions: list[Subscription],
    n: int,
    seed: int,
) -> list[Subscription]:
    """Deterministically subsample n subscriptions from the full set."""
    if n >= len(subscriptions):
        return list(subscriptions)
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(subscriptions), n, replace=False)
    return [subscriptions[i] for i in sorted(indices)]


def run_crossover_point(
    dataset: Dataset,
    config_name: str,
    n_subs: int,
    max_context_tokens: int,
    seed: int,
    llm_client: LLMClient,
    embedder: EmbeddingModel,
    output_dir: Path,
    batch_size: int = 200,
    llm_timeout: int = 600,
) -> dict:
    """Run one crossover data point: (config, n_subs, seed).

    Subsamples subscriptions to n_subs, then runs the config with
    max_context_tokens as the hard token budget.  Returns a dict
    with all metrics.

    Args:
        dataset: Full dataset (subscriptions are subsampled from this).
        config_name: Ablation variant key (e.g., "A0", "A4").
        n_subs: Number of subscriptions to use.
        max_context_tokens: Hard token budget for the LLM context window.
        seed: Random seed for subsampling and k-means.
        llm_client: LLM client (real or dry-run).
        embedder: Embedding model instance.
        output_dir: Output directory (for logging).
        batch_size: Max events per LLM call.
        llm_timeout: Per-request timeout in seconds.

    Returns:
        Dict with all crossover metrics for one data point.
    """
    # Subsample subscriptions
    sub_subs = _subsample_subscriptions(dataset.subscriptions, n_subs, seed)

    # Rebuild dataset with subsampled subscriptions
    sub_dataset = Dataset(
        name=dataset.name,
        short_name=dataset.short_name,
        events=dataset.events,
        subscriptions=sub_subs,
        metadata=dataset.metadata,
    )

    logger.info(
        f"Crossover: {config_name}, |S|={n_subs}, seed={seed}, "
        f"W={max_context_tokens}"
    )

    # Reset LLM stats for clean accounting
    if hasattr(llm_client, 'reset_stats'):
        llm_client.reset_stats()

    # Build config with context truncation
    model_overrides = {}
    if "ollama" in llm_client.model.lower():
        model_overrides["max_parallel"] = 1
        model_overrides["use_async"] = False
    if "claude-3-haiku" in llm_client.model.lower():
        model_overrides["llm_max_tokens"] = 4096
        model_overrides["max_parallel"] = 2

    base_config = ABLATION_CONFIGS[config_name]
    k = min(max(1, n_subs // 2), 30) if base_config.use_clustering else 1

    config = RouterConfig(**{
        **base_config.__dict__,
        "seed": seed,
        "k": k,
        "kappa": 3,
        "llm_model": llm_client.model,
        "max_batch_events": batch_size,
        "llm_timeout": llm_timeout,
        "max_context_tokens": max_context_tokens,
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
        config_name=f"{config_name}_S{n_subs}",
        seed=seed,
        router_stats=router.stats,
    )

    return {
        "config": config_name,
        "n_subscriptions": n_subs,
        "seed": seed,
        "max_context_tokens": max_context_tokens,
        "n_subs_effective": router.stats.num_subscriptions_effective,
        "f1": result.f1.mean,
        "precision": result.precision.mean,
        "recall": result.recall.mean,
        "invocations": result.invocations,
        "compression_ratio": result.compression_ratio,
        "latency_s": wall_time,
        "tokens_prompt": result.tokens_prompt,
        "tokens_response": result.tokens_response,
        "cost_per_1k": result.cost_per_1k,
    }


def run_crossover_sweep(
    dataset: Dataset,
    configs: list[str],
    sub_volumes: list[int],
    max_context_tokens: int,
    seeds: list[int],
    llm_client: LLMClient,
    embedder: EmbeddingModel,
    output_dir: Path,
    batch_size: int = 200,
    llm_timeout: int = 600,
) -> pd.DataFrame:
    """Run the full crossover sweep: all (config, n_subs, seed) combos.

    Writes results incrementally to CSV for crash recovery. Resumes
    from existing CSV on restart.

    Args:
        dataset: Full dataset.
        configs: List of config names (e.g., ["A0", "A4"]).
        sub_volumes: List of subscription counts to test.
        max_context_tokens: Hard token budget.
        seeds: Random seeds.
        llm_client: LLM client.
        embedder: Embedding model.
        output_dir: Directory for CSV output.
        batch_size: Max events per LLM call.
        llm_timeout: Per-request timeout.

    Returns:
        DataFrame with all results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"crossover_{dataset.short_name}.csv"
    done = _load_completed_keys(csv_path)
    rows = []

    for config_name in configs:
        for n_subs in sub_volumes:
            if n_subs > dataset.num_subscriptions:
                logger.info(
                    f"Skipping |S|={n_subs} (dataset has only "
                    f"{dataset.num_subscriptions} subscriptions)"
                )
                continue

            for seed in seeds:
                key = (config_name, n_subs, seed)
                if key in done:
                    logger.info(f"Crossover: {key} already done, skipping")
                    continue

                row = run_crossover_point(
                    dataset=dataset,
                    config_name=config_name,
                    n_subs=n_subs,
                    max_context_tokens=max_context_tokens,
                    seed=seed,
                    llm_client=llm_client,
                    embedder=embedder,
                    output_dir=output_dir,
                    batch_size=batch_size,
                    llm_timeout=llm_timeout,
                )
                rows.append(row)
                _append_row(row, csv_path)
                logger.info(
                    f"  {config_name} |S|={n_subs} seed={seed}: "
                    f"F1={row['f1']:.4f}, effective={row['n_subs_effective']}"
                )

    # Return full DataFrame (including previously completed rows)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame(rows)


def run_crossover_stage(
    entry,
    profile: dict,
    llm_client: LLMClient | None = None,
    embedding_model: EmbeddingModel | None = None,
    progress_tracker=None,
) -> None:
    """Run a crossover entry from the manifest.

    Thin wrapper for integration with the run_one.py dispatcher.
    """
    from src.config import get_model_overrides

    overrides = get_model_overrides(entry.model_key)
    batch_size = overrides["batch_size"] or 200
    llm_timeout = overrides["llm_timeout"] or 600

    dataset = load_dataset_by_name(entry.dataset, cache_dir="data")

    output_dir = Path(entry.result_file).parent

    crossover_cfg = profile.get("crossover", {})
    sub_volumes = crossover_cfg.get("sub_volumes", DEFAULT_SUB_VOLUMES)
    max_context_tokens = crossover_cfg.get("max_context_tokens", 4096)
    configs = crossover_cfg.get("configs", DEFAULT_CONFIGS)
    seeds = crossover_cfg.get("seeds", DEFAULT_SEEDS)

    run_crossover_sweep(
        dataset=dataset,
        configs=configs,
        sub_volumes=sub_volumes,
        max_context_tokens=max_context_tokens,
        seeds=seeds,
        llm_client=llm_client,
        embedder=embedding_model,
        output_dir=output_dir,
        batch_size=batch_size,
        llm_timeout=llm_timeout,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Crossover validation experiment")
    parser.add_argument("--dataset", type=str, default="D1")
    parser.add_argument("--configs", type=str, default=",".join(DEFAULT_CONFIGS),
                        help="Comma-separated configs (default: A0,A4)")
    parser.add_argument("--sub-volumes", type=str,
                        default=",".join(map(str, DEFAULT_SUB_VOLUMES)),
                        help="Comma-separated subscription volumes")
    parser.add_argument("--max-context-tokens", type=int, default=4096,
                        help="Hard token budget for context window (simulates constrained model)")
    parser.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)),
                        help="Comma-separated random seeds")
    parser.add_argument("--llm-model", type=str, default="ollama/qwen2.5:7b")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--cache-dir", type=str, default="data")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--llm-timeout", type=int, default=None)
    parser.add_argument("--mode", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    load_dotenv(override=True)

    profile = None
    try:
        profile = load_profile(mode=args.mode)
    except (FileNotFoundError, ValueError) as e:
        if args.mode is not None:
            raise
        logger.debug(f"No params.yaml profile loaded: {e}")

    if profile and args.output_dir == "results":
        args.output_dir = str(output_dir_for_mode("results", profile["mode"], "crossover"))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    is_ollama = "ollama" in args.llm_model.lower()
    batch_size = args.batch_size or (50 if is_ollama else 200)
    llm_timeout = args.llm_timeout or (600 if is_ollama else 600)

    dataset = load_dataset_by_name(args.dataset, cache_dir=args.cache_dir)
    logger.info(dataset.summary())

    if args.dry_run:
        llm_client = DryRunLLMClient(model=args.llm_model)
    else:
        llm_client = LLMClient(model=args.llm_model)

    embedder = EmbeddingModel("all-MiniLM-L6-v2", cache_dir=args.cache_dir)

    configs = [c.strip() for c in args.configs.split(",")]
    sub_volumes = [int(v.strip()) for v in args.sub_volumes.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    run_crossover_sweep(
        dataset=dataset,
        configs=configs,
        sub_volumes=sub_volumes,
        max_context_tokens=args.max_context_tokens,
        seeds=seeds,
        llm_client=llm_client,
        embedder=embedder,
        output_dir=output_dir,
        batch_size=batch_size,
        llm_timeout=llm_timeout,
    )


if __name__ == "__main__":
    main()
