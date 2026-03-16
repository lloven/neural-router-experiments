#!/usr/bin/env python3
"""Main experiment runner for Neural Router evaluation.

Usage:
    # Run full ablation on D1 (CardiffNLP)
    python scripts/run_experiment.py --dataset D1 --configs A0,A1,A2,A3,A4,A5,A6

    # Run single config with specific seed
    python scripts/run_experiment.py --dataset D1 --configs A3 --seeds 42

    # Run baselines only
    python scripts/run_experiment.py --dataset D1 --baselines bm25,sbert,cross_encoder,tfidf,glove,word2vec

    # Run all experiments on all datasets
    python scripts/run_experiment.py --dataset all --configs all --baselines all

    # Dry run (no LLM calls, estimate costs)
    python scripts/run_experiment.py --dataset D1 --configs A3 --dry-run

    # Limit events for testing
    python scripts/run_experiment.py --dataset D1 --configs A3 --max-events 100
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data import load_dataset_by_name, Dataset
from src.router import NeuralRouter, RouterConfig, ABLATION_CONFIGS
from src.baselines import run_baseline
from src.evaluation import evaluate_matches, aggregate_seeds, EvaluationResult
from src.embeddings import EmbeddingModel
from src.llm import LLMClient, DryRunLLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# All baselines
ALL_BASELINES = ["bm25", "sbert", "cross_encoder", "tfidf", "glove", "word2vec"]

# All datasets
ALL_DATASETS = ["D1", "D2", "D3"]

# Default seeds
DEFAULT_SEEDS = [42, 123, 456, 789, 1024]


def parse_args():
    parser = argparse.ArgumentParser(description="Neural Router experiment runner")

    parser.add_argument(
        "--dataset", type=str, default="D1",
        help="Dataset to use: D1, D2, D3, or 'all'"
    )
    parser.add_argument(
        "--configs", type=str, default="A3",
        help="Comma-separated ablation configs (A0-A6) or 'all'"
    )
    parser.add_argument(
        "--baselines", type=str, default="",
        help="Comma-separated baselines or 'all'"
    )
    parser.add_argument(
        "--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)),
        help="Comma-separated random seeds"
    )
    parser.add_argument(
        "--kappa", type=int, default=3,
        help="Top-K matches per event"
    )
    parser.add_argument(
        "--k", type=int, default=None,
        help="Override number of clusters (default: per-dataset)"
    )
    parser.add_argument(
        "--tau", type=float, default=0.3,
        help="Cosine similarity threshold"
    )
    parser.add_argument(
        "--embedding-model", type=str, default="all-MiniLM-L6-v2",
        help="Embedding model name"
    )
    parser.add_argument(
        "--llm-model", type=str, default="gpt-4o-mini",
        help="LLM model name"
    )
    parser.add_argument(
        "--max-events", type=int, default=None,
        help="Limit number of events (for testing)"
    )
    parser.add_argument(
        "--max-parallel", type=int, default=10,
        help="Max parallel LLM instances"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Simulate without LLM calls (estimate costs)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--cache-dir", type=str, default="data",
        help="Cache directory for datasets"
    )

    return parser.parse_args()


def get_default_k(dataset: Dataset) -> int:
    """Get default k value for a dataset (matched to label count)."""
    n = dataset.num_subscriptions
    if n <= 20:
        return n  # D1: k=19
    elif n <= 120:
        return 20  # D3: reasonable cluster count
    else:
        return 30  # D2: 201 labels, use ~30 clusters


def run_ablation_config(
    config_name: str,
    dataset: Dataset,
    embedding_model: EmbeddingModel,
    llm_client: LLMClient,
    seed: int,
    k_override: int | None = None,
    tau: float = 0.3,
    kappa: int = 3,
) -> EvaluationResult:
    """Run a single ablation configuration on a dataset with a given seed."""
    config = RouterConfig(**{
        **ABLATION_CONFIGS[config_name].__dict__,
        "seed": seed,
        "tau": tau,
        "kappa": kappa,
    })

    if k_override is not None:
        config.k = k_override
    elif config.use_clustering:
        config.k = get_default_k(dataset)

    logger.info(
        f"Running {config_name} on {dataset.short_name} "
        f"(seed={seed}, k={config.k}, τ={config.tau}, κ={config.kappa})"
    )

    router = NeuralRouter(config=config, llm_client=llm_client, embedding_model=embedding_model)

    # Offline: optimize subscriptions
    router.optimize_subscriptions(dataset.subscriptions)

    # Online: match events
    matches = router.match_events(dataset.events)

    # Evaluate
    result = evaluate_matches(
        matches=matches,
        dataset=dataset,
        config_name=config_name,
        seed=seed,
        router_stats=router.stats,
    )

    logger.info(
        f"  {config_name} seed={seed}: F1={result.f1.mean:.4f}, "
        f"I={result.invocations}, ρ={result.compression_ratio:.2f}, "
        f"L={result.latency_s:.1f}s"
    )

    return result


def run_baselines_on_dataset(
    dataset: Dataset,
    baselines: list[str],
    kappa: int = 3,
) -> list[EvaluationResult]:
    """Run all specified baselines on a dataset."""
    results = []
    for method in baselines:
        logger.info(f"Running baseline {method} on {dataset.short_name}")
        try:
            baseline_result = run_baseline(method, dataset, kappa=kappa)
            eval_result = evaluate_matches(
                matches=baseline_result.matches,
                dataset=dataset,
                config_name=f"baseline_{method}",
            )
            eval_result.latency_s = baseline_result.latency_s
            results.append(eval_result)
            logger.info(
                f"  {method}: F1={eval_result.f1.mean:.4f}, "
                f"L={eval_result.latency_s:.1f}s"
            )
        except Exception as e:
            logger.error(f"  Baseline {method} failed: {e}")
    return results


def save_results(
    results: list[EvaluationResult],
    output_dir: Path,
    tag: str = "",
) -> Path:
    """Save results to CSV and JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{tag}_{timestamp}" if tag else timestamp

    # CSV summary
    rows = [r.summary_row() for r in results]
    df = pd.DataFrame(rows)
    csv_path = output_dir / f"{prefix}_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved results to {csv_path}")

    # JSON with full detail
    json_path = output_dir / f"{prefix}_results.json"
    json_data = []
    for r in results:
        d = r.summary_row()
        d["precision_ci"] = [r.precision.ci_lower, r.precision.ci_upper]
        d["recall_ci"] = [r.recall.ci_lower, r.recall.ci_upper]
        d["f1_ci"] = [r.f1.ci_lower, r.f1.ci_upper]
        d["fpr_ci"] = [r.fpr.ci_lower, r.fpr.ci_upper]
        d["tokens_prompt"] = r.tokens_prompt
        d["tokens_response"] = r.tokens_response
        json_data.append(d)

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    logger.info(f"Saved detailed results to {json_path}")

    return csv_path


def main():
    args = parse_args()

    # Parse arguments
    datasets = ALL_DATASETS if args.dataset.lower() == "all" else args.dataset.split(",")
    configs = list(ABLATION_CONFIGS.keys()) if args.configs.lower() == "all" else args.configs.split(",")
    baselines = ALL_BASELINES if args.baselines.lower() == "all" else (
        args.baselines.split(",") if args.baselines else []
    )
    seeds = [int(s) for s in args.seeds.split(",")]

    output_dir = Path(args.output_dir)

    # Load API key
    from dotenv import load_dotenv
    load_dotenv()

    # Initialize LLM client
    if args.dry_run:
        llm_client = DryRunLLMClient(model=args.llm_model)
        logger.info("DRY RUN: no LLM calls will be made")
    else:
        llm_client = LLMClient(
            model=args.llm_model,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    # Initialize embedding model (with disk cache to avoid recomputing across seeds)
    embedding_model = EmbeddingModel(args.embedding_model, cache_dir=args.cache_dir)

    all_results = []

    for ds_name in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset: {ds_name}")
        logger.info(f"{'='*60}")

        dataset = load_dataset_by_name(ds_name, cache_dir=args.cache_dir, max_events=args.max_events)
        logger.info(dataset.summary())

        # Run ablation configs
        for config_name in configs:
            if config_name not in ABLATION_CONFIGS:
                logger.warning(f"Unknown config: {config_name}, skipping")
                continue

            seed_results = []
            for seed in seeds:
                llm_client.reset_stats()
                result = run_ablation_config(
                    config_name=config_name,
                    dataset=dataset,
                    embedding_model=embedding_model,
                    llm_client=llm_client,
                    seed=seed,
                    k_override=args.k,
                    tau=args.tau,
                    kappa=args.kappa,
                )
                seed_results.append(result)
                all_results.append(result)

            # Aggregate across seeds
            if len(seed_results) > 1:
                agg = aggregate_seeds(seed_results)
                logger.info(
                    f"  {config_name} aggregated: "
                    f"F1={agg.f1.mean:.4f} ± {agg.f1.ci_width/2:.4f}"
                )

        # Run baselines
        if baselines:
            baseline_results = run_baselines_on_dataset(dataset, baselines, kappa=args.kappa)
            all_results.extend(baseline_results)

    # Save all results
    tag = f"{'_'.join(datasets)}_{'_'.join(configs)}"
    save_results(all_results, output_dir, tag=tag)

    # Print summary table
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    df = pd.DataFrame([r.summary_row() for r in all_results])
    print(df.to_string(index=False))

    if not args.dry_run:
        logger.info(f"\n{llm_client.stats_summary()}")


if __name__ == "__main__":
    main()
