#!/usr/bin/env python3
"""Parameter sensitivity analysis for Neural Router.

Runs the Neural Router (A3 config) while sweeping one parameter at a time,
keeping others at their defaults. Produces CSV results for plotting.

Experiments:
  1. k sweep: number of subscription clusters (k=1,5,10,15,19,25,30)
  2. tau sweep: cosine similarity threshold (tau=0.0,0.1,...,0.9)
  3. kappa sweep: top-K matches per event (kappa=1,2,3,5,7,10)
  4. embedding model sweep: 4 embedding models

Usage:
    python scripts/run_sensitivity.py --dataset D1 --sweep k
    python scripts/run_sensitivity.py --dataset D1 --sweep tau
    python scripts/run_sensitivity.py --dataset D1 --sweep kappa
    python scripts/run_sensitivity.py --dataset D1 --sweep embedding
    python scripts/run_sensitivity.py --dataset D1 --sweep all
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

import pandas as pd
from dotenv import load_dotenv

from src.data import load_dataset_by_name
from src.router import NeuralRouter, RouterConfig, ABLATION_CONFIGS
from src.embeddings import EmbeddingModel, EMBEDDING_MODELS
from src.evaluation import evaluate_matches
from src.llm import LLMClient, DryRunLLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Default parameter values (held constant during sweeps)
DEFAULTS = {
    "k": 19,
    "tau": 0.3,
    "kappa": 3,
    "embedding_model": "all-MiniLM-L6-v2",
    "seed": 42,
}

# Sweep ranges
K_VALUES = [1, 5, 10, 15, 19, 25, 30]
TAU_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
KAPPA_VALUES = [1, 2, 3, 5, 7, 10]


def run_one(
    dataset,
    embedder: EmbeddingModel,
    llm_client: LLMClient,
    k: int,
    tau: float,
    kappa: int,
    seed: int,
    label: str = "",
) -> dict:
    """Run a single configuration and return a result dict."""
    config = RouterConfig(
        **{
            **ABLATION_CONFIGS["A3"].__dict__,
            "seed": seed,
            "k": k,
            "tau": tau,
            "kappa": kappa,
        }
    )

    router = NeuralRouter(config=config, llm_client=llm_client, embedding_model=embedder)
    router.optimize_subscriptions(dataset.subscriptions)
    matches = router.match_events(dataset.events)

    result = evaluate_matches(
        matches=matches,
        dataset=dataset,
        config_name=label or f"k{k}_t{tau}_K{kappa}",
        seed=seed,
        router_stats=router.stats,
    )

    return {
        "label": label or f"k{k}_t{tau}_K{kappa}",
        "k": k,
        "tau": tau,
        "kappa": kappa,
        "seed": seed,
        "precision": result.precision.mean,
        "recall": result.recall.mean,
        "f1": result.f1.mean,
        "fpr": result.fpr.mean,
        "invocations": result.invocations,
        "compression_ratio": result.compression_ratio,
        "latency_s": result.latency_s,
    }


def sweep_k(dataset, embedder, llm_client, seeds, output_dir):
    """Sweep number of clusters k."""
    rows = []
    for k in K_VALUES:
        if k > dataset.num_subscriptions:
            continue
        for seed in seeds:
            logger.info(f"k sweep: k={k}, seed={seed}")
            llm_client.reset_stats()
            row = run_one(
                dataset, embedder, llm_client,
                k=k, tau=DEFAULTS["tau"], kappa=DEFAULTS["kappa"],
                seed=seed, label=f"k={k}",
            )
            rows.append(row)

    df = pd.DataFrame(rows)
    path = output_dir / f"sensitivity_k_{dataset.short_name}.csv"
    df.to_csv(path, index=False)
    logger.info(f"Saved k sweep to {path}")
    return df


def sweep_tau(dataset, embedder, llm_client, seeds, output_dir):
    """Sweep cosine threshold tau."""
    rows = []
    for tau in TAU_VALUES:
        for seed in seeds:
            logger.info(f"tau sweep: tau={tau}, seed={seed}")
            llm_client.reset_stats()
            row = run_one(
                dataset, embedder, llm_client,
                k=DEFAULTS["k"], tau=tau, kappa=DEFAULTS["kappa"],
                seed=seed, label=f"tau={tau}",
            )
            rows.append(row)

    df = pd.DataFrame(rows)
    path = output_dir / f"sensitivity_tau_{dataset.short_name}.csv"
    df.to_csv(path, index=False)
    logger.info(f"Saved tau sweep to {path}")
    return df


def sweep_kappa(dataset, embedder, llm_client, seeds, output_dir):
    """Sweep top-K matches kappa."""
    rows = []
    for kappa in KAPPA_VALUES:
        for seed in seeds:
            logger.info(f"kappa sweep: kappa={kappa}, seed={seed}")
            llm_client.reset_stats()
            row = run_one(
                dataset, embedder, llm_client,
                k=DEFAULTS["k"], tau=DEFAULTS["tau"], kappa=kappa,
                seed=seed, label=f"kappa={kappa}",
            )
            rows.append(row)

    df = pd.DataFrame(rows)
    path = output_dir / f"sensitivity_kappa_{dataset.short_name}.csv"
    df.to_csv(path, index=False)
    logger.info(f"Saved kappa sweep to {path}")
    return df


def sweep_embedding(dataset, llm_client, seeds, output_dir, cache_dir):
    """Sweep embedding models."""
    rows = []
    for model_name, model_path in EMBEDDING_MODELS.items():
        logger.info(f"Embedding sweep: {model_name}")
        embedder = EmbeddingModel(model_path, cache_dir=cache_dir)
        for seed in seeds:
            llm_client.reset_stats()
            row = run_one(
                dataset, embedder, llm_client,
                k=DEFAULTS["k"], tau=DEFAULTS["tau"], kappa=DEFAULTS["kappa"],
                seed=seed, label=model_name,
            )
            row["embedding_model"] = model_name
            rows.append(row)

    df = pd.DataFrame(rows)
    path = output_dir / f"sensitivity_embedding_{dataset.short_name}.csv"
    df.to_csv(path, index=False)
    logger.info(f"Saved embedding sweep to {path}")
    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Parameter sensitivity analysis")
    parser.add_argument("--dataset", type=str, default="D1")
    parser.add_argument("--sweep", type=str, default="all",
                        help="Which sweep: k, tau, kappa, embedding, or all")
    parser.add_argument("--seeds", type=str, default="42",
                        help="Comma-separated seeds (default: single seed for speed)")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--cache-dir", type=str, default="data")
    return parser.parse_args()


def main():
    args = parse_args()
    load_dotenv()

    seeds = [int(s) for s in args.seeds.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = load_dataset_by_name(args.dataset, cache_dir=args.cache_dir, max_events=args.max_events)
    logger.info(dataset.summary())

    # Initialize LLM
    if args.dry_run:
        llm_client = DryRunLLMClient(model=args.llm_model)
    else:
        llm_client = LLMClient(model=args.llm_model, api_key=os.getenv("OPENAI_API_KEY"))

    # Initialize default embedder
    embedder = EmbeddingModel(DEFAULTS["embedding_model"], cache_dir=args.cache_dir)

    sweeps = args.sweep.lower().split(",") if args.sweep.lower() != "all" else ["k", "tau", "kappa", "embedding"]

    for sweep in sweeps:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {sweep} sweep on {dataset.short_name}")
        logger.info(f"{'='*60}")

        if sweep == "k":
            sweep_k(dataset, embedder, llm_client, seeds, output_dir)
        elif sweep == "tau":
            sweep_tau(dataset, embedder, llm_client, seeds, output_dir)
        elif sweep == "kappa":
            sweep_kappa(dataset, embedder, llm_client, seeds, output_dir)
        elif sweep == "embedding":
            sweep_embedding(dataset, llm_client, seeds, output_dir, args.cache_dir)
        else:
            logger.warning(f"Unknown sweep: {sweep}")


if __name__ == "__main__":
    main()
