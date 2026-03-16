#!/usr/bin/env python3
"""Parallel experiment runner for the Neural Router.

Uses Python's ``ProcessPoolExecutor`` to run multiple experiment jobs
concurrently. Each (config, dataset, seed) triple or (baseline, dataset) pair
is an independent job that loads its own models in-process (avoiding pickling
issues with PyTorch models).

Progress is tracked in ``results/.progress.json`` so that long runs can be
monitored externally.

Usage:
    # Run full ablation on D1 with all seeds, 4 parallel workers
    python scripts/run_parallel.py --dataset D1 --configs all --workers 4

    # Run A3 on all datasets, 3 workers
    python scripts/run_parallel.py --dataset all --configs A3 --workers 3

    # Include baselines
    python scripts/run_parallel.py --dataset D1 --configs A3 --baselines all --workers 4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ALL_BASELINES = ["bm25", "sbert", "cross_encoder", "tfidf", "glove", "word2vec"]
ALL_DATASETS = ["D1", "D2", "D3"]
ALL_CONFIGS = ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]
DEFAULT_SEEDS = [42, 123, 456, 789, 1024]

# Progress tracking file (shared across experiments)
PROGRESS_FILE = "results/.progress.json"


def update_progress(job_id: str, status: str, detail: str = ""):
    """Update the shared JSON progress file (results/.progress.json).

    Args:
        job_id: Unique job identifier (e.g., "A3_D1_s42").
        status: One of "queued", "running", "done", "failed".
        detail: Optional extra info (e.g., F1 score or error message).
    """
    progress_path = Path(PROGRESS_FILE)
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if progress_path.exists():
            with open(progress_path) as f:
                progress = json.load(f)
        else:
            progress = {}
    except (json.JSONDecodeError, IOError):
        progress = {}

    progress[job_id] = {
        "status": status,
        "detail": detail,
        "timestamp": datetime.now().isoformat(),
    }

    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)


def run_single_job(job: dict) -> dict:
    """Run a single experiment job in a worker process.

    Each job is self-contained: it loads the dataset, initializes models,
    runs the pipeline, evaluates, and returns a result dict. Imports are
    done inside the function to avoid pickling issues across process boundaries.

    Args:
        job: Dict with keys "type" ("ablation" or "baseline"), "dataset",
            "config"/"baseline", "seed", and hyperparameter overrides.

    Returns:
        Dict suitable for a pandas DataFrame row, or an error dict.
    """
    from src.data import load_dataset_by_name
    from src.router import NeuralRouter, RouterConfig, ABLATION_CONFIGS
    from src.baselines import run_baseline
    from src.evaluation import evaluate_matches
    from src.embeddings import EmbeddingModel
    from src.llm import LLMClient, DryRunLLMClient

    job_id = job["job_id"]
    update_progress(job_id, "running")

    try:
        # Load dataset
        dataset = load_dataset_by_name(
            job["dataset"],
            cache_dir=job.get("cache_dir", "data"),
            max_events=job.get("max_events"),
        )

        if job["type"] == "ablation":
            # Initialize models (per-process to avoid pickling issues)
            embedding_model = EmbeddingModel(
                job.get("embedding_model", "all-MiniLM-L6-v2"),
                cache_dir=job.get("cache_dir", "data"),
            )

            if job.get("dry_run"):
                llm_client = DryRunLLMClient(model=job.get("llm_model", "gpt-4o-mini"))
            else:
                llm_client = LLMClient(
                    model=job.get("llm_model", "gpt-4o-mini"),
                    api_key=os.getenv("OPENAI_API_KEY"),
                )

            config = RouterConfig(**{
                **ABLATION_CONFIGS[job["config"]].__dict__,
                "seed": job["seed"],
                "tau": job.get("tau", 0.3),
                "kappa": job.get("kappa", 3),
            })

            if job.get("k"):
                config.k = job["k"]
            elif config.use_clustering:
                n = dataset.num_subscriptions
                config.k = n if n <= 20 else (20 if n <= 120 else 30)

            router = NeuralRouter(config=config, llm_client=llm_client, embedding_model=embedding_model)
            router.optimize_subscriptions(dataset.subscriptions)
            matches = router.match_events(dataset.events)

            result = evaluate_matches(
                matches=matches,
                dataset=dataset,
                config_name=job["config"],
                seed=job["seed"],
                router_stats=router.stats,
            )

            row = result.summary_row()
            update_progress(job_id, "done", f"F1={result.f1.mean:.4f}")
            return row

        elif job["type"] == "baseline":
            baseline_result = run_baseline(
                job["baseline"],
                dataset,
                kappa=job.get("kappa", 3),
            )
            result = evaluate_matches(
                matches=baseline_result.matches,
                dataset=dataset,
                config_name=f"baseline_{job['baseline']}",
            )
            result.latency_s = baseline_result.latency_s

            row = result.summary_row()
            update_progress(job_id, "done", f"F1={result.f1.mean:.4f}")
            return row

    except Exception as e:
        update_progress(job_id, "failed", str(e))
        logger.error(f"Job {job_id} failed: {e}")
        return {"job_id": job_id, "error": str(e)}


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel experiment runner")
    parser.add_argument("--dataset", type=str, default="D1")
    parser.add_argument("--configs", type=str, default="A3")
    parser.add_argument("--baselines", type=str, default="")
    parser.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--kappa", type=int, default=3)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--tau", type=float, default=0.3)
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--output-tag", type=str, default=None,
                        help="Deterministic output prefix (overrides timestamp). For DVC.")
    parser.add_argument("--cache-dir", type=str, default="data")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load API key from .env
    from dotenv import load_dotenv
    load_dotenv()

    datasets = ALL_DATASETS if args.dataset.lower() == "all" else args.dataset.split(",")
    configs = ALL_CONFIGS if args.configs.lower() == "all" else args.configs.split(",")
    baselines = ALL_BASELINES if args.baselines.lower() == "all" else (
        args.baselines.split(",") if args.baselines else []
    )
    seeds = [int(s) for s in args.seeds.split(",")]

    # Build job list
    jobs = []
    for ds in datasets:
        for cfg in configs:
            for seed in seeds:
                job_id = f"{cfg}_{ds}_s{seed}"
                jobs.append({
                    "job_id": job_id,
                    "type": "ablation",
                    "config": cfg,
                    "dataset": ds,
                    "seed": seed,
                    "kappa": args.kappa,
                    "k": args.k,
                    "tau": args.tau,
                    "embedding_model": args.embedding_model,
                    "llm_model": args.llm_model,
                    "max_events": args.max_events,
                    "dry_run": args.dry_run,
                    "cache_dir": args.cache_dir,
                })

        for bl in baselines:
            job_id = f"bl_{bl}_{ds}"
            jobs.append({
                "job_id": job_id,
                "type": "baseline",
                "baseline": bl,
                "dataset": ds,
                "kappa": args.kappa,
                "max_events": args.max_events,
                "cache_dir": args.cache_dir,
            })

    logger.info(f"Total jobs: {len(jobs)}, workers: {args.workers}")

    # Initialize progress
    for job in jobs:
        update_progress(job["job_id"], "queued")

    # Run in parallel
    results = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_job, job): job for job in jobs}

        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                if "error" not in result:
                    results.append(result)
                    logger.info(f"✓ {job['job_id']}: F1={result.get('f1', 'N/A')}")
                else:
                    logger.error(f"✗ {job['job_id']}: {result['error']}")
            except Exception as e:
                logger.error(f"✗ {job['job_id']}: {e}")

    elapsed = time.time() - t0
    logger.info(f"\nCompleted {len(results)}/{len(jobs)} jobs in {elapsed:.1f}s")

    # Save results
    if results:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.output_tag:
            prefix = args.output_tag
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = f"parallel_{timestamp}"

        df = pd.DataFrame(results)
        csv_path = output_dir / f"{prefix}_results.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved to {csv_path}")

        print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
