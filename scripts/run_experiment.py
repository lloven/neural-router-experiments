#!/usr/bin/env python3
"""Main experiment runner for the Neural Router evaluation (Section 4).

Orchestrates the full experimental pipeline: loads a dataset, runs ablation
configurations (A0-A6) and/or baseline methods across multiple random seeds,
evaluates results, and saves CSV + JSON output.

This is the primary entry point for reproducing the paper's results. For
parallel execution across multiple processes, see ``run_parallel.py``.

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

    # Resume a crashed run (skips already-completed config×seed pairs)
    python scripts/run_experiment.py --dataset D3 --configs A0,A3,A6 --baselines bm25,sbert --seeds 42 --output-tag D3_ablation --resume

    # Checkpoint every 100 events (resumes automatically if interrupted)
    python scripts/run_experiment.py --dataset D3 --configs A3 --seeds 42 --checkpoint-interval 100
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
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
from src.config import load_profile, output_dir_for_mode, get_model_overrides
from src.manifest import RunEntry
from src.progress import ProgressTracker

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

# Default checkpoint interval (events)
DEFAULT_CHECKPOINT_INTERVAL = 50


# ---------------------------------------------------------------------------
# Event-level checkpointing
# ---------------------------------------------------------------------------

def _checkpoint_path(
    output_dir: Path,
    config_name: str,
    dataset_name: str,
    seed: int,
    model_tag: str = "",
) -> Path:
    """Return the path for a checkpoint file (e.g., .checkpoints/A3_D3_seed42_qwen7b.pkl).

    Checkpoint files store partial MatchResult lists so that a crashed run can
    resume from the last saved position instead of reprocessing all events.
    The model_tag prevents collisions when multiple models run in parallel.
    """
    ckpt_dir = output_dir / ".checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{model_tag}" if model_tag else ""
    return ckpt_dir / f"{config_name}_{dataset_name}_seed{seed}{suffix}.pkl"


def load_checkpoint(
    output_dir: Path,
    config_name: str,
    dataset_name: str,
    seed: int,
    model_tag: str = "",
) -> tuple[list, int]:
    """Load a partial checkpoint if one exists.

    Returns:
        (partial_matches, num_completed): The list of MatchResults saved so far
        and the number of events already processed. Returns ([], 0) if no
        checkpoint exists.
    """
    path = _checkpoint_path(output_dir, config_name, dataset_name, seed, model_tag)
    if not path.exists():
        return [], 0
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        matches = data["matches"]
        num_done = data["num_completed"]
        logger.info(
            f"  Loaded checkpoint: {num_done} events already processed "
            f"({path.name})"
        )
        return matches, num_done
    except Exception as e:
        logger.warning(f"  Corrupt checkpoint {path.name}, starting fresh: {e}")
        return [], 0


def save_checkpoint(
    output_dir: Path,
    config_name: str,
    dataset_name: str,
    seed: int,
    matches: list,
    num_completed: int,
    model_tag: str = "",
) -> None:
    """Save partial match results to a checkpoint file.

    Writes atomically (write to .tmp then rename) to avoid corruption if the
    process is killed mid-write.
    """
    path = _checkpoint_path(output_dir, config_name, dataset_name, seed, model_tag)
    tmp_path = path.with_suffix(".tmp")
    data = {"matches": matches, "num_completed": num_completed}
    with open(tmp_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    # Retry rename up to 3 times (Dropbox sync can transiently lock files)
    for attempt in range(3):
        try:
            tmp_path.rename(path)
            break
        except OSError:
            if attempt < 2:
                time.sleep(1)
            else:
                raise
    logger.debug(f"  Checkpoint saved: {num_completed} events ({path.name})")


def cleanup_checkpoint(
    output_dir: Path,
    config_name: str,
    dataset_name: str,
    seed: int,
    model_tag: str = "",
) -> None:
    """Remove the checkpoint file after a config x seed completes successfully."""
    path = _checkpoint_path(output_dir, config_name, dataset_name, seed, model_tag)
    if path.exists():
        path.unlink()
        logger.debug(f"  Cleaned up checkpoint {path.name}")


def parse_args():
    parser = argparse.ArgumentParser(description="Neural Router experiment runner")

    parser.add_argument(
        "--dataset", type=str, default="D1",
        help="Dataset to use: D1, D2, D3, or 'all'"
    )
    parser.add_argument(
        "--configs", type=str, default=None,
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
        "--output-tag", type=str, default=None,
        help="Deterministic output prefix (overrides timestamp). For DVC pipelines."
    )
    parser.add_argument(
        "--cache-dir", type=str, default="data",
        help="Cache directory for datasets"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from partial results: skip config×seed pairs already in the output file"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Max events per LLM call (default: 200 for API, 50 for local models)"
    )
    parser.add_argument(
        "--llm-timeout", type=int, default=None,
        help="Per-request LLM timeout in seconds (default: 300 for API, 600 for local)"
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL,
        help=f"Save partial results every N events (default: {DEFAULT_CHECKPOINT_INTERVAL}). "
             "Set to 0 to disable event-level checkpointing."
    )
    parser.add_argument(
        "--max-event-words", type=int, default=None,
        help="Truncate each event's text to this many words before LLM matching. "
             "Reduces prompt size for datasets with long documents (e.g., D2, D3). "
             "Embeddings are computed on full text; only LLM prompts are truncated."
    )
    parser.add_argument(
        "--max-sub-words", type=int, default=None,
        help="Truncate each subscription's description to this many words in LLM prompts. "
             "Useful when subscription topics are verbose (e.g., D3/MN-DS hierarchical labels)."
    )

    parser.add_argument(
        "--mode", type=str, default=None,
        help="Override test mode from params.yaml (unit_smoke, integration_smoke, full). "
             "When set, unspecified CLI args use the profile defaults."
    )

    return parser.parse_args()


def get_default_k(dataset: Dataset) -> int:
    """Get default cluster count k for a dataset based on subscription count.

    Heuristic: use k equal to the number of subscriptions for small sets
    (D1 with 19 topics), and a fixed value for larger sets.

    Args:
        dataset: Loaded dataset.

    Returns:
        Default number of clusters.
    """
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
    batch_size: int | None = None,
    llm_timeout: int | None = None,
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
    output_dir: Path | None = None,
    max_event_words: int | None = None,
    max_sub_words: int | None = None,
    progress_tracker: ProgressTracker | None = None,
    tasks_done: int = 0,
    tasks_total: int = 0,
) -> EvaluationResult:
    """Run a single ablation configuration on a dataset with a given seed.

    Executes the full offline + online pipeline (Algorithms 1-2) and evaluates.
    When checkpoint_interval > 0 and output_dir is provided, partial match
    results are saved every checkpoint_interval events so that a crashed run
    can be resumed without reprocessing from the beginning.

    Args:
        config_name: Ablation variant key (e.g., "A3").
        dataset: Loaded dataset with events and subscriptions.
        embedding_model: Shared embedding model instance.
        llm_client: LLM client (real or dry-run).
        seed: Random seed for k-means.
        k_override: If set, overrides the default cluster count.
        tau: Cosine similarity threshold.
        kappa: Top-kappa matches per event.
        batch_size: Max events per LLM call (overrides config default).
        llm_timeout: Per-request LLM timeout in seconds.
        checkpoint_interval: Save partial results every N events. 0 = disabled.
        output_dir: Directory for checkpoint files. Required if checkpointing
                    is enabled.

    Returns:
        EvaluationResult with all accuracy and system metrics.
    """
    # Derive a short model tag for checkpoint filenames to avoid collisions
    # when multiple models run in parallel on the same output_dir
    model_id = llm_client.model.lower()
    if "qwen" in model_id:
        model_tag = "qwen7b"
    elif "haiku" in model_id:
        model_tag = "haiku"
    elif "sonnet" in model_id:
        model_tag = "sonnet"
    else:
        model_tag = model_id.split("/")[-1].split(":")[0][:12]

    overrides = {
        "seed": seed,
        "tau": tau,
        "kappa": kappa,
        "llm_model": llm_client.model,  # sync config with the actual LLM client model
    }
    if batch_size is not None:
        overrides["max_batch_events"] = batch_size
    if llm_timeout is not None:
        overrides["llm_timeout"] = llm_timeout
    if max_event_words is not None:
        overrides["max_event_words"] = max_event_words
    if max_sub_words is not None:
        overrides["max_sub_words"] = max_sub_words
    # Auto-detect max output tokens: Haiku supports only 4096
    if "claude-3-haiku" in llm_client.model.lower():
        overrides["llm_max_tokens"] = 4096
        overrides["max_parallel"] = 2  # Haiku has strict rate limits (50k tokens/min)
    # Local models: disable async (causes timeouts/crashes with Ollama), limit parallelism
    if "ollama" in llm_client.model.lower():
        overrides["max_parallel"] = 1
        overrides["use_async"] = False
    config = RouterConfig(**{
        **ABLATION_CONFIGS[config_name].__dict__,
        **overrides,
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

    # Offline: optimize subscriptions (cheap, always re-run for correctness)
    router.optimize_subscriptions(dataset.subscriptions)

    # Online: match events (with optional checkpointing)
    use_checkpoints = checkpoint_interval > 0 and output_dir is not None
    all_events = dataset.events

    if use_checkpoints:
        # Try to resume from a previous checkpoint
        prior_matches, num_done = load_checkpoint(
            output_dir, config_name, dataset.short_name, seed, model_tag,
        )
        if num_done > 0 and num_done < len(all_events):
            logger.info(
                f"  Resuming from event {num_done}/{len(all_events)} "
                f"({len(all_events) - num_done} remaining)"
            )
        elif num_done >= len(all_events):
            logger.info(f"  Checkpoint covers all {len(all_events)} events, skipping LLM matching")
            matches = prior_matches
            # Jump to evaluation
            result = evaluate_matches(
                matches=matches,
                dataset=dataset,
                config_name=config_name,
                seed=seed,
                router_stats=router.stats,
            )
            cleanup_checkpoint(output_dir, config_name, dataset.short_name, seed, model_tag)
            logger.info(
                f"  {config_name} seed={seed}: F1={result.f1.mean:.4f}, "
                f"I={result.invocations}, ρ={result.compression_ratio:.2f}, "
                f"L={result.latency_s:.1f}s"
            )
            return result

        # Process remaining events in checkpoint-sized chunks
        remaining_events = all_events[num_done:]
        matches = list(prior_matches)

        for chunk_start in range(0, len(remaining_events), checkpoint_interval):
            chunk_end = min(chunk_start + checkpoint_interval, len(remaining_events))
            chunk = remaining_events[chunk_start:chunk_end]
            abs_start = num_done + chunk_start
            abs_end = num_done + chunk_end

            logger.info(
                f"  Processing events {abs_start+1}-{abs_end}/{len(all_events)}"
            )
            chunk_matches = router.match_events(chunk)
            matches.extend(chunk_matches)

            # Save checkpoint after each chunk
            save_checkpoint(
                output_dir, config_name, dataset.short_name, seed,
                matches, abs_end, model_tag,
            )
            if progress_tracker:
                progress_tracker.update(
                    current_task=f"{config_name} seed={seed}",
                    items_done=abs_end,
                    items_total=len(all_events),
                    tasks_done=tasks_done,
                    tasks_total=tasks_total,
                    detail=f"Events {abs_end}/{len(all_events)}",
                    force=True,
                )
    else:
        # Original path: process all events at once
        if progress_tracker:
            progress_tracker.update(
                current_task=f"{config_name} seed={seed}",
                items_done=0,
                items_total=len(all_events),
                detail="Processing all events (no checkpointing)",
            )
        matches = router.match_events(all_events)

    # Evaluate
    result = evaluate_matches(
        matches=matches,
        dataset=dataset,
        config_name=config_name,
        seed=seed,
        router_stats=router.stats,
    )

    # Clean up checkpoint on success
    if use_checkpoints:
        cleanup_checkpoint(output_dir, config_name, dataset.short_name, seed, model_tag)

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
    """Run all specified baselines on a dataset.

    Args:
        dataset: Loaded dataset.
        baselines: List of baseline method names (e.g., ["bm25", "sbert"]).
        kappa: Top-kappa matches per event.

    Returns:
        List of EvaluationResults, one per baseline (failed baselines are skipped).
    """
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
    output_tag: str | None = None,
) -> Path:
    """Save results to CSV (summary) and JSON (with CIs and token counts).

    Args:
        results: List of EvaluationResults to persist.
        output_dir: Directory to write output files.
        tag: Descriptive prefix derived from dataset/config names.
        output_tag: If provided, use as deterministic prefix (for DVC pipelines).
                    Otherwise, a timestamp-based prefix is generated.

    Returns:
        Path to the saved CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_tag:
        prefix = output_tag
    else:
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


def _output_prefix(output_tag: str | None, tag: str) -> str:
    """Compute deterministic or timestamped output prefix."""
    if output_tag:
        return output_tag
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{tag}_{timestamp}" if tag else timestamp


def load_completed_keys(output_dir: Path, prefix: str) -> set[tuple[str, str, int]]:
    """Load (config_name, dataset, seed) triples already present in a partial results CSV.

    Returns an empty set if the file does not exist.
    """
    csv_path = output_dir / f"{prefix}_results.csv"
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path)
        keys = set()
        for _, row in df.iterrows():
            config = row.get("config", row.get("config_name", ""))
            dataset = row.get("dataset", "")
            seed = int(row.get("seed", -1))
            keys.add((str(config), str(dataset), seed))
        logger.info(f"Resuming: found {len(keys)} completed runs in {csv_path}")
        return keys
    except Exception as e:
        logger.warning(f"Could not read partial results from {csv_path}: {e}")
        return set()


def flush_result(result: EvaluationResult, output_dir: Path, prefix: str) -> None:
    """Append a single result to the CSV and JSON output files incrementally.

    Creates the files (with header) if they do not exist; appends otherwise.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{prefix}_results.csv"
    row = result.summary_row()
    df_row = pd.DataFrame([row])

    if csv_path.exists():
        df_row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(csv_path, index=False)

    # Append to JSON (read-modify-write; small file so this is fine)
    json_path = output_dir / f"{prefix}_results.json"
    json_entry = row.copy()
    json_entry["precision_ci"] = [result.precision.ci_lower, result.precision.ci_upper]
    json_entry["recall_ci"] = [result.recall.ci_lower, result.recall.ci_upper]
    json_entry["f1_ci"] = [result.f1.ci_lower, result.f1.ci_upper]
    json_entry["fpr_ci"] = [result.fpr.ci_lower, result.fpr.ci_upper]
    json_entry["tokens_prompt"] = result.tokens_prompt
    json_entry["tokens_response"] = result.tokens_response

    if json_path.exists():
        with open(json_path) as f:
            json_data = json.load(f)
    else:
        json_data = []
    json_data.append(json_entry)
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    logger.info(f"  Flushed result to {csv_path}")


def run_ablation_stage(
    entry: RunEntry,
    profile: dict,
    llm_client: LLMClient | None = None,
    embedding_model: EmbeddingModel | None = None,
    progress_tracker: ProgressTracker | None = None,
) -> list[EvaluationResult]:
    """Run a single ablation entry from the manifest.

    Thin wrapper that resolves model overrides via get_model_overrides(),
    loads the dataset if needed, and delegates to run_ablation_config()
    (for LLM configs) or run_baselines_on_dataset() (for baselines).

    Args:
        entry: RunEntry describing the ablation run.
        profile: Loaded mode profile from params.yaml.
        llm_client: Pre-initialised LLM client (caller manages lifecycle).
        embedding_model: Pre-initialised embedding model.
        progress_tracker: Optional progress callback for live status updates.

    Returns:
        List of EvaluationResult (one element for config runs, possibly
        multiple for baselines).
    """
    overrides = get_model_overrides(entry.model_key)

    # Load dataset
    dataset = load_dataset_by_name(
        entry.dataset,
        cache_dir="data",
        max_events=entry.max_events,
    )

    output_dir = Path(entry.result_file).parent

    if entry.config.startswith("baseline_"):
        # Baseline run
        method = entry.config.removeprefix("baseline_")
        return run_baselines_on_dataset(dataset, [method], kappa=3)

    # LLM config run
    result = run_ablation_config(
        config_name=entry.config,
        dataset=dataset,
        embedding_model=embedding_model,
        llm_client=llm_client,
        seed=entry.seed,
        batch_size=overrides["batch_size"],
        llm_timeout=overrides["llm_timeout"],
        checkpoint_interval=overrides["checkpoint_interval"],
        output_dir=output_dir,
        max_event_words=entry.max_event_words,
        progress_tracker=progress_tracker,
    )
    return [result]


def main():
    args = parse_args()

    # Load profile defaults (if --mode is set or params.yaml has a mode)
    profile = None
    try:
        profile = load_profile(mode=args.mode)
    except (FileNotFoundError, ValueError) as e:
        if args.mode is not None:
            raise  # explicit --mode should fail loudly
        logger.debug(f"No params.yaml profile loaded: {e}")

    # Resolve parameters: CLI args override profile defaults
    if profile:
        p_abl = profile.get("ablation", {})
        # Only override if user didn't explicitly pass the flag
        if args.configs is None:  # argparse default: no --configs given
            configs_list = p_abl.get("configs", ["A3"])
            args.configs = ",".join(configs_list)
        if args.baselines == "":  # argparse default
            baselines_list = p_abl.get("baselines", [])
            args.baselines = ",".join(baselines_list)
        if args.seeds == ",".join(map(str, DEFAULT_SEEDS)):  # argparse default
            args.seeds = ",".join(map(str, profile.get("seeds", DEFAULT_SEEDS)))
        if args.max_events is None:
            args.max_events = profile.get("max_events")
        if args.max_event_words is None:
            args.max_event_words = profile.get("max_event_words")
        if args.output_dir == "results":  # argparse default
            args.output_dir = str(output_dir_for_mode("results", profile["mode"], "ablation"))

    # Parse arguments
    datasets = ALL_DATASETS if args.dataset.lower() == "all" else args.dataset.split(",")
    # If no profile was loaded and no --configs given, fall back to the historical default.
    if args.configs is None:
        args.configs = "A3"
    configs = list(ABLATION_CONFIGS.keys()) if args.configs.lower() == "all" else args.configs.split(",")
    if args.baselines.lower() in ("all",):
        baselines = ALL_BASELINES
    elif args.baselines.lower() in ("none", ""):
        baselines = []
    else:
        baselines = args.baselines.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    output_dir = Path(args.output_dir)
    tag = f"{'_'.join(datasets)}_{'_'.join(configs)}"
    prefix = _output_prefix(args.output_tag, tag)

    # Load completed runs for --resume
    completed_keys: set[tuple[str, str, int]] = set()
    if args.resume:
        completed_keys = load_completed_keys(output_dir, prefix)

    # Load API keys from .env — parse manually because python-dotenv 1.2.2
    # silently returns empty values for symlinked .env files on macOS/Dropbox
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.is_symlink():
        env_path = env_path.resolve()
    if env_path.exists():
        with open(env_path) as _ef:
            for _line in _ef:
                _line = _line.strip()
                if _line and "=" in _line and not _line.startswith("#"):
                    _k, _v = _line.split("=", 1)
                    os.environ[_k] = _v

    # Initialize LLM client
    if args.dry_run:
        llm_client = DryRunLLMClient(model=args.llm_model)
        logger.info("DRY RUN: no LLM calls will be made")
    else:
        # Auto-detect timeout for local models
        client_timeout = args.llm_timeout
        if client_timeout is None and "ollama" in args.llm_model.lower():
            client_timeout = 900
        # Auto-detect max_tokens: Haiku supports only 4096
        max_tokens = 16384
        if "claude-3-haiku" in args.llm_model.lower():
            max_tokens = 4096
        llm_client = LLMClient(
            model=args.llm_model,
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=client_timeout,
            max_tokens=max_tokens,
        )

    # Initialize embedding model (with disk cache to avoid recomputing across seeds)
    embedding_model = EmbeddingModel(args.embedding_model, cache_dir=args.cache_dir)

    all_results = []

    # Determine model key for progress tracking
    model_key = args.llm_model.split("/")[-1].split(":")[0]  # e.g., "qwen2.5" or "claude-sonnet-4-20250514"
    # Simplify common model keys
    if "qwen" in model_key.lower():
        model_key = "qwen7b" if "7b" in args.llm_model else "qwen32b"
    elif "haiku" in model_key.lower():
        model_key = "haiku"
    elif "sonnet" in model_key.lower():
        model_key = "sonnet"

    mode = profile["mode"] if profile else "unknown"

    for ds_name in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset: {ds_name}")
        logger.info(f"{'='*60}")

        dataset = load_dataset_by_name(ds_name, cache_dir=args.cache_dir, max_events=args.max_events)
        logger.info(dataset.summary())

        stage_name = f"ablation-{ds_name}-{model_key}"
        tracker = ProgressTracker(stage_name, mode=mode)
        total_tasks = len(configs) * len(seeds) + len(baselines)
        tasks_done_count = 0

        # Run ablation configs
        for config_name in configs:
            if config_name not in ABLATION_CONFIGS:
                logger.warning(f"Unknown config: {config_name}, skipping")
                continue

            seed_results = []
            for seed in seeds:
                # Skip if already completed (--resume)
                if (config_name, ds_name, seed) in completed_keys:
                    logger.info(f"  Skipping {config_name} on {ds_name} seed={seed} (already completed)")
                    continue

                llm_client.reset_stats()
                # Auto-detect batch size and timeout for local models
                batch_size = args.batch_size
                llm_timeout = args.llm_timeout
                if "ollama" in args.llm_model.lower():
                    if batch_size is None:
                        batch_size = 10  # local models need small batches to avoid context-window slowdown
                    if llm_timeout is None:
                        llm_timeout = 900  # 15 min for local inference

                # Adaptive checkpoint interval: local models are slow,
                # checkpoint every 5 events to avoid losing hours of compute
                checkpoint_interval = args.checkpoint_interval
                if checkpoint_interval == DEFAULT_CHECKPOINT_INTERVAL:  # user didn't override
                    if "ollama" in args.llm_model.lower():
                        checkpoint_interval = 5   # ~minutes per checkpoint for local models
                    elif "haiku" in args.llm_model.lower():
                        checkpoint_interval = 25  # fast API, moderate checkpointing
                    # sonnet/other API: keep default (50)

                result = run_ablation_config(
                    config_name=config_name,
                    dataset=dataset,
                    embedding_model=embedding_model,
                    llm_client=llm_client,
                    seed=seed,
                    k_override=args.k,
                    tau=args.tau,
                    kappa=args.kappa,
                    batch_size=batch_size,
                    llm_timeout=llm_timeout,
                    checkpoint_interval=checkpoint_interval,
                    output_dir=output_dir,
                    max_event_words=args.max_event_words,
                    max_sub_words=args.max_sub_words,
                    progress_tracker=tracker,
                    tasks_done=tasks_done_count,
                    tasks_total=total_tasks,
                )
                seed_results.append(result)
                all_results.append(result)

                # Flush incrementally after each config×seed
                flush_result(result, output_dir, prefix)
                tasks_done_count += 1
                tracker.update(
                    current_task=f"{config_name} seed={seed}",
                    items_done=0,
                    items_total=0,
                    tasks_done=tasks_done_count,
                    tasks_total=total_tasks,
                    detail=f"Completed {config_name} seed={seed}",
                    force=True,
                )

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
            for br in baseline_results:
                all_results.append(br)
                # Baselines have seed=-1; skip if already present for this dataset
                br_row = br.summary_row()
                br_config = br_row.get("config", br_row.get("config_name", ""))
                br_dataset = br_row.get("dataset", ds_name)
                if (br_config, br_dataset, -1) not in completed_keys:
                    flush_result(br, output_dir, prefix)
                    tasks_done_count += 1
                    tracker.update(tasks_done=tasks_done_count, tasks_total=total_tasks, force=True)

        tracker.finish(tasks_done=tasks_done_count, tasks_total=total_tasks)

    # Final save: overwrite with clean complete file (de-duplicated, sorted)
    if all_results:
        save_results(all_results, output_dir, tag=tag, output_tag=args.output_tag)

    # Print summary table
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    if all_results:
        df = pd.DataFrame([r.summary_row() for r in all_results])
        print(df.to_string(index=False))
    else:
        logger.info("No new results (all runs already completed)")

    # Aggregate token/cost stats from evaluation results (router stats),
    # not from LLMClient (which misses async calls via litellm.acompletion).
    if all_results:
        total_invocations = sum(r.invocations for r in all_results)
        total_prompt = sum(r.tokens_prompt for r in all_results)
        total_response = sum(r.tokens_response for r in all_results)
        total_latency = sum(r.latency_s for r in all_results)
        # Standardised cost at GPT-4o-mini rates ($0.15/1M in, $0.60/1M out)
        total_cost = (total_prompt * 0.15 + total_response * 0.60) / 1_000_000
        logger.info(
            f"\nLLM Stats: {total_invocations} invocations, "
            f"{total_prompt} prompt tokens, {total_response} response tokens, "
            f"${total_cost:.4f} est. cost (GPT-4o-mini rates), "
            f"{total_latency:.1f}s total latency"
        )


if __name__ == "__main__":
    main()
