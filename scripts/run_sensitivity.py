#!/usr/bin/env python3
"""Parameter sensitivity analysis for the Neural Router (Section 4.6, Fig. 6).

Runs the full pipeline (A3 config) while sweeping one hyperparameter at a
time and keeping the others at their default values. Produces per-sweep CSV
files in the output directory, ready for plotting.

Sweeps (each corresponds to a panel in Fig. 6):
  1. k sweep     -- number of subscription clusters: k=1,5,10,15,19,25,30
  2. tau sweep   -- cosine similarity threshold: tau=0.0,0.1,...,0.9
  3. kappa sweep -- top-K matches per event: kappa=1,2,3,5,7,10
  4. embedding   -- four sentence-transformer models

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
from src.config import load_profile, output_dir_for_mode, get_model_overrides
from src.manifest import RunEntry
from src.progress import ProgressTracker

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
K_VALUES_DEFAULT = [1, 5, 10, 15, 19, 25, 30]
TAU_VALUES_DEFAULT = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
KAPPA_VALUES_DEFAULT = [1, 2, 3, 5, 7, 10]


def run_one(
    dataset,
    embedder: EmbeddingModel,
    llm_client: LLMClient,
    k: int,
    tau: float,
    kappa: int,
    seed: int,
    label: str = "",
    batch_size: int = 200,
    llm_timeout: int = 600,
    max_event_words: int | None = None,
) -> dict:
    """Run a single A3 configuration with specified hyperparameters.

    Args:
        dataset: Loaded dataset.
        embedder: Embedding model instance.
        llm_client: LLM client (real or dry-run).
        k: Number of subscription clusters.
        tau: Cosine similarity threshold.
        kappa: Top-kappa matches per event.
        seed: Random seed for k-means.
        label: Human-readable label for this data point.
        batch_size: Hard cap on events per LLM call (max_batch_events).
        llm_timeout: Per-request timeout in seconds.

    Returns:
        Dict with hyperparameters and evaluation metrics (one row).
    """
    overrides = {
        "seed": seed,
        "k": k,
        "tau": tau,
        "kappa": kappa,
        "llm_model": llm_client.model,
        "max_batch_events": batch_size,
        "llm_timeout": llm_timeout,
    }
    if max_event_words is not None:
        overrides["max_event_words"] = max_event_words
    # Ollama: disable async, limit parallelism (same as run_experiment.py)
    if "ollama" in llm_client.model.lower():
        overrides["max_parallel"] = 1
        overrides["use_async"] = False
    # Haiku: max output tokens is 4096 (not 16384), strict rate limits
    if "claude-3-haiku" in llm_client.model.lower():
        overrides["llm_max_tokens"] = 4096
        overrides["max_parallel"] = 2
    config = RouterConfig(
        **{**ABLATION_CONFIGS["A3"].__dict__, **overrides}
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


def _append_row(row: dict, path: Path):
    """Append a single result row to a CSV, creating it with a header if needed."""
    write_header = not path.exists() or path.stat().st_size == 0
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", header=write_header, index=False)


def _load_completed(path: Path, key_col: str) -> set[str]:
    """Load already-completed parameter values from a partial CSV for resume."""
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path)
        return set(df[key_col].astype(str).tolist())
    except Exception:
        return set()


def sweep_k(dataset, embedder, llm_client, seeds, output_dir, batch_size=200, llm_timeout=600, max_event_words=None, k_values=None, progress_callback=None):
    """Sweep number of clusters k (Fig. 6a)."""
    if k_values is None:
        k_values = K_VALUES_DEFAULT
    path = output_dir / f"sensitivity_k_{dataset.short_name}.csv"
    done = _load_completed(path, "label")
    rows = []
    for i, k in enumerate(k_values):
        if k > dataset.num_subscriptions:
            continue
        for seed in seeds:
            label = f"k={k}"
            if label in done:
                logger.info(f"k sweep: k={k}, seed={seed} — already done, skipping")
                continue
            logger.info(f"k sweep: k={k}, seed={seed}")
            llm_client.reset_stats()
            row = run_one(
                dataset, embedder, llm_client,
                k=k, tau=DEFAULTS["tau"], kappa=DEFAULTS["kappa"],
                seed=seed, label=label,
                batch_size=batch_size, llm_timeout=llm_timeout, max_event_words=max_event_words,
            )
            rows.append(row)
            _append_row(row, path)
            logger.info(f"  k={k} seed={seed}: F1={row.get('f1', 0):.4f}")
        if progress_callback:
            progress_callback(sweep_name="k", value_idx=i, total_values=len(k_values))

    logger.info(f"Saved k sweep to {path} ({len(rows)} new rows)")
    return pd.DataFrame(rows) if rows else pd.read_csv(path)


def sweep_tau(dataset, embedder, llm_client, seeds, output_dir, batch_size=200, llm_timeout=600, max_event_words=None, tau_values=None, progress_callback=None):
    """Sweep cosine threshold tau (Fig. 6b)."""
    if tau_values is None:
        tau_values = TAU_VALUES_DEFAULT
    path = output_dir / f"sensitivity_tau_{dataset.short_name}.csv"
    done = _load_completed(path, "label")
    rows = []
    for i, tau in enumerate(tau_values):
        for seed in seeds:
            label = f"tau={tau}"
            if label in done:
                logger.info(f"tau sweep: tau={tau}, seed={seed} — already done, skipping")
                continue
            logger.info(f"tau sweep: tau={tau}, seed={seed}")
            llm_client.reset_stats()
            row = run_one(
                dataset, embedder, llm_client,
                k=DEFAULTS["k"], tau=tau, kappa=DEFAULTS["kappa"],
                seed=seed, label=label,
                batch_size=batch_size, llm_timeout=llm_timeout, max_event_words=max_event_words,
            )
            rows.append(row)
            _append_row(row, path)
            logger.info(f"  tau={tau} seed={seed}: F1={row.get('f1', 0):.4f}")
        if progress_callback:
            progress_callback(sweep_name="tau", value_idx=i, total_values=len(tau_values))

    logger.info(f"Saved tau sweep to {path} ({len(rows)} new rows)")
    return pd.DataFrame(rows) if rows else pd.read_csv(path)


def sweep_kappa(dataset, embedder, llm_client, seeds, output_dir, batch_size=200, llm_timeout=600, max_event_words=None, kappa_values=None, progress_callback=None):
    """Sweep top-K matches kappa (Fig. 6c)."""
    if kappa_values is None:
        kappa_values = KAPPA_VALUES_DEFAULT
    path = output_dir / f"sensitivity_kappa_{dataset.short_name}.csv"
    done = _load_completed(path, "label")
    rows = []
    for i, kappa in enumerate(kappa_values):
        for seed in seeds:
            label = f"kappa={kappa}"
            if label in done:
                logger.info(f"kappa sweep: kappa={kappa}, seed={seed} — already done, skipping")
                continue
            logger.info(f"kappa sweep: kappa={kappa}, seed={seed}")
            llm_client.reset_stats()
            row = run_one(
                dataset, embedder, llm_client,
                k=DEFAULTS["k"], tau=DEFAULTS["tau"], kappa=kappa,
                seed=seed, label=label,
                batch_size=batch_size, llm_timeout=llm_timeout, max_event_words=max_event_words,
            )
            rows.append(row)
            _append_row(row, path)
            logger.info(f"  kappa={kappa} seed={seed}: F1={row.get('f1', 0):.4f}")
        if progress_callback:
            progress_callback(sweep_name="kappa", value_idx=i, total_values=len(kappa_values))

    logger.info(f"Saved kappa sweep to {path} ({len(rows)} new rows)")
    return pd.DataFrame(rows) if rows else pd.read_csv(path)


def sweep_embedding(dataset, llm_client, seeds, output_dir, cache_dir, batch_size=200, llm_timeout=600, max_event_words=None, progress_callback=None):
    """Sweep embedding models (Fig. 6d)."""
    path = output_dir / f"sensitivity_embedding_{dataset.short_name}.csv"
    done = _load_completed(path, "label")
    rows = []
    embedding_items = list(EMBEDDING_MODELS.items())
    for i, (model_name, model_path) in enumerate(embedding_items):
        logger.info(f"Embedding sweep: {model_name}")
        embedder = EmbeddingModel(model_path, cache_dir=cache_dir)
        for seed in seeds:
            if model_name in done:
                logger.info(f"  {model_name} seed={seed} — already done, skipping")
                continue
            llm_client.reset_stats()
            row = run_one(
                dataset, embedder, llm_client,
                k=DEFAULTS["k"], tau=DEFAULTS["tau"], kappa=DEFAULTS["kappa"],
                seed=seed, label=model_name,
                batch_size=batch_size, llm_timeout=llm_timeout, max_event_words=max_event_words,
            )
            row["embedding_model"] = model_name
            rows.append(row)
            _append_row(row, path)
            logger.info(f"  {model_name} seed={seed}: F1={row.get('f1', 0):.4f}")
        if progress_callback:
            progress_callback(sweep_name="embedding", value_idx=i, total_values=len(embedding_items))

    logger.info(f"Saved embedding sweep to {path} ({len(rows)} new rows)")
    return pd.DataFrame(rows) if rows else pd.read_csv(path)


def run_sensitivity_stage(
    entry: RunEntry,
    profile: dict,
    llm_client: LLMClient | None = None,
    embedding_model: EmbeddingModel | None = None,
    progress_tracker=None,
) -> None:
    """Run a single sensitivity sweep entry from the manifest.

    Thin wrapper that resolves the sweep type from entry.config
    (e.g., "sweep_k", "sweep_tau") and calls the corresponding
    sweep function.

    Args:
        entry: RunEntry describing the sensitivity run.
        profile: Loaded mode profile from params.yaml.
        llm_client: Pre-initialised LLM client.
        embedding_model: Pre-initialised embedding model.
        progress_tracker: Optional progress callback (unused for now).
    """
    overrides = get_model_overrides(entry.model_key)
    batch_size = overrides["batch_size"] or 200
    llm_timeout = overrides["llm_timeout"] or 600

    dataset = load_dataset_by_name(
        entry.dataset,
        cache_dir="data",
        max_events=entry.max_events,
    )

    output_dir = Path(entry.result_file).parent

    p_sens = profile.get("sensitivity", {})
    seeds = [entry.seed]

    # Determine which sweep to run from entry.config (e.g., "sweep_k")
    sweep_param = entry.config.removeprefix("sweep_")

    if sweep_param == "k":
        k_values = p_sens.get("k_values", K_VALUES_DEFAULT)
        sweep_k(
            dataset, embedding_model, llm_client, seeds, output_dir,
            batch_size=batch_size, llm_timeout=llm_timeout,
            max_event_words=entry.max_event_words, k_values=k_values,
        )
    elif sweep_param == "tau":
        tau_values = p_sens.get("tau_values", TAU_VALUES_DEFAULT)
        sweep_tau(
            dataset, embedding_model, llm_client, seeds, output_dir,
            batch_size=batch_size, llm_timeout=llm_timeout,
            max_event_words=entry.max_event_words, tau_values=tau_values,
        )
    elif sweep_param == "kappa":
        kappa_values = p_sens.get("kappa_values", KAPPA_VALUES_DEFAULT)
        sweep_kappa(
            dataset, embedding_model, llm_client, seeds, output_dir,
            batch_size=batch_size, llm_timeout=llm_timeout,
            max_event_words=entry.max_event_words, kappa_values=kappa_values,
        )
    elif sweep_param == "embedding":
        sweep_embedding(
            dataset, llm_client, seeds, output_dir, "data",
            batch_size=batch_size, llm_timeout=llm_timeout,
            max_event_words=entry.max_event_words,
        )
    else:
        raise ValueError(f"Unknown sensitivity sweep parameter: {sweep_param}")


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
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override max_batch_events in RouterConfig (default: 50 for ollama, 200 otherwise)")
    parser.add_argument("--llm-timeout", type=int, default=None,
                        help="Override llm_timeout in RouterConfig (default: 600 for ollama, 600 otherwise)")
    parser.add_argument("--max-event-words", type=int, default=None,
                        help="Truncate event text to N words in LLM prompts (default: no truncation)")
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

    # Resolve sweep ranges from profile
    if profile:
        p_sens = profile.get("sensitivity", {})
        k_values = p_sens.get("k_values", K_VALUES_DEFAULT)
        tau_values = p_sens.get("tau_values", TAU_VALUES_DEFAULT)
        kappa_values = p_sens.get("kappa_values", KAPPA_VALUES_DEFAULT)
        if args.max_events is None:
            args.max_events = profile.get("max_events")
        if args.max_event_words is None:
            args.max_event_words = profile.get("max_event_words")
        if args.output_dir == "results":  # argparse default
            args.output_dir = str(output_dir_for_mode("results", profile["mode"], "sensitivity"))
    else:
        k_values = K_VALUES_DEFAULT
        tau_values = TAU_VALUES_DEFAULT
        kappa_values = KAPPA_VALUES_DEFAULT

    seeds = [int(s) for s in args.seeds.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect ollama and set defaults accordingly
    is_ollama = "ollama" in args.llm_model.lower()
    batch_size = args.batch_size if args.batch_size is not None else (50 if is_ollama else 200)
    llm_timeout = args.llm_timeout if args.llm_timeout is not None else (600 if is_ollama else 600)
    if is_ollama:
        logger.info(f"Ollama model detected: using batch_size={batch_size}, llm_timeout={llm_timeout}s")

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

    max_event_words = args.max_event_words
    sweeps = args.sweep.lower().split(",") if args.sweep.lower() != "all" else ["k", "tau", "kappa", "embedding"]

    # Progress tracking
    model_key = args.llm_model.split("/")[-1].split(":")[0]
    if "qwen" in model_key.lower():
        model_key = "qwen7b" if "7b" in args.llm_model else "qwen32b"
    elif "haiku" in model_key.lower():
        model_key = "haiku"
    elif "sonnet" in model_key.lower():
        model_key = "sonnet"
    mode = profile["mode"] if profile else "unknown"
    stage_name = f"sensitivity-{args.dataset}-{model_key}"
    tracker = ProgressTracker(stage_name, mode=mode)

    # Count total items across all sweeps
    total_items = 0
    for sw in sweeps:
        if sw == "k": total_items += len(k_values)
        elif sw == "tau": total_items += len(tau_values)
        elif sw == "kappa": total_items += len(kappa_values)
        elif sw == "embedding": total_items += 4  # 4 embedding models
    items_done = 0

    def on_sweep_progress(sweep_name, value_idx, total_values):
        nonlocal items_done
        # items_done tracks cumulative across all sweeps
        tracker.update(
            current_task=f"{sweep_name} sweep ({value_idx+1}/{total_values})",
            items_done=items_done + value_idx + 1,
            items_total=total_items,
            force=True,
        )

    for sweep in sweeps:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {sweep} sweep on {dataset.short_name}")
        logger.info(f"{'='*60}")

        if sweep == "k":
            sweep_k(dataset, embedder, llm_client, seeds, output_dir, batch_size=batch_size, llm_timeout=llm_timeout, max_event_words=max_event_words, k_values=k_values, progress_callback=on_sweep_progress)
            items_done += len(k_values)
        elif sweep == "tau":
            sweep_tau(dataset, embedder, llm_client, seeds, output_dir, batch_size=batch_size, llm_timeout=llm_timeout, max_event_words=max_event_words, tau_values=tau_values, progress_callback=on_sweep_progress)
            items_done += len(tau_values)
        elif sweep == "kappa":
            sweep_kappa(dataset, embedder, llm_client, seeds, output_dir, batch_size=batch_size, llm_timeout=llm_timeout, max_event_words=max_event_words, kappa_values=kappa_values, progress_callback=on_sweep_progress)
            items_done += len(kappa_values)
        elif sweep == "embedding":
            sweep_embedding(dataset, llm_client, seeds, output_dir, args.cache_dir, batch_size=batch_size, llm_timeout=llm_timeout, max_event_words=max_event_words, progress_callback=on_sweep_progress)
            items_done += 4
        else:
            logger.warning(f"Unknown sweep: {sweep}")

    tracker.finish()


if __name__ == "__main__":
    main()
