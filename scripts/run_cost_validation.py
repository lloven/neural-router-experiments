#!/usr/bin/env python3
"""Cost-model validation runner: re-runs a small (config × k) grid with the
new per-cluster logging in place so fig:cost-validation can plot honest
predicted-vs-measured I per cluster (figures justify load-bearing claims).

Each row of the output CSV carries:
- aggregate metrics (f1, invocations, latency_s, tokens_*, cost_per_1k)
- per-cluster decomposition: per_cluster_invocations, per_cluster_events,
  per_cluster_active_subs (JSON-encoded lists), one slot per cluster that
  processed at least one event.

Default grid: A0,A3 × k∈{1,5,19} × seed=42 — small enough for one Mahti
gputest job (~30 min wall-clock, ~4 BU at 8 BU/GPU-h).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data import load_dataset_by_name
from src.embeddings import EmbeddingModel
from src.evaluation import evaluate_matches
from src.llm import DryRunLLMClient, LLMClient
from src.router import ABLATION_CONFIGS, NeuralRouter, RouterConfig

logger = logging.getLogger(__name__)


def _build_llm(model: str) -> LLMClient:
    """Return an LLM client. 'dry-run' = DryRunLLMClient for unit smoke;
    anything else (e.g. 'ollama/qwen2.5:7b') goes through the real LLMClient."""
    if model in ("dry-run", "dry_run", "dummy"):
        return DryRunLLMClient(model=model)
    return LLMClient(model=model)


def run_one(*, dataset, config_name: str, k: int, seed: int,
            llm_client: LLMClient, embedder: EmbeddingModel) -> dict:
    """Run one (config, k, seed) cell; return summary_row + (k, config, seed)."""
    base = ABLATION_CONFIGS[config_name]
    overrides = {"k": k, "seed": seed, "llm_model": llm_client.model}
    if "ollama" in llm_client.model.lower():
        overrides["max_parallel"] = 1
        overrides["use_async"] = False
    config = RouterConfig(**{**base.__dict__, **overrides})

    router = NeuralRouter(config=config, llm_client=llm_client, embedding_model=embedder)
    router.optimize_subscriptions(dataset.subscriptions)
    matches = router.match_events(dataset.events)
    result = evaluate_matches(
        matches=matches, dataset=dataset,
        config_name=config_name, seed=seed,
        router_stats=router.stats,
    )
    row = result.summary_row()
    row["k"] = k
    return row


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["D1", "D2", "D3"])
    p.add_argument("--configs", default="A0,A3",
                   help="Comma-separated ablation config names (default: A0,A3)")
    p.add_argument("--k-values", default="1,5,19",
                   help="Comma-separated k values (default: 1,5,19)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-events", type=int, default=None,
                   help="Truncate dataset to this many events (smoke testing).")
    p.add_argument("--llm-model", default="dry-run",
                   help="LLM model identifier; 'dry-run' for unit smoke.")
    p.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"cost_validation_{args.dataset}.csv"

    dataset = load_dataset_by_name(args.dataset, max_events=args.max_events)
    llm = _build_llm(args.llm_model)
    embedder = EmbeddingModel(args.embedding_model)

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    ks = [int(k.strip()) for k in args.k_values.split(",") if k.strip()]

    # Write each row as it completes so a SLURM timeout does not lose all
    # already-completed cells. This was the actual failure mode of job
    # 6615989 (cost-val full v1) — 5 of 6 cells finished, all data lost.
    rows = []
    write_header = not out_csv.exists() or out_csv.stat().st_size == 0
    for cfg in configs:
        for k in ks:
            logger.info("Running %s k=%d seed=%d on %s", cfg, k, args.seed, args.dataset)
            row = run_one(
                dataset=dataset, config_name=cfg, k=k, seed=args.seed,
                llm_client=llm, embedder=embedder,
            )
            rows.append(row)
            pd.DataFrame([row]).to_csv(out_csv, mode="a", header=write_header, index=False)
            write_header = False
            logger.info("Appended row for %s k=%d to %s (cumulative %d rows)",
                        cfg, k, out_csv, len(rows))
    logger.info("Wrote %d rows to %s", len(rows), out_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
