"""Evaluation metrics and statistical tests for Neural Router experiments.

Implements the evaluation protocol from Section 4.7 of the paper:

  Matching accuracy (per-event, macro-averaged):
    - Precision, Recall, F1-score, False Positive Rate (FPR)

  System metrics:
    - LLM invocation count, wall-clock latency, throughput

  Cost metrics:
    - Compression ratio (rho), prompt/response tokens, monetary cost per 1k events

  Statistical analysis:
    - 95% confidence intervals via Student's t-distribution
    - Wilcoxon signed-rank test for pairwise configuration comparison

Key functions:
  evaluate_matches()      -- compute all metrics for one (config, dataset, seed)
  aggregate_seeds()       -- combine per-seed results into cross-seed CIs
  pairwise_significance() -- Wilcoxon test between two configurations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats

from .data import Dataset, Event
from .router import MatchResult, RouterStats

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """A single metric aggregated over events, with 95% confidence interval."""
    name: str
    mean: float
    ci_lower: float
    ci_upper: float
    std: float
    n: int

    @property
    def ci_width(self) -> float:
        return self.ci_upper - self.ci_lower

    def __str__(self) -> str:
        return f"{self.name}: {self.mean:.4f} ± {self.ci_width/2:.4f}"


@dataclass
class EvaluationResult:
    """Complete evaluation result for one (config, dataset, seed) triple.

    Contains macro-averaged accuracy metrics with CIs, plus system and cost
    metrics. Use ``summary_row()`` to export as a flat dict for DataFrames.
    """
    config_name: str
    dataset_name: str
    seed: int
    precision: MetricResult
    recall: MetricResult
    f1: MetricResult
    fpr: MetricResult
    invocations: int = 0
    compression_ratio: float = 1.0
    latency_s: float = 0.0
    cost_per_1k: float = 0.0
    tokens_prompt: int = 0
    tokens_response: int = 0
    per_event_metrics: Optional[dict] = None

    def summary_row(self) -> dict:
        """Return a dict suitable for a DataFrame row."""
        return {
            "config": self.config_name,
            "dataset": self.dataset_name,
            "seed": self.seed,
            "precision": self.precision.mean,
            "recall": self.recall.mean,
            "f1": self.f1.mean,
            "fpr": self.fpr.mean,
            "invocations": self.invocations,
            "compression_ratio": self.compression_ratio,
            "latency_s": self.latency_s,
            "tokens_prompt": self.tokens_prompt,
            "tokens_response": self.tokens_response,
            "cost_per_1k": self.cost_per_1k,
        }


def evaluate_matches(
    matches: list[MatchResult],
    dataset: Dataset,
    config_name: str = "",
    seed: int = 0,
    router_stats: Optional[RouterStats] = None,
) -> EvaluationResult:
    """Evaluate matching results against ground truth.

    Computes per-event precision, recall, F1, and FPR, then macro-averages
    with 95% confidence intervals. Handles compound subscription IDs
    produced by CoverAndMerge (e.g., "arts+music" expands to both).

    Args:
        matches: One MatchResult per event from the router or baseline.
        dataset: The dataset (provides ground truth and subscription universe).
        config_name: Label for this configuration (e.g., "A3").
        seed: Random seed used for this run.
        router_stats: Optional RouterStats for system/cost metrics.

    Returns:
        EvaluationResult with all accuracy, system, and cost metrics.
    """
    # Build ground truth lookup
    gt_lookup = {e.id: set(e.ground_truth) for e in dataset.events}

    # All subscription IDs for FPR computation
    all_sub_ids = {s.id for s in dataset.subscriptions}

    precisions = []
    recalls = []
    f1s = []
    fprs = []

    for result in matches:
        gt = gt_lookup.get(result.event_id, set())
        predicted = set(result.matched_subscription_ids)

        # Expand compound IDs from CoverAndMerge (e.g., "arts+music" -> {"arts", "music"})
        expanded_predicted = set()
        for pid in predicted:
            if "+" in pid:
                expanded_predicted.update(pid.split("+"))
            else:
                expanded_predicted.add(pid)
        predicted = expanded_predicted

        # Standard binary classification metrics per event
        tp = len(predicted & gt)
        fp = len(predicted - gt)
        fn = len(gt - predicted)
        tn = len(all_sub_ids - predicted - gt)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        fprs.append(fpr_val)

    # Compute aggregated metrics with CIs
    precision_result = _compute_ci(precisions, "precision")
    recall_result = _compute_ci(recalls, "recall")
    f1_result = _compute_ci(f1s, "f1")
    fpr_result = _compute_ci(fprs, "fpr")

    # System stats
    invocations = router_stats.llm_invocations if router_stats else 0
    compression = router_stats.compression_ratio if router_stats else 1.0
    latency = router_stats.total_time_s if router_stats else 0.0
    prompt_tokens = router_stats.total_prompt_tokens if router_stats else 0
    resp_tokens = router_stats.total_response_tokens if router_stats else 0

    # Cost: GPT-4o-mini pricing
    cost = (prompt_tokens * 0.15 + resp_tokens * 0.60) / 1_000_000
    cost_per_1k = cost * 1000 / max(len(matches), 1)

    return EvaluationResult(
        config_name=config_name,
        dataset_name=dataset.short_name,
        seed=seed,
        precision=precision_result,
        recall=recall_result,
        f1=f1_result,
        fpr=fpr_result,
        invocations=invocations,
        compression_ratio=compression,
        latency_s=latency,
        cost_per_1k=cost_per_1k,
        tokens_prompt=prompt_tokens,
        tokens_response=resp_tokens,
        per_event_metrics={
            "precisions": precisions,
            "recalls": recalls,
            "f1s": f1s,
            "fprs": fprs,
        },
    )


def _compute_ci(
    values: list[float],
    name: str,
    confidence: float = 0.95,
) -> MetricResult:
    """Compute mean and confidence interval via Student's t-distribution.

    Args:
        values: Per-event (or per-seed) metric values.
        name: Metric name for the result label.
        confidence: Confidence level (default 0.95 for 95% CI).

    Returns:
        MetricResult with mean, CI bounds, std, and sample size.
    """
    arr = np.array(values)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0

    if n > 1:
        se = std / np.sqrt(n)  # standard error of the mean
        t_val = stats.t.ppf((1 + confidence) / 2, n - 1)  # two-tailed t critical value
        ci_lower = mean - t_val * se
        ci_upper = mean + t_val * se
    else:
        ci_lower = mean
        ci_upper = mean

    return MetricResult(
        name=name,
        mean=mean,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        std=std,
        n=n,
    )


def aggregate_seeds(
    results: list[EvaluationResult],
) -> EvaluationResult:
    """Aggregate results across multiple random seeds.

    Takes a list of per-seed EvaluationResults and produces one result
    with CIs computed across seeds (not across events). This is the
    outer aggregation used for the final reported numbers in the paper.

    Args:
        results: List of EvaluationResults, one per seed, for the same
            (config, dataset) pair.

    Returns:
        A single EvaluationResult with seed=-1 and cross-seed CIs.

    Raises:
        ValueError: If the input list is empty.
    """
    if not results:
        raise ValueError("No results to aggregate")

    # Collect per-seed means
    precisions = [r.precision.mean for r in results]
    recalls = [r.recall.mean for r in results]
    f1s = [r.f1.mean for r in results]
    fprs = [r.fpr.mean for r in results]
    invocations = [r.invocations for r in results]
    latencies = [r.latency_s for r in results]
    costs = [r.cost_per_1k for r in results]
    compressions = [r.compression_ratio for r in results]

    return EvaluationResult(
        config_name=results[0].config_name,
        dataset_name=results[0].dataset_name,
        seed=-1,  # aggregated
        precision=_compute_ci(precisions, "precision"),
        recall=_compute_ci(recalls, "recall"),
        f1=_compute_ci(f1s, "f1"),
        fpr=_compute_ci(fprs, "fpr"),
        invocations=int(np.mean(invocations)),
        compression_ratio=float(np.mean(compressions)),
        latency_s=float(np.mean(latencies)),
        cost_per_1k=float(np.mean(costs)),
    )


def pairwise_significance(
    results_a: list[EvaluationResult],
    results_b: list[EvaluationResult],
    metric: str = "f1",
    alpha: float = 0.05,
) -> dict:
    """Wilcoxon signed-rank test between two configurations.

    Compares per-seed metric means as paired observations. Requires at
    least 5 seeds for a meaningful test (logs a warning otherwise).

    Args:
        results_a: Per-seed results for configuration A.
        results_b: Per-seed results for configuration B (same seeds).
        metric: Metric attribute name to compare (e.g., "f1", "precision").
        alpha: Significance threshold (default 0.05).

    Returns:
        Dict with test statistic, p-value, significance boolean, and metadata.
    """
    vals_a = [getattr(r, metric).mean for r in results_a]
    vals_b = [getattr(r, metric).mean for r in results_b]

    if len(vals_a) < 5 or len(vals_b) < 5:
        logger.warning("Fewer than 5 seeds; significance test may be unreliable")

    try:
        stat, p_value = stats.wilcoxon(vals_a, vals_b)
    except ValueError:
        # All differences are zero
        stat, p_value = 0.0, 1.0

    return {
        "metric": metric,
        "config_a": results_a[0].config_name if results_a else "",
        "config_b": results_b[0].config_name if results_b else "",
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": p_value < alpha,
        "alpha": alpha,
    }
