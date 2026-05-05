"""Per-cluster invocation logging in RouterStats.

Required to honestly populate fig:cost-validation. The cost model claim
(Eq. 4) is per-cluster: I = Σ_c ⌈m_c / b_max(c)⌉. Aggregate I is
insufficient to validate this; we need per-cluster I_c and m_c.

Per L60: figure data gap → fix the experiment (logging), not the figure.
"""
from __future__ import annotations

import pytest

from src.router import RouterStats


def test_router_stats_records_per_cluster_invocations():
    """RouterStats must expose lists indexed by cluster: invocations,
    events processed, active subscription count after CoverAndMerge.
    """
    s = RouterStats()
    assert s.per_cluster_invocations == []
    assert s.per_cluster_events == []
    assert s.per_cluster_active_subs == []


def test_per_cluster_invocations_sum_to_total():
    """Per-cluster I_c values must sum to aggregate llm_invocations.
    This is the cost-model identity Σ_c I_c = I_total. Run a tiny
    in-memory matching pass against the dry-run LLM client and verify.
    """
    from src.router import NeuralRouter, RouterConfig
    from src.llm import DryRunLLMClient
    from src.embeddings import EmbeddingModel
    from src.data import Subscription, Event

    subs = [Subscription(id=f"s{i}", name=f"S{i}", description=f"d{i}") for i in range(6)]
    events = [Event(id=f"e{i}", text=f"event text {i}", ground_truth=[]) for i in range(8)]

    config = RouterConfig(k=2, use_clustering=True, use_cover_merge=False)
    router = NeuralRouter(
        config=config,
        llm_client=DryRunLLMClient(model="test"),
        embedding_model=EmbeddingModel("all-MiniLM-L6-v2"),
    )
    router.optimize_subscriptions(subs)
    router.match_events(events)

    s = router.stats
    assert s.num_clusters == 2, f"expected 2 clusters, got {s.num_clusters}"
    # The list must have one slot per cluster that actually processed events.
    # Empty-queue clusters do not invoke the LLM and contribute 0 (or absent).
    assert len(s.per_cluster_invocations) <= s.num_clusters
    assert sum(s.per_cluster_invocations) == s.llm_invocations, \
        f"per-cluster I_c must sum to total: {s.per_cluster_invocations} vs {s.llm_invocations}"
    # Per-cluster events sum equals all routed events
    assert sum(s.per_cluster_events) == len(events)
    # Per-cluster active-sub counts: every entry is positive (clusters that
    # processed events have at least one subscription)
    assert all(n > 0 for n in s.per_cluster_active_subs)
