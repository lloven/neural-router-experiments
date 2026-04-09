"""Tests for context-window truncation (crossover experiment support).

When max_context_tokens is set on RouterConfig, the subscription list
fed to each LLM call is truncated to fit within the token budget.
Subscriptions that don't fit are silently dropped.

The crossover experiment relies on this: A0 (raw) hits the window
limit at fewer subscriptions than A4 (compressed), demonstrating
the crossover point where the pipeline becomes beneficial.

TDD RED phase: all tests written before implementation.
"""

import pytest

from src.data import Subscription, Event, Dataset
from src.router import (
    RouterConfig,
    NeuralRouter,
    ABLATION_CONFIGS,
    Cluster,
)
from src.llm import DryRunLLMClient
from src.embeddings import EmbeddingModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_subscriptions(n: int, desc_words: int = 20) -> list[Subscription]:
    """Create n subscriptions with deterministic descriptions."""
    subs = []
    for i in range(n):
        desc = " ".join(f"word{i}_{j}" for j in range(desc_words))
        subs.append(Subscription(id=f"sub_{i}", name=f"Sub {i}", description=desc))
    return subs


def _make_events(n: int, gt_ids: list[str] | None = None) -> list[Event]:
    """Create n events with short text."""
    events = []
    for i in range(n):
        events.append(Event(
            id=f"evt_{i}",
            text=f"This is test event number {i} about some topic.",
            ground_truth=gt_ids or ["sub_0"],
        ))
    return events


def _make_dataset(n_subs: int = 50, n_events: int = 10, desc_words: int = 20) -> Dataset:
    subs = _make_subscriptions(n_subs, desc_words=desc_words)
    events = _make_events(n_events, gt_ids=[subs[0].id])
    return Dataset(
        name="Test Dataset",
        short_name="T1",
        events=events,
        subscriptions=subs,
    )


# ---------------------------------------------------------------------------
# 1. RouterConfig accepts max_context_tokens
# ---------------------------------------------------------------------------

def test_router_config_has_max_context_tokens_field():
    """RouterConfig should accept a max_context_tokens parameter (None = no limit)."""
    config = RouterConfig(max_context_tokens=4096)
    assert config.max_context_tokens == 4096


def test_router_config_default_max_context_tokens_is_none():
    """Default max_context_tokens should be None (unlimited)."""
    config = RouterConfig()
    assert config.max_context_tokens is None


# ---------------------------------------------------------------------------
# 2. truncate_subscriptions_to_budget utility
# ---------------------------------------------------------------------------

def test_truncate_subscriptions_returns_all_when_budget_unlimited():
    """When budget is None or very large, all subscriptions should be retained."""
    from src.router import truncate_subscriptions_to_budget

    subs = _make_subscriptions(10, desc_words=5)
    result = truncate_subscriptions_to_budget(subs, budget_tokens=None, t_inst=200, t_resp=500)
    assert len(result) == 10


def test_truncate_subscriptions_drops_when_budget_tight():
    """When the budget is tight, subscriptions should be dropped to fit."""
    from src.router import truncate_subscriptions_to_budget

    # Create 50 subscriptions with ~20 words each (~5 tokens/word = ~100 tokens/sub)
    subs = _make_subscriptions(50, desc_words=20)
    # Very tight budget: only room for a few subscriptions
    # t_inst=200, t_resp=500, so available = budget - 200 - 500 = budget - 700
    # Each sub is roughly "- [sub_X] word..." ~25 tokens (conservative estimate)
    # With budget=1500, available=800, ~800/25 = ~32 subs max
    result = truncate_subscriptions_to_budget(subs, budget_tokens=1500, t_inst=200, t_resp=500)
    assert len(result) < 50, "Should drop some subscriptions"
    assert len(result) > 0, "Should retain at least some subscriptions"


def test_truncate_subscriptions_preserves_order():
    """Truncation should preserve original subscription order (first N that fit)."""
    from src.router import truncate_subscriptions_to_budget

    subs = _make_subscriptions(20, desc_words=10)
    result = truncate_subscriptions_to_budget(subs, budget_tokens=1000, t_inst=200, t_resp=500)
    # Result should be a prefix of the original list
    for i, s in enumerate(result):
        assert s.id == subs[i].id


def test_truncate_subscriptions_empty_when_budget_too_small():
    """When budget is so small nothing fits, return empty list."""
    from src.router import truncate_subscriptions_to_budget

    subs = _make_subscriptions(10, desc_words=20)
    result = truncate_subscriptions_to_budget(subs, budget_tokens=100, t_inst=200, t_resp=500)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# 3. NeuralRouter respects max_context_tokens in matching
# ---------------------------------------------------------------------------

def test_router_with_context_limit_drops_subscriptions():
    """When max_context_tokens is set, the router should use fewer subscriptions
    in the LLM prompt than the full set."""
    # Create a dataset with many subscriptions
    subs = _make_subscriptions(100, desc_words=30)
    events = _make_events(5, gt_ids=[subs[0].id])
    dataset = Dataset(
        name="Test", short_name="T1",
        events=events, subscriptions=subs,
    )

    llm = DryRunLLMClient(model="test-model")
    embedder = EmbeddingModel("all-MiniLM-L6-v2")

    # A0 with very small context window -- should silently drop subscriptions
    config = RouterConfig(
        **{
            **ABLATION_CONFIGS["A0"].__dict__,
            "max_context_tokens": 2000,
            "llm_model": "test-model",
            "seed": 42,
        }
    )
    router = NeuralRouter(config=config, llm_client=llm, embedding_model=embedder)
    router.optimize_subscriptions(dataset.subscriptions)
    matches = router.match_events(dataset.events)

    # Should complete without error and produce results
    assert len(matches) == len(events)
    # The router should have tracked the number of subscriptions used
    assert router.stats.num_subscriptions_effective < 100


# ---------------------------------------------------------------------------
# 4. Compressed (A4) fits more subscriptions than raw (A0) under same budget
# ---------------------------------------------------------------------------

def test_compressed_fits_more_subscriptions_than_raw():
    """Under the same context budget, A4 (compressed) should retain more
    original subscriptions than A0 (raw), because compression reduces
    the per-subscription token footprint."""
    from src.router import truncate_subscriptions_to_budget

    # Same number of subscriptions, but compressed have shorter descriptions
    raw_subs = _make_subscriptions(50, desc_words=30)  # ~8 tokens each
    compressed_subs = _make_subscriptions(50, desc_words=5)  # ~2 tokens each (after C&M)

    budget = 500  # tight enough that raw can't fit all 50
    raw_fit = truncate_subscriptions_to_budget(raw_subs, budget_tokens=budget, t_inst=50, t_resp=100)
    comp_fit = truncate_subscriptions_to_budget(compressed_subs, budget_tokens=budget, t_inst=50, t_resp=100)

    # Compressed (shorter descriptions) should fit more than raw (longer descriptions)
    assert len(comp_fit) >= len(raw_fit), (
        f"Compressed ({len(comp_fit)}) should fit at least as many as raw ({len(raw_fit)})"
    )


# ---------------------------------------------------------------------------
# 5. RouterStats tracks effective subscription count
# ---------------------------------------------------------------------------

def test_router_stats_tracks_effective_subscriptions():
    """RouterStats should have a num_subscriptions_effective field tracking
    how many subscriptions were actually used in LLM calls after truncation."""
    from src.router import RouterStats
    stats = RouterStats()
    assert hasattr(stats, 'num_subscriptions_effective')
    assert stats.num_subscriptions_effective == 0
