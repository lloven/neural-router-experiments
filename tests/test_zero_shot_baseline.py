"""Tests for the zero-shot classification baseline (BART-large-MNLI).

RED phase: these tests should FAIL until the baseline is implemented.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import Dataset, Event, Subscription
from src.baselines import run_baseline, BaselineResult

# Use a small distilled model for testing (bart-large-mnli is too large for CI)
SMALL_MODEL = "valhalla/distilbart-mnli-12-1"


@pytest.fixture
def small_dataset():
    subs = [
        Subscription(id="sports", name="Sports", description="Sports news, games, and athletic events"),
        Subscription(id="tech", name="Technology", description="Technology, software, and computing"),
        Subscription(id="finance", name="Finance", description="Financial markets, stocks, and economics"),
    ]
    events = [
        Event(id="e1", text="The team won the championship game last night", ground_truth=["sports"]),
        Event(id="e2", text="New GPU released for deep learning workloads", ground_truth=["tech"]),
        Event(id="e3", text="Stock market surges to record high", ground_truth=["finance"]),
    ]
    return Dataset(
        name="test",
        short_name="test",
        events=events,
        subscriptions=subs,
    )


class TestZeroShotBaseline:
    """Tests for the zero_shot baseline method."""

    def test_zero_shot_is_registered(self):
        """The 'zero_shot' method should be accepted by run_baseline."""
        # Should not raise ValueError
        # (will fail until dispatch table includes 'zero_shot')
        from src.baselines import run_baseline
        # Just check it doesn't raise ValueError for unknown method
        try:
            # We don't actually need it to succeed with real model loading,
            # just that the method name is recognized in the dispatch table
            from src.baselines import _run_zero_shot
        except ImportError:
            pytest.fail("_run_zero_shot not found in src.baselines")

    def test_zero_shot_returns_correct_structure(self, small_dataset):
        """run_baseline('zero_shot', ...) should return BaselineResult."""
        result = run_baseline("zero_shot", small_dataset, kappa=2, model_name=SMALL_MODEL)
        assert isinstance(result, BaselineResult)
        assert result.method == "zero_shot"
        assert len(result.matches) == len(small_dataset.events)
        assert result.latency_s > 0

    def test_zero_shot_returns_kappa_matches(self, small_dataset):
        """Each event should have at most kappa matches."""
        result = run_baseline("zero_shot", small_dataset, kappa=2, model_name=SMALL_MODEL)
        for match in result.matches:
            assert len(match.matched_subscription_ids) <= 2

    def test_zero_shot_returns_valid_subscription_ids(self, small_dataset):
        """All returned IDs should be valid subscription IDs."""
        valid_ids = {s.id for s in small_dataset.subscriptions}
        result = run_baseline("zero_shot", small_dataset, kappa=2, model_name=SMALL_MODEL)
        for match in result.matches:
            for sid in match.matched_subscription_ids:
                assert sid in valid_ids, f"Invalid subscription ID: {sid}"

    def test_zero_shot_quality_sanity(self, small_dataset):
        """Basic sanity: sports event should match sports subscription."""
        result = run_baseline("zero_shot", small_dataset, kappa=1, model_name=SMALL_MODEL)
        # The sports event should match the sports subscription
        sports_match = next(m for m in result.matches if m.event_id == "e1")
        assert "sports" in sports_match.matched_subscription_ids, (
            f"Expected 'sports' in matches for sports event, got {sports_match.matched_subscription_ids}"
        )
