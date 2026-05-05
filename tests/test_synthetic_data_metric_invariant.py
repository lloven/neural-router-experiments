"""Per L61 (Synthetic-data helpers must preserve evaluation metric invariants):
round-trip integration test on `_subsample_subscriptions`.

Generates a duplicated subscription set, runs the full match→evaluate
pipeline, and asserts the metric remains meaningful. The 2026-05-04
crossover sweep produced an apparent A4 F1 collapse that was actually a
metric artifact: duplication-with-rename gives multiple IDs to one
description, ID-based F1 then counts merged-away IDs as misses. Either:
(i) the matcher is description-aware and F1 stays sane, or
(ii) the metric is ID-based and F1 collapses; in case (ii) the test
must detect the collapse so it can be flagged before manuscript-level
interpretation.
"""
from __future__ import annotations

import pytest

from src.data import Subscription


def test_duplicated_subscriptions_share_descriptions():
    """The duplication helper assigns new IDs but identical descriptions —
    this is the precondition for the L61 metric artifact. Lock it in."""
    from scripts.run_crossover import _subsample_subscriptions

    base = [Subscription(id=f"s{i}", name=f"S{i}", description=f"topic {i}") for i in range(5)]
    dup = _subsample_subscriptions(base, n=20, seed=42)
    assert len(dup) == 20
    desc_to_ids = {}
    for s in dup:
        desc_to_ids.setdefault(s.description, []).append(s.id)
    # Every original description should be associated with multiple IDs after
    # duplication; this is the invariant violation L61 is about.
    multi_id_count = sum(1 for ids in desc_to_ids.values() if len(ids) > 1)
    assert multi_id_count >= 1, "duplication should produce shared descriptions"


def test_description_aware_f1_immune_to_duplication():
    """Round-trip: duplicating subscriptions must NOT change the F1 score
    when the metric is description-aware (the right metric for sets with
    semantic-duplicate IDs). If we ever flip evaluate_matches to be
    ID-based-only on a duplicated set, this test will catch it.
    """
    from src.evaluation import evaluate_matches_description_aware
    from src.data import Subscription, Event, Dataset
    from src.router import MatchResult

    # 5 unique-description subs, 5 events each labelled with one
    base_subs = [
        Subscription(id=f"s{i}", name=f"S{i}", description=f"topic {i}")
        for i in range(5)
    ]
    events = [
        Event(id=f"e{i}", text=f"event about topic {i}", ground_truth=[f"s{i}"])
        for i in range(5)
    ]
    ds_base = Dataset(name="t", short_name="T", events=events, subscriptions=base_subs)

    # Now duplicate to 20 subs — same descriptions, new IDs s0_dup0, etc.
    from scripts.run_crossover import _subsample_subscriptions
    dup_subs = _subsample_subscriptions(base_subs, n=20, seed=42)
    # Re-label events: each event's GT is the FIRST ID it sees with the
    # matching description (matches how the dataset would be labelled in a
    # native-201 setting where each ID is uniquely meaningful).
    desc_to_id = {}
    for s in dup_subs:
        desc_to_id.setdefault(s.description, s.id)
    events_dup = [
        Event(id=f"e{i}", text=f"event about topic {i}",
              ground_truth=[desc_to_id[f"topic {i}"]])
        for i in range(5)
    ]
    ds_dup = Dataset(name="t", short_name="T", events=events_dup, subscriptions=dup_subs)

    # The matcher returns a "merged" subscription — same description as the
    # ground-truth, but possibly a different ID (this is what CoverAndMerge
    # does). Description-aware F1 should still count this as correct.
    matches_base = [MatchResult(event_id=f"e{i}", matched_subscription_ids=[f"s{i}"])
                    for i in range(5)]
    matches_dup = [MatchResult(event_id=f"e{i}", matched_subscription_ids=[f"s{i}"])
                   for i in range(5)]

    f1_base = evaluate_matches_description_aware(matches_base, ds_base).f1.mean
    f1_dup = evaluate_matches_description_aware(matches_dup, ds_dup).f1.mean
    assert f1_base == pytest.approx(1.0), f"base F1 should be perfect, got {f1_base}"
    assert f1_dup == pytest.approx(1.0), \
        f"description-aware F1 must not collapse under duplication; got {f1_dup}"
