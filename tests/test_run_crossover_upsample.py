"""Tests for the subscription-resize behaviour of scripts/run_crossover.py.

The crossover validation experiment in §5.7 needs to drive |S| from 50 up
to 2000 on D1, but D1 has only 19 native subscriptions. The function the
crossover sweep uses to build the per-task subscription set must therefore
support BOTH subsampling (|S| <= native) AND duplication-with-rename
(|S| > native), matching the scaling-experiment helper
`scripts.run_scaling.generate_scaled_subscriptions`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.data import Subscription


def test_run_crossover_helper_subsamples_when_target_le_native():
    """target_count <= native: deterministic subsample, IDs preserved."""
    from scripts.run_crossover import _subsample_subscriptions

    subs = [Subscription(id=f"s{i}", name=f"S{i}", description=f"d{i}") for i in range(20)]
    result = _subsample_subscriptions(subs, n=10, seed=42)
    assert len(result) == 10, f"expected 10 subscriptions, got {len(result)}"
    # All returned IDs are from the originals (no duplication-with-rename here).
    orig_ids = {s.id for s in subs}
    assert all(s.id in orig_ids for s in result), \
        "subsample mode must preserve original IDs"


def test_run_crossover_helper_duplicates_when_target_gt_native():
    """target_count > native: include all originals, duplicate the rest with
    suffixed IDs."""
    from scripts.run_crossover import _subsample_subscriptions

    subs = [Subscription(id=f"s{i}", name=f"S{i}", description=f"d{i}") for i in range(19)]
    result = _subsample_subscriptions(subs, n=200, seed=42)
    assert len(result) == 200, f"expected 200 subscriptions, got {len(result)}"

    # All 19 originals must appear (potentially also duplicated, but at least once).
    orig_ids = {s.id for s in subs}
    result_ids = {s.id for s in result}
    assert orig_ids.issubset(result_ids), \
        f"missing originals: {orig_ids - result_ids}"

    # IDs must all be unique (cluster bookkeeping requires unique IDs).
    assert len(result_ids) == 200, \
        f"non-unique IDs: {len(result_ids)} unique among 200 entries"


def test_run_crossover_helper_is_deterministic_in_seed():
    from scripts.run_crossover import _subsample_subscriptions
    subs = [Subscription(id=f"s{i}", name="S", description="d") for i in range(19)]
    a = _subsample_subscriptions(subs, n=200, seed=42)
    b = _subsample_subscriptions(subs, n=200, seed=42)
    assert [s.id for s in a] == [s.id for s in b], "must be deterministic in seed"

    c = _subsample_subscriptions(subs, n=200, seed=123)
    assert [s.id for s in a] != [s.id for s in c], "different seeds must differ"
