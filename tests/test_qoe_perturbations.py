"""TDD for C9: perturbation hooks for the QoE adaptive loop.

The TAAS round-1 reviewer panel mandated a perturbation experiment that
exercises the calibration→assignment→evaluation loop under environmental
shift (5-reviewer consensus). This file is the RED phase.

Design principles applied:
  - TDD: every hook gets its failing test before any implementation.
  - Treatment verification: each test asserts the perturbation was
    APPLIED, not just that the run succeeded.
  - Injection functions raise, not log-and-continue.
  - Failure perturbations target the critical path (one of two
    backends, not both, not none).
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Task 1.1: PerturbationSpec dataclass
# ---------------------------------------------------------------------------

def test_perturbation_spec_defaults_to_none():
    """A default-constructed perturbation must be a no-op (treatment-verifiable)."""
    from src.qoe import PerturbationSpec
    p = PerturbationSpec()
    assert p.kind == "none"
    assert p.is_noop()


def test_perturbation_spec_topic_restricted_calibration():
    """The topic-restricted calibration perturbation must declare a
    topic mask and have is_noop() == False."""
    from src.qoe import PerturbationSpec
    p = PerturbationSpec(
        kind="topic_restricted_cal",
        topic_mask=[0, 1, 2],
        injection_event_index=None,
        injected_latency_s=None,
    )
    assert p.kind == "topic_restricted_cal"
    assert p.topic_mask == [0, 1, 2]
    assert not p.is_noop()


def test_perturbation_spec_latency_injection():
    """Latency injection requires both injection_event_index and
    injected_latency_s."""
    from src.qoe import PerturbationSpec
    p = PerturbationSpec(
        kind="latency_injection",
        injection_event_index=500,
        injected_latency_s=0.5,
    )
    assert p.kind == "latency_injection"
    assert p.injection_event_index == 500
    assert p.injected_latency_s == 0.5
    assert not p.is_noop()


def test_perturbation_spec_failure_injection():
    """Backend-failure injection requires backend_to_fail (target ONE
    of N backends, not all — partial-degradation experiment)."""
    from src.qoe import PerturbationSpec
    p = PerturbationSpec(
        kind="failure_injection",
        injection_event_index=500,
        backend_to_fail="qwen2.5:7b",
    )
    assert p.kind == "failure_injection"
    assert p.backend_to_fail == "qwen2.5:7b"
    assert not p.is_noop()


def test_perturbation_spec_invalid_kind_raises():
    """Injection functions must raise on invalid input, not silently
    log-and-continue."""
    from src.qoe import PerturbationSpec
    with pytest.raises(ValueError, match=r"unknown perturbation kind"):
        PerturbationSpec(kind="garbage_kind").validate()


def test_perturbation_spec_topic_restricted_requires_mask():
    """topic_restricted_cal without a topic_mask must raise."""
    from src.qoe import PerturbationSpec
    with pytest.raises(ValueError, match=r"topic_mask is required"):
        PerturbationSpec(kind="topic_restricted_cal").validate()


def test_perturbation_spec_latency_injection_requires_index_and_latency():
    """latency_injection without injection_event_index or
    injected_latency_s must raise."""
    from src.qoe import PerturbationSpec
    with pytest.raises(ValueError, match=r"injection_event_index is required"):
        PerturbationSpec(
            kind="latency_injection",
            injected_latency_s=0.5,
        ).validate()
    with pytest.raises(ValueError, match=r"injected_latency_s is required"):
        PerturbationSpec(
            kind="latency_injection",
            injection_event_index=500,
        ).validate()


def test_perturbation_spec_failure_injection_requires_backend_and_index():
    """failure_injection requires both injection_event_index and
    backend_to_fail (target ONE backend; not none, not all)."""
    from src.qoe import PerturbationSpec
    with pytest.raises(ValueError, match=r"backend_to_fail is required"):
        PerturbationSpec(
            kind="failure_injection",
            injection_event_index=500,
        ).validate()


# ---------------------------------------------------------------------------
# Task 1.2: topic-restricted calibration hook
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_d1_events():
    """Synthetic D1-shaped events: 100 events labelled with one of 19
    subscription IDs as ground_truth, deterministically distributed."""
    from src.data import Event
    events = []
    for i in range(100):
        topic = i % 19  # deterministic topic distribution: 0, 1, ..., 18, 0, 1, ...
        events.append(Event(
            id=f"e{i}",
            text=f"synthetic event {i} on topic {topic}",
            ground_truth=[f"s{topic}"],
        ))
    return events


def test_sample_calibration_events_no_perturbation(synthetic_d1_events):
    """Default (no perturbation): the sample size respects calibration_fraction
    and contains a mix of all topics."""
    from src.qoe import QoEAssigner

    assigner = QoEAssigner(
        clusters=[],
        backends=["x"],
        calibration_fraction=0.20,
    )
    sample = assigner._sample_calibration_events(synthetic_d1_events)
    assert len(sample) == 20, f"calibration_fraction=0.20 on n=100 → 20 events; got {len(sample)}"
    sampled_topics = {e.ground_truth[0] for e in sample}
    # No restriction → expect mix across topics
    assert len(sampled_topics) > 5, (
        f"unrestricted sample should span many topics; got {len(sampled_topics)}"
    )


def test_sample_calibration_events_topic_restricted(synthetic_d1_events):
    """Treatment-verification: with topic_restricted_cal perturbation,
    every event in the calibration sample must have ≥1 ground_truth label
    in the topic_mask. Non-mask topics must be excluded entirely."""
    from src.qoe import QoEAssigner, PerturbationSpec

    perturbation = PerturbationSpec(
        kind="topic_restricted_cal",
        topic_mask=["s0", "s1", "s2"],
    )
    assigner = QoEAssigner(
        clusters=[],
        backends=["x"],
        calibration_fraction=0.20,
        perturbation=perturbation,
    )
    sample = assigner._sample_calibration_events(synthetic_d1_events)
    # Treatment: every sampled event must have at least one ground_truth in mask
    sampled_topics = {e.ground_truth[0] for e in sample}
    assert sampled_topics.issubset({"s0", "s1", "s2"}), (
        f"topic mask not respected. Sampled topics: {sampled_topics}"
    )
    assert len(sample) > 0, "empty calibration sample is silent failure"
    # Sample size: cap at the number of in-mask events available, then apply fraction.
    # 100 events / 19 topics ≈ 5-6 events per topic; 3 topics → ~15-18 in-mask events.
    # 20% of 100 = 20 → capped at the in-mask pool.
    in_mask = [e for e in synthetic_d1_events if set(e.ground_truth) & {"s0", "s1", "s2"}]
    assert len(sample) <= len(in_mask), (
        f"sample size cannot exceed in-mask pool ({len(in_mask)}); got {len(sample)}"
    )


def test_sample_calibration_events_topic_restricted_empty_pool_raises(synthetic_d1_events):
    """If the topic_mask excludes ALL events (empty calibration pool),
    raise ValueError rather than silently sampling zero events."""
    from src.qoe import QoEAssigner, PerturbationSpec

    perturbation = PerturbationSpec(
        kind="topic_restricted_cal",
        topic_mask=["s99"],  # no event has this topic
    )
    assigner = QoEAssigner(
        clusters=[],
        backends=["x"],
        calibration_fraction=0.20,
        perturbation=perturbation,
    )
    with pytest.raises(ValueError, match=r"topic_restricted_cal.*empty calibration pool"):
        assigner._sample_calibration_events(synthetic_d1_events)


# ---------------------------------------------------------------------------
# Task 1.3: latency-injection hook (mid-run shift)
# ---------------------------------------------------------------------------

def test_apply_latency_injection_below_index():
    """Treatment-verification: for events with idx < injection_event_index,
    the latency-injection helper returns 0 added latency (passthrough)."""
    from src.qoe import PerturbationSpec, apply_latency_injection
    perturbation = PerturbationSpec(
        kind="latency_injection",
        injection_event_index=500,
        injected_latency_s=0.5,
    )
    assert apply_latency_injection(perturbation, event_idx=0) == 0.0
    assert apply_latency_injection(perturbation, event_idx=499) == 0.0


def test_apply_latency_injection_at_or_above_index():
    """Treatment-verification: for events with idx >= injection_event_index,
    the helper returns the configured latency offset."""
    from src.qoe import PerturbationSpec, apply_latency_injection
    perturbation = PerturbationSpec(
        kind="latency_injection",
        injection_event_index=500,
        injected_latency_s=0.5,
    )
    assert apply_latency_injection(perturbation, event_idx=500) == 0.5
    assert apply_latency_injection(perturbation, event_idx=10000) == 0.5


def test_apply_latency_injection_noop_for_other_kinds():
    """The helper must be a no-op for any non-latency_injection perturbation
    kind (including None)."""
    from src.qoe import PerturbationSpec, apply_latency_injection
    assert apply_latency_injection(None, event_idx=500) == 0.0
    p = PerturbationSpec(kind="none")
    assert apply_latency_injection(p, event_idx=500) == 0.0
    p = PerturbationSpec(kind="topic_restricted_cal", topic_mask=["s0"])
    assert apply_latency_injection(p, event_idx=500) == 0.0
    p = PerturbationSpec(kind="failure_injection", injection_event_index=500,
                         backend_to_fail="x")
    assert apply_latency_injection(p, event_idx=500) == 0.0


# ---------------------------------------------------------------------------
# Task 1.4: backend-failure-injection hook
# ---------------------------------------------------------------------------

def test_is_backend_failed_targets_only_specified_backend():
    """Failure-injection targets ONE of N backends. The other backend
    must NOT be marked as failed regardless of event_idx."""
    from src.qoe import PerturbationSpec, is_backend_failed
    perturbation = PerturbationSpec(
        kind="failure_injection",
        injection_event_index=500,
        backend_to_fail="qwen2.5:7b",
    )
    # Targeted backend, post-injection: failed
    assert is_backend_failed(perturbation, "qwen2.5:7b", event_idx=500) is True
    assert is_backend_failed(perturbation, "qwen2.5:7b", event_idx=10000) is True
    # Targeted backend, pre-injection: not failed
    assert is_backend_failed(perturbation, "qwen2.5:7b", event_idx=499) is False
    # NON-targeted backend: never failed (partial degradation)
    assert is_backend_failed(perturbation, "qwen2.5:32b", event_idx=500) is False
    assert is_backend_failed(perturbation, "qwen2.5:32b", event_idx=10000) is False


def test_is_backend_failed_noop_for_other_kinds():
    """The predicate is False for None and any non-failure_injection kind."""
    from src.qoe import PerturbationSpec, is_backend_failed
    assert is_backend_failed(None, "qwen2.5:7b", event_idx=500) is False
    p = PerturbationSpec(kind="none")
    assert is_backend_failed(p, "qwen2.5:7b", event_idx=500) is False
    p = PerturbationSpec(kind="topic_restricted_cal", topic_mask=["s0"])
    assert is_backend_failed(p, "qwen2.5:7b", event_idx=500) is False
    p = PerturbationSpec(kind="latency_injection", injection_event_index=500,
                         injected_latency_s=0.5)
    assert is_backend_failed(p, "qwen2.5:7b", event_idx=500) is False


# ---------------------------------------------------------------------------
# Task 1.6/1.7: per-queue helpers for evaluation-time wiring
# ---------------------------------------------------------------------------

def test_total_latency_offset_for_queue_no_perturbation():
    """No perturbation → zero offset regardless of queue contents."""
    from src.qoe import total_latency_offset_for_queue
    assert total_latency_offset_for_queue(None, event_indices=[10, 20, 30]) == 0.0


def test_total_latency_offset_for_queue_partial_post_injection():
    """With latency_injection at idx=50 + 0.1s offset, a queue with
    event indices [40, 55, 60] should accumulate 0.1+0.1=0.2s (two
    post-injection events; the pre-injection event at idx=40 contributes 0)."""
    from src.qoe import PerturbationSpec, total_latency_offset_for_queue
    perturbation = PerturbationSpec(
        kind="latency_injection",
        injection_event_index=50,
        injected_latency_s=0.1,
    )
    offset = total_latency_offset_for_queue(perturbation, event_indices=[40, 55, 60])
    assert abs(offset - 0.2) < 1e-9, f"expected 0.2s; got {offset}"


def test_total_latency_offset_for_queue_other_kinds_zero():
    """Non-latency_injection perturbations contribute zero offset."""
    from src.qoe import PerturbationSpec, total_latency_offset_for_queue
    p = PerturbationSpec(kind="topic_restricted_cal", topic_mask=["s0"])
    assert total_latency_offset_for_queue(p, event_indices=[100, 200]) == 0.0
    p = PerturbationSpec(kind="failure_injection", injection_event_index=10,
                         backend_to_fail="x")
    assert total_latency_offset_for_queue(p, event_indices=[100, 200]) == 0.0


def test_partition_queue_by_failure_no_perturbation():
    """No perturbation → entire queue passes through to LLM, none failed."""
    from src.qoe import partition_queue_by_failure
    queue_indices = [10, 20, 30]
    healthy, failed = partition_queue_by_failure(
        perturbation=None, backend_name="qwen7b", event_indices=queue_indices
    )
    assert healthy == queue_indices
    assert failed == []


def test_partition_queue_by_failure_targeted_backend_post_injection():
    """failure_injection at idx=50 on backend 'qwen7b' splits events
    into pre-injection (healthy → LLM call) and post-injection (failed
    → empty match). Only the targeted backend is affected."""
    from src.qoe import PerturbationSpec, partition_queue_by_failure
    perturbation = PerturbationSpec(
        kind="failure_injection",
        injection_event_index=50,
        backend_to_fail="qwen7b",
    )
    healthy, failed = partition_queue_by_failure(
        perturbation=perturbation,
        backend_name="qwen7b",
        event_indices=[40, 55, 60, 49],
    )
    assert sorted(healthy) == [40, 49], (
        f"healthy events should be pre-injection only: got healthy={healthy}, failed={failed}"
    )
    assert sorted(failed) == [55, 60], (
        f"failed events should be post-injection on the targeted backend: got {failed}"
    )


def test_partition_queue_by_failure_non_targeted_backend_unchanged():
    """A non-targeted backend (qwen32b) is NOT failed even
    post-injection — partial degradation."""
    from src.qoe import PerturbationSpec, partition_queue_by_failure
    perturbation = PerturbationSpec(
        kind="failure_injection",
        injection_event_index=50,
        backend_to_fail="qwen7b",
    )
    healthy, failed = partition_queue_by_failure(
        perturbation=perturbation,
        backend_name="qwen32b",
        event_indices=[40, 55, 60, 49],
    )
    assert healthy == [40, 55, 60, 49]
    assert failed == []
