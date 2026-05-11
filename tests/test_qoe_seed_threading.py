"""TDD for per-seed calibration-sample stochasticity.

Audit found that QoEAssigner._sample_calibration_events used a hardcoded
np.random.RandomState(42), so every seed in --seeds 42,123,456,789,0 drew
the same calibration sample. The 5-seed CI95 was therefore measuring
k-means + LLM jitter, not calibration-sample stochasticity. These tests
pin the seed-threading contract.

Design principles applied:
  - TDD: every fix gets a failing test before any implementation.
  - Hardcoded RNG seeds in seeded experiments are statistical theatre.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def synthetic_events():
    """Synthetic events large enough that two random samples should differ."""
    from src.data import Event
    return [
        Event(id=f"e{i}", text=f"event {i}", ground_truth=[f"s{i % 19}"])
        for i in range(200)
    ]


def test_qoe_assigner_accepts_seed_parameter():
    """QoEAssigner.__init__ must accept a seed parameter (default 42 for
    backwards compatibility with pre-fix tests)."""
    from src.qoe import QoEAssigner
    a = QoEAssigner(clusters=[], backends=["x"], seed=123)
    assert a.seed == 123
    a_default = QoEAssigner(clusters=[], backends=["x"])
    assert a_default.seed == 42, "default seed must remain 42 for backwards compat"


def test_different_seeds_produce_different_calibration_samples(synthetic_events):
    """Core claim: with different seeds, the calibration sample composition
    must differ (otherwise the 5-seed CI is statistical theatre)."""
    from src.qoe import QoEAssigner

    a42 = QoEAssigner(clusters=[], backends=["x"], calibration_fraction=0.10, seed=42)
    a123 = QoEAssigner(clusters=[], backends=["x"], calibration_fraction=0.10, seed=123)
    s42 = a42._sample_calibration_events(synthetic_events)
    s123 = a123._sample_calibration_events(synthetic_events)

    assert len(s42) == 20
    assert len(s123) == 20
    ids42 = {e.id for e in s42}
    ids123 = {e.id for e in s123}
    assert ids42 != ids123, (
        "different seeds must produce different calibration samples; "
        f"got identical sets of {len(ids42)} ids"
    )


def test_same_seed_produces_identical_sample(synthetic_events):
    """Reproducibility: repeated calls with the same seed must return the same sample."""
    from src.qoe import QoEAssigner

    a1 = QoEAssigner(clusters=[], backends=["x"], calibration_fraction=0.10, seed=789)
    a2 = QoEAssigner(clusters=[], backends=["x"], calibration_fraction=0.10, seed=789)
    s1 = a1._sample_calibration_events(synthetic_events)
    s2 = a2._sample_calibration_events(synthetic_events)
    assert [e.id for e in s1] == [e.id for e in s2]


def test_calibrate_router_config_uses_assigner_seed():
    """The RouterConfig built inside QoEAssigner.calibrate() must use
    self.seed, not a hardcoded 42. Pre-fix the 'seed=42' in src/qoe.py:521
    overrode the assigner's seed; this test pins that line."""
    import inspect
    from src import qoe

    src_text = inspect.getsource(qoe.QoEAssigner.calibrate)
    # The fix replaces "seed": 42 with "seed": self.seed inside RouterConfig
    assert '"seed": 42' not in src_text, (
        "calibrate() must not hardcode seed=42 in RouterConfig; use self.seed"
    )
    assert '"seed": self.seed' in src_text, (
        "calibrate() must thread self.seed into the inner RouterConfig"
    )


def test_run_qoe_passes_seed_to_assigner():
    """scripts/run_qoe.py:438 (QoEAssigner instantiation) must thread the
    outer per-seed seed into the assigner constructor."""
    from pathlib import Path
    src = (
        Path(__file__).resolve().parents[1]
        / "scripts" / "run_qoe.py"
    ).read_text()
    # Find the QoEAssigner( ... call in run_qoe_experiment and verify it
    # passes seed=seed (not just calibration_fraction + perturbation).
    import re
    m = re.search(r"assigner = QoEAssigner\(\s*([^)]+)\s*\)", src, re.DOTALL)
    assert m is not None, "could not find QoEAssigner( ... ) call in run_qoe.py"
    args_block = m.group(1)
    assert "seed=seed" in args_block, (
        "QoEAssigner instantiation must include seed=seed; got args:\n" + args_block
    )
