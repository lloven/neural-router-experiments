"""Tests for manifest generation of new experiment stages (crossover, qoe, scaling_subs).

TDD RED phase: tests written before implementation.
"""

import pytest

from src.manifest import generate_runs
from src.config import load_profile, load_params


# ---------------------------------------------------------------------------
# 1. Crossover runs generated from params.yaml
# ---------------------------------------------------------------------------

def test_generate_runs_includes_crossover_stage():
    """When crossover config exists in params.yaml, generate_runs should
    produce crossover-stage runs."""
    profile = load_profile(mode="full")
    params = load_params()

    # Only generate crossover runs if crossover config is present
    if "crossover" not in params:
        pytest.skip("No crossover config in params.yaml")

    runs = generate_runs(profile)
    crossover_runs = [r for r in runs if r.stage == "crossover"]

    # Should have at least one crossover run
    assert len(crossover_runs) > 0, "Expected crossover runs in manifest"

    # All crossover runs should reference the configured dataset
    crossover_cfg = params["crossover"]
    ds = crossover_cfg.get("dataset", "D1")
    for r in crossover_runs:
        assert r.dataset == ds


# ---------------------------------------------------------------------------
# 2. QoE runs generated
# ---------------------------------------------------------------------------

def test_generate_runs_includes_qoe_stage():
    """When qoe config exists in params.yaml, generate_runs should
    produce qoe-stage runs."""
    profile = load_profile(mode="full")
    params = load_params()

    if "qoe" not in params:
        pytest.skip("No qoe config in params.yaml")

    runs = generate_runs(profile)
    qoe_runs = [r for r in runs if r.stage == "qoe"]

    assert len(qoe_runs) > 0, "Expected QoE runs in manifest"


# ---------------------------------------------------------------------------
# 3. Scaling subscription count runs generated
# ---------------------------------------------------------------------------

def test_generate_runs_includes_scaling_subs_count():
    """When scaling_subs config exists, generate_runs should produce
    scaling runs for subscription count."""
    profile = load_profile(mode="full")
    params = load_params()

    if "scaling_subs" not in params:
        pytest.skip("No scaling_subs config in params.yaml")

    runs = generate_runs(profile)
    subs_count_runs = [r for r in runs if r.stage == "scaling" and "scale_subs_count" in r.config]

    assert len(subs_count_runs) > 0, "Expected subscription-count scaling runs"


# ---------------------------------------------------------------------------
# 4. New stages don't break existing run count
# ---------------------------------------------------------------------------

def test_generate_runs_ablation_count_unchanged():
    """Adding new stages should not change the ablation run count."""
    profile = load_profile(mode="full")
    runs = generate_runs(profile)

    # Ablation count: qwen7b(126) + haiku(126) + sonnet(42) = 294
    ablation_runs = [r for r in runs if r.stage == "ablation"]
    assert len(ablation_runs) == 294, f"Ablation runs changed: {len(ablation_runs)}"
