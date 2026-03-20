"""Tests for manifest schema, I/O, and run generation.

TDD RED phase: all tests written before implementation.
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.manifest import Manifest, RunEntry, generate_runs
from src.config import load_profile


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_entry() -> RunEntry:
    return RunEntry(
        run_id="ablation__D1__A3__seed42__qwen7b",
        stage="ablation",
        dataset="D1",
        model_key="qwen7b",
        model_id="ollama/qwen2.5:7b",
        config="A3",
        seed=42,
        status="pending",
        slot_type="ollama",
        max_events=None,
        max_event_words=None,
        started_at=None,
        finished_at=None,
        error=None,
        result_file="results/full/ablation/D1_ablation_qwen7b_results.csv",
        metrics=None,
    )


@pytest.fixture
def full_profile() -> dict:
    """Load the real full profile from params.yaml."""
    return load_profile(mode="full")


# ---------------------------------------------------------------------------
# 1. RunEntry creation
# ---------------------------------------------------------------------------

def test_run_entry_creation(sample_entry: RunEntry):
    assert sample_entry.run_id == "ablation__D1__A3__seed42__qwen7b"
    assert sample_entry.stage == "ablation"
    assert sample_entry.dataset == "D1"
    assert sample_entry.model_key == "qwen7b"
    assert sample_entry.model_id == "ollama/qwen2.5:7b"
    assert sample_entry.config == "A3"
    assert sample_entry.seed == 42
    assert sample_entry.status == "pending"
    assert sample_entry.slot_type == "ollama"
    assert sample_entry.max_events is None
    assert sample_entry.max_event_words is None
    assert sample_entry.started_at is None
    assert sample_entry.finished_at is None
    assert sample_entry.error is None
    assert sample_entry.result_file == "results/full/ablation/D1_ablation_qwen7b_results.csv"
    assert sample_entry.metrics is None


# ---------------------------------------------------------------------------
# 2. Save and load round-trip
# ---------------------------------------------------------------------------

def test_manifest_save_and_load(sample_entry: RunEntry):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "manifest.json"
        m = Manifest(mode="full", runs={sample_entry.run_id: sample_entry})
        m.save(path)

        loaded = Manifest.load(path)
        assert loaded.mode == "full"
        assert sample_entry.run_id in loaded.runs
        r = loaded.runs[sample_entry.run_id]
        assert r.run_id == sample_entry.run_id
        assert r.stage == sample_entry.stage
        assert r.seed == sample_entry.seed
        assert r.status == sample_entry.status
        assert r.metrics is None


# ---------------------------------------------------------------------------
# 3. Atomic write leaves no .tmp artefact
# ---------------------------------------------------------------------------

def test_manifest_atomic_write_creates_no_partial(sample_entry: RunEntry):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "manifest.json"
        m = Manifest(mode="full", runs={sample_entry.run_id: sample_entry})
        m.save(path)

        tmp_path = path.with_suffix(".json.tmp")
        assert not tmp_path.exists(), ".tmp file should not persist after save"
        assert path.exists(), "Final manifest.json should exist"


# ---------------------------------------------------------------------------
# 4. add_run
# ---------------------------------------------------------------------------

def test_manifest_add_run(sample_entry: RunEntry):
    m = Manifest(mode="full", runs={})
    m.add_run(sample_entry)
    assert sample_entry.run_id in m.runs
    assert m.runs[sample_entry.run_id] is sample_entry


# ---------------------------------------------------------------------------
# 5. get_pending_runs
# ---------------------------------------------------------------------------

def test_manifest_get_pending_runs():
    entries = {}
    for i, status in enumerate(["pending", "running", "done", "failed", "pending"]):
        e = RunEntry(
            run_id=f"run_{i}",
            stage="ablation",
            dataset="D1",
            model_key="qwen7b",
            model_id="ollama/qwen2.5:7b",
            config="A0",
            seed=42,
            status=status,
            slot_type="ollama",
            max_events=None,
            max_event_words=None,
            started_at=None,
            finished_at=None,
            error=None,
            result_file=f"results/full/ablation/D1_ablation_qwen7b_{i}.csv",
            metrics=None,
        )
        entries[e.run_id] = e

    m = Manifest(mode="full", runs=entries)
    pending = m.get_pending_runs()
    assert len(pending) == 2
    assert all(r.status == "pending" for r in pending)


# ---------------------------------------------------------------------------
# 6. update_run
# ---------------------------------------------------------------------------

def test_manifest_update_run(sample_entry: RunEntry):
    m = Manifest(mode="full", runs={sample_entry.run_id: sample_entry})
    m.update_run(
        sample_entry.run_id,
        status="done",
        metrics={"f1": 0.85, "precision": 0.9, "recall": 0.8},
    )
    updated = m.runs[sample_entry.run_id]
    assert updated.status == "done"
    assert updated.metrics == {"f1": 0.85, "precision": 0.9, "recall": 0.8}


# ---------------------------------------------------------------------------
# 7. generate_runs produces correct count
# ---------------------------------------------------------------------------

def test_generate_runs_produces_correct_count(full_profile: dict):
    """Full profile run count:

    For each model, count runs across its allowed stages.
    qwen7b and haiku: stages=[ablation, sensitivity, scaling], no overrides
    sonnet: stages=[ablation, sensitivity, scaling], seed_override=[42]

    Ablation per (model, dataset):
      configs(7) * seeds + baselines(6) * 1
      qwen7b/haiku: 7*5 + 6 = 41 per dataset, 3 datasets = 123
      sonnet: 7*1 + 6 = 13 per dataset, 3 datasets = 39

    Sensitivity per (model, dataset):
      4 sweep params, 1 run each = 4 per dataset, 3 datasets = 12
      (same for all models since no seed dimension)

    Scaling per (model, dataset):
      event_counts: 7 runs (D2 + D3 only = 14 event runs)
      No subscription counts in params.yaml, so just events
      qwen7b/haiku: 7*2 = 14
      sonnet: 7*2 = 14

    Totals:
      qwen7b: 123 + 12 + 14 = 149
      haiku:   123 + 12 + 14 = 149
      sonnet:   39 + 12 + 14 =  65

    Grand total: 149 + 149 + 65 = 363
    """
    runs = generate_runs(full_profile)
    # Count per model to debug if needed
    by_model = {}
    for r in runs:
        by_model.setdefault(r.model_key, []).append(r)

    # Ablation: datasets(3) * [configs(7) * seeds + baselines(6)]
    # qwen7b/haiku: 3 * (7*5 + 6) = 3*41 = 123
    # sonnet: 3 * (7*1 + 6) = 3*13 = 39
    abl_qwen = [r for r in by_model.get("qwen7b", []) if r.stage == "ablation"]
    abl_sonnet = [r for r in by_model.get("sonnet", []) if r.stage == "ablation"]
    assert len(abl_qwen) == 123, f"qwen7b ablation: expected 123, got {len(abl_qwen)}"
    assert len(abl_sonnet) == 39, f"sonnet ablation: expected 39, got {len(abl_sonnet)}"

    # Sensitivity: 4 sweeps * 3 datasets = 12 per model
    sens_qwen = [r for r in by_model.get("qwen7b", []) if r.stage == "sensitivity"]
    assert len(sens_qwen) == 12, f"qwen7b sensitivity: expected 12, got {len(sens_qwen)}"

    # Scaling: 7 event_counts * 2 datasets (D2, D3) = 14 per model
    scl_qwen = [r for r in by_model.get("qwen7b", []) if r.stage == "scaling"]
    assert len(scl_qwen) == 14, f"qwen7b scaling: expected 14, got {len(scl_qwen)}"

    expected_total = 149 + 149 + 65
    assert len(runs) == expected_total, f"Expected {expected_total}, got {len(runs)}"


# ---------------------------------------------------------------------------
# 8. Seed override respected (sonnet → single seed 42)
# ---------------------------------------------------------------------------

def test_generate_runs_respects_seed_override(full_profile: dict):
    runs = generate_runs(full_profile)
    sonnet_ablation = [
        r for r in runs if r.model_key == "sonnet" and r.stage == "ablation"
        and r.config.startswith("A")  # exclude baselines
    ]
    seeds_used = {r.seed for r in sonnet_ablation}
    assert seeds_used == {42}, f"sonnet should use only seed 42, got {seeds_used}"


# ---------------------------------------------------------------------------
# 9. max_events_override respected (sonnet D2 → 5000)
# ---------------------------------------------------------------------------

def test_generate_runs_respects_max_events_override(full_profile: dict):
    runs = generate_runs(full_profile)
    sonnet_d2 = [r for r in runs if r.model_key == "sonnet" and r.dataset == "D2"]
    assert len(sonnet_d2) > 0, "Should have sonnet D2 runs"
    for r in sonnet_d2:
        assert r.max_events == 5000, (
            f"sonnet D2 run {r.run_id} should have max_events=5000, got {r.max_events}"
        )

    # sonnet D1 should NOT have max_events override
    sonnet_d1 = [r for r in runs if r.model_key == "sonnet" and r.dataset == "D1"]
    for r in sonnet_d1:
        assert r.max_events is None, (
            f"sonnet D1 run {r.run_id} should have max_events=None, got {r.max_events}"
        )


# ---------------------------------------------------------------------------
# 10. Slot type assignment
# ---------------------------------------------------------------------------

def test_generate_runs_slot_type_assignment(full_profile: dict):
    runs = generate_runs(full_profile)
    for r in runs:
        if r.model_key == "qwen7b":
            assert r.slot_type == "ollama", f"{r.run_id} should be ollama"
        elif r.model_key in ("haiku", "sonnet"):
            assert r.slot_type == "api", f"{r.run_id} should be api"
