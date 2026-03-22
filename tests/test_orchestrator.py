"""Tests for SlotPool, sort_by_priority, and orchestrator helpers."""

from unittest.mock import MagicMock, patch
import subprocess
import sys

from src.manifest import Manifest, RunEntry, SlotPool, sort_by_priority
from scripts.run_all import create_or_load_manifest, launch_pending_runs, print_dry_run


def _make_run(run_id: str, stage: str = "ablation", dataset: str = "D1",
              slot_type: str = "ollama", model_key: str = "m1") -> RunEntry:
    """Helper to build a minimal RunEntry for testing."""
    return RunEntry(
        run_id=run_id,
        stage=stage,
        dataset=dataset,
        model_key=model_key,
        model_id=f"{'ollama/' if slot_type == 'ollama' else ''}model",
        config="cfg",
        seed=0,
        status="pending",
        slot_type=slot_type,
        max_events=None,
        max_event_words=None,
        started_at=None,
        finished_at=None,
        error=None,
        result_file="results/test.csv",
        metrics=None,
    )


# -- SlotPool acquire/release ------------------------------------------------

class TestSlotPoolAcquire:

    def test_slot_pool_acquire_ollama(self):
        """Acquire 1 ollama slot succeeds, second fails."""
        pool = SlotPool(ollama_max=1, api_max=2)
        assert pool.acquire("ollama", "run-1") is True
        assert pool.acquire("ollama", "run-2") is False

    def test_slot_pool_acquire_api(self):
        """Acquire 2 api slots succeeds, third fails."""
        pool = SlotPool(ollama_max=1, api_max=2)
        assert pool.acquire("api", "run-a") is True
        assert pool.acquire("api", "run-b") is True
        assert pool.acquire("api", "run-c") is False

    def test_slot_pool_release_frees_slot(self):
        """Release then acquire again succeeds."""
        pool = SlotPool(ollama_max=1, api_max=1)
        assert pool.acquire("api", "run-x") is True
        assert pool.acquire("api", "run-y") is False
        pool.release("api", "run-x")
        assert pool.acquire("api", "run-y") is True

    def test_slot_pool_is_available(self):
        """Check availability without acquiring."""
        pool = SlotPool(ollama_max=1, api_max=2)
        assert pool.is_available("ollama") is True
        pool.acquire("ollama", "run-1")
        assert pool.is_available("ollama") is False
        # Acquiring ollama should not affect api availability.
        assert pool.is_available("api") is True


# -- sort_by_priority --------------------------------------------------------

class TestSortByPriority:

    def test_sort_by_priority_ollama_first(self):
        """Ollama runs sort before api runs."""
        api_run = _make_run("api-1", slot_type="api")
        ollama_run = _make_run("oll-1", slot_type="ollama")
        result = sort_by_priority([api_run, ollama_run])
        assert result[0].slot_type == "ollama"
        assert result[1].slot_type == "api"

    def test_sort_by_priority_ablation_before_sensitivity(self):
        """Within same slot_type, ablation sorts before sensitivity."""
        sens = _make_run("s1", stage="sensitivity", slot_type="ollama")
        abl = _make_run("a1", stage="ablation", slot_type="ollama")
        result = sort_by_priority([sens, abl])
        assert result[0].stage == "ablation"
        assert result[1].stage == "sensitivity"

    def test_sort_by_priority_d1_before_d2(self):
        """Within same stage, D1 sorts before D2."""
        d2 = _make_run("d2", dataset="D2", stage="ablation", slot_type="ollama")
        d1 = _make_run("d1", dataset="D1", stage="ablation", slot_type="ollama")
        result = sort_by_priority([d2, d1])
        assert result[0].dataset == "D1"
        assert result[1].dataset == "D2"


# -- Orchestrator helpers (Task 5) -------------------------------------------

class TestCreateOrLoadManifest:

    def test_creates_manifest_when_none_exists(self, tmp_path):
        """Fresh directory: create manifest from profile, all runs pending."""
        manifest_path = tmp_path / "full" / "manifest.json"
        profile = _minimal_profile()

        manifest = create_or_load_manifest(manifest_path, "full", profile)

        assert manifest_path.exists()
        assert manifest.mode == "full"
        assert len(manifest.runs) > 0
        # Every run should be pending
        for run in manifest.runs.values():
            assert run.status == "pending"

    def test_loads_existing_manifest(self, tmp_path):
        """If manifest.json exists, load it instead of generating."""
        manifest_path = tmp_path / "full" / "manifest.json"
        # Pre-create a manifest with one "done" run
        run = _make_run("existing-run")
        run.status = "done"
        m = Manifest(mode="full", runs={"existing-run": run})
        manifest_path.parent.mkdir(parents=True)
        m.save(manifest_path)

        profile = _minimal_profile()
        manifest = create_or_load_manifest(manifest_path, "full", profile)

        assert len(manifest.runs) == 1
        assert manifest.runs["existing-run"].status == "done"

    def test_reset_failed_to_pending(self, tmp_path):
        """Manifest.reset_failed_to_pending() transitions all failed runs."""
        run_a = _make_run("run-a")
        run_a.status = "failed"
        run_b = _make_run("run-b")
        run_b.status = "done"
        run_c = _make_run("run-c")
        run_c.status = "failed"
        run_d = _make_run("run-d")
        run_d.status = "pending"
        m = Manifest(mode="full", runs={
            "run-a": run_a, "run-b": run_b, "run-c": run_c, "run-d": run_d,
        })

        count = m.reset_failed_to_pending()

        assert count == 2
        assert m.runs["run-a"].status == "pending"
        assert m.runs["run-b"].status == "done"  # unchanged
        assert m.runs["run-c"].status == "pending"
        assert m.runs["run-d"].status == "pending"  # was already pending

    def test_resets_stale_running_to_pending(self, tmp_path):
        """Running entries from a crashed previous run reset to pending."""
        manifest_path = tmp_path / "full" / "manifest.json"
        run_a = _make_run("run-a")
        run_a.status = "running"
        run_b = _make_run("run-b")
        run_b.status = "done"
        run_c = _make_run("run-c")
        run_c.status = "running"
        m = Manifest(mode="full", runs={
            "run-a": run_a, "run-b": run_b, "run-c": run_c,
        })
        manifest_path.parent.mkdir(parents=True)
        m.save(manifest_path)

        profile = _minimal_profile()
        manifest = create_or_load_manifest(manifest_path, "full", profile)

        assert manifest.runs["run-a"].status == "pending"
        assert manifest.runs["run-b"].status == "done"
        assert manifest.runs["run-c"].status == "pending"

        # Verify persisted to disk
        reloaded = Manifest.load(manifest_path)
        assert reloaded.runs["run-a"].status == "pending"


class TestDryRun:

    def test_dry_run_does_not_launch_processes(self, tmp_path, capsys):
        """print_dry_run prints info but never spawns subprocesses."""
        run = _make_run("dry-1")
        m = Manifest(mode="full", runs={"dry-1": run})

        with patch("subprocess.Popen") as mock_popen:
            print_dry_run(m)
            mock_popen.assert_not_called()

        captured = capsys.readouterr()
        assert "dry-1" in captured.out


class TestLaunchPendingRuns:

    def test_respects_slot_limits(self, tmp_path):
        """Only launch as many runs as slots allow."""
        manifest_path = tmp_path / "full" / "manifest.json"
        runs = {}
        for i in range(5):
            r = _make_run(f"run-{i}", slot_type="api")
            runs[r.run_id] = r
        m = Manifest(mode="full", runs=runs)
        manifest_path.parent.mkdir(parents=True)
        m.save(manifest_path)

        pool = SlotPool(ollama_max=1, api_max=2)
        active: dict[str, subprocess.Popen] = {}

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            new_active = launch_pending_runs(
                m, pool, active, manifest_path, "full",
            )

        assert len(new_active) == 2  # only 2 api slots
        assert mock_popen.call_count == 2

    def test_launches_ollama_and_api_independently(self, tmp_path):
        """Ollama and API slots are independent pools."""
        manifest_path = tmp_path / "full" / "manifest.json"
        runs = {}
        for i in range(3):
            r = _make_run(f"oll-{i}", slot_type="ollama")
            runs[r.run_id] = r
        for i in range(3):
            r = _make_run(f"api-{i}", slot_type="api")
            runs[r.run_id] = r
        m = Manifest(mode="full", runs=runs)
        manifest_path.parent.mkdir(parents=True)
        m.save(manifest_path)

        pool = SlotPool(ollama_max=1, api_max=2)
        active: dict[str, subprocess.Popen] = {}

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            new_active = launch_pending_runs(
                m, pool, active, manifest_path, "full",
            )

        # 1 ollama + 2 api = 3 total
        assert len(new_active) == 3
        assert mock_popen.call_count == 3

    def test_marks_launched_runs_as_running(self, tmp_path):
        """Launched runs should be marked running in the manifest."""
        manifest_path = tmp_path / "full" / "manifest.json"
        r = _make_run("launch-1", slot_type="api")
        m = Manifest(mode="full", runs={"launch-1": r})
        manifest_path.parent.mkdir(parents=True)
        m.save(manifest_path)

        pool = SlotPool(ollama_max=1, api_max=2)
        active: dict[str, subprocess.Popen] = {}

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            launch_pending_runs(m, pool, active, manifest_path, "full")

        assert m.runs["launch-1"].status == "running"
        # Also persisted
        reloaded = Manifest.load(manifest_path)
        assert reloaded.runs["launch-1"].status == "running"

    def test_skips_already_active_runs(self, tmp_path):
        """Runs already in active dict should not be launched again."""
        manifest_path = tmp_path / "full" / "manifest.json"
        # Two pending runs, but one is already in active
        r1 = _make_run("act-1", slot_type="api")
        r2 = _make_run("act-2", slot_type="api")
        m = Manifest(mode="full", runs={"act-1": r1, "act-2": r2})
        manifest_path.parent.mkdir(parents=True)
        m.save(manifest_path)

        pool = SlotPool(ollama_max=1, api_max=2)
        # Simulate act-1 already running
        pool.acquire("api", "act-1")
        existing_proc = MagicMock()
        active: dict[str, subprocess.Popen] = {"act-1": existing_proc}

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            new_active = launch_pending_runs(
                m, pool, active, manifest_path, "full",
            )

        # Only act-2 should be newly launched (act-1 was already active)
        assert mock_popen.call_count == 1
        assert "act-2" in new_active
        assert "act-1" in new_active  # preserved from existing


# -- Test helpers ------------------------------------------------------------

def _minimal_profile() -> dict:
    """Return a minimal profile dict that generate_runs() can consume."""
    return {
        "mode": "full",
        "seeds": [42],
        "datasets": ["D1"],
        "max_events": 10,
        "max_event_words": 50,
        "ablation": {
            "configs": ["A0"],
            "baselines": ["bm25"],
        },
        "sensitivity": {
            "sweeps": ["k"],
        },
        "scaling": {
            "event_counts": [50],
        },
        "llm_models": {
            "testmodel": {
                "id": "ollama/test",
                "stages": ["ablation"],
            },
        },
        "defaults": {},
    }
