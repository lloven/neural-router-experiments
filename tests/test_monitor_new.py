"""Tests for the new manifest-based monitor (scripts/monitor.py).

These tests exercise:
- read_progress_sidecars(): reading progress JSON files from disk
- render_dashboard(): rendering a dashboard string from manifest + sidecars
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.manifest import Manifest, RunEntry

# We import from scripts.monitor — the NEW monitor module.
from scripts.monitor import read_progress_sidecars, render_dashboard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry(
    run_id: str,
    status: str = "pending",
    stage: str = "ablation",
    dataset: str = "D1",
    model_key: str = "qwen7b",
    config: str = "A0",
    seed: int = 42,
    error: str | None = None,
    metrics: dict | None = None,
    **kwargs,
) -> RunEntry:
    defaults = dict(
        run_id=run_id,
        stage=stage,
        dataset=dataset,
        model_key=model_key,
        model_id="ollama:qwen2.5:7b",
        config=config,
        seed=seed,
        status=status,
        slot_type="ollama",
        max_events=100,
        max_event_words=None,
        started_at=None,
        finished_at=None,
        error=error,
        result_file=f"results/full/ablation/{dataset}_ablation_{model_key}_results.csv",
        metrics=metrics,
    )
    defaults.update(kwargs)
    return RunEntry(**defaults)


def _make_manifest(entries: list[RunEntry], mode: str = "full") -> Manifest:
    m = Manifest(mode=mode)
    for e in entries:
        m.add_run(e)
    return m


# ---------------------------------------------------------------------------
# test_read_progress_sidecars
# ---------------------------------------------------------------------------

def test_read_progress_sidecars(tmp_path: Path) -> None:
    """Write sidecar files and verify they're read correctly."""
    mode = "full"
    prog_dir = tmp_path / mode / ".progress"
    prog_dir.mkdir(parents=True)

    sidecar_data = {
        "stage": "ablation__D1__A0__seed42__qwen7b",
        "status": "running",
        "items_done": 450,
        "items_total": 1000,
        "detail": "Processing events 451-500/1000",
        "eta_s": 120.0,
    }
    (prog_dir / "ablation__D1__A0__seed42__qwen7b.json").write_text(
        json.dumps(sidecar_data)
    )

    # Also write one corrupt file that should be skipped
    (prog_dir / "corrupt.json").write_text("{bad json")

    sidecars = read_progress_sidecars(mode, results_root=str(tmp_path))

    assert "ablation__D1__A0__seed42__qwen7b" in sidecars
    assert sidecars["ablation__D1__A0__seed42__qwen7b"]["items_done"] == 450
    # Corrupt file should be silently skipped
    assert "corrupt" not in sidecars


def test_read_progress_sidecars_missing_dir(tmp_path: Path) -> None:
    """Non-existent progress directory returns empty dict."""
    sidecars = read_progress_sidecars("nonexistent", results_root=str(tmp_path))
    assert sidecars == {}


# ---------------------------------------------------------------------------
# test_render_dashboard_shows_progress_counts
# ---------------------------------------------------------------------------

def test_render_dashboard_shows_progress_counts() -> None:
    """Manifest with mixed statuses; verify output contains correct counts."""
    entries = [
        _make_entry("run1", status="done"),
        _make_entry("run2", status="done"),
        _make_entry("run3", status="running"),
        _make_entry("run4", status="failed", error="SIGTERM"),
        _make_entry("run5", status="pending"),
        _make_entry("run6", status="pending"),
    ]
    manifest = _make_manifest(entries)
    output = render_dashboard(manifest, sidecars={})

    assert "2/6 done" in output
    assert "1 running" in output
    assert "2 pending" in output
    assert "1 failed" in output


# ---------------------------------------------------------------------------
# test_render_dashboard_shows_running_with_detail
# ---------------------------------------------------------------------------

def test_render_dashboard_shows_running_with_detail() -> None:
    """Running entry with matching sidecar shows events progress."""
    entries = [
        _make_entry("ablation__D1__A6__seed42__qwen7b", status="running"),
    ]
    manifest = _make_manifest(entries)
    sidecars = {
        "ablation__D1__A6__seed42__qwen7b": {
            "items_done": 1350,
            "items_total": 3200,
            "eta_s": 720.0,
            "detail": "Processing events",
        },
    }
    output = render_dashboard(manifest, sidecars)

    assert "ablation__D1__A6__seed42__qwen7b" in output
    assert "1350/3200" in output
    # ETA should appear (720s = 12m)
    assert "12m" in output


# ---------------------------------------------------------------------------
# test_render_dashboard_shows_done_with_metrics
# ---------------------------------------------------------------------------

def test_render_dashboard_shows_done_with_metrics() -> None:
    """Done entry with metrics shows F1 score."""
    entries = [
        _make_entry(
            "ablation__D1__A0__seed42__qwen7b",
            status="done",
            metrics={"f1": 0.530, "accuracy": 0.85},
            finished_at="2026-03-20T10:00:00",
        ),
    ]
    manifest = _make_manifest(entries)
    output = render_dashboard(manifest, sidecars={})

    assert "ablation__D1__A0__seed42__qwen7b" in output
    assert "F1=0.530" in output


# ---------------------------------------------------------------------------
# test_render_dashboard_shows_failed_with_error
# ---------------------------------------------------------------------------

def test_render_dashboard_shows_failed_with_error() -> None:
    """Failed entry shows error message."""
    entries = [
        _make_entry(
            "sensitivity__D3__sweep_k__seed0__qwen7b",
            status="failed",
            stage="sensitivity",
            error="SIGTERM",
        ),
    ]
    manifest = _make_manifest(entries)
    output = render_dashboard(manifest, sidecars={})

    assert "sensitivity__D3__sweep_k__seed0__qwen7b" in output
    assert "SIGTERM" in output


# ---------------------------------------------------------------------------
# test_render_dashboard_empty_manifest
# ---------------------------------------------------------------------------

def test_render_dashboard_empty_manifest() -> None:
    """Empty manifest produces graceful output (no crash)."""
    manifest = _make_manifest([])
    output = render_dashboard(manifest, sidecars={})

    assert "0/0 done" in output
    # Should still have the header
    assert "Neural Router Experiment Pipeline" in output
