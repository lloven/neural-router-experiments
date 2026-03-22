"""Experiment manifest: run entries, matrix expansion, and atomic I/O.

The manifest tracks every run in the experiment as a RunEntry dataclass.
``generate_runs()`` expands the full experiment matrix from a params.yaml
profile.  ``Manifest`` provides dict-based storage with atomic JSON save/load.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.config import (
    get_model_max_events_override,
    get_model_seed_override,
    get_model_stages,
)


@dataclass
class RunEntry:
    """Single experiment run."""

    run_id: str
    stage: str
    dataset: str
    model_key: str
    model_id: str
    config: str
    seed: int
    status: str
    slot_type: str
    max_events: int | None
    max_event_words: int | None
    started_at: str | None
    finished_at: str | None
    error: str | None
    result_file: str
    metrics: dict | None


class Manifest:
    """Collection of RunEntries with atomic JSON persistence."""

    def __init__(self, mode: str, runs: dict[str, RunEntry] | None = None):
        self.mode = mode
        self.runs: dict[str, RunEntry] = runs if runs is not None else {}

    # -- persistence ---------------------------------------------------------

    def save(self, path: Path) -> None:
        """Atomic write: serialise to .tmp then rename."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".json.tmp")
        payload = {
            "mode": self.mode,
            "runs": {rid: asdict(entry) for rid, entry in self.runs.items()},
        }
        tmp_path.write_text(json.dumps(payload, indent=2))
        os.rename(str(tmp_path), str(path))

    @classmethod
    def load(cls, path: Path) -> Manifest:
        """Deserialise manifest from JSON."""
        path = Path(path)
        data = json.loads(path.read_text())
        runs = {
            rid: RunEntry(**entry_data)
            for rid, entry_data in data["runs"].items()
        }
        return cls(mode=data["mode"], runs=runs)

    # -- mutators ------------------------------------------------------------

    def add_run(self, entry: RunEntry) -> None:
        self.runs[entry.run_id] = entry

    def get_pending_runs(self) -> list[RunEntry]:
        return [r for r in self.runs.values() if r.status == "pending"]

    def reset_failed_to_pending(self) -> int:
        """Reset all failed runs back to pending so they can be retried.

        Returns the number of runs reset.
        """
        count = 0
        for entry in self.runs.values():
            if entry.status == "failed":
                entry.status = "pending"
                count += 1
        return count

    def update_run(self, run_id: str, **kwargs: Any) -> None:
        entry = self.runs[run_id]
        for k, v in kwargs.items():
            setattr(entry, k, v)


# ---------------------------------------------------------------------------
# Matrix expansion
# ---------------------------------------------------------------------------

def _slot_type(model_id: str) -> str:
    return "ollama" if "ollama" in model_id else "api"


def _make_run_id(stage: str, dataset: str, config: str, seed: int, model_key: str) -> str:
    return f"{stage}__{dataset}__{config}__seed{seed}__{model_key}"


@dataclass
class SlotPool:
    """Concurrency limiter for ollama and API inference slots."""

    ollama_max: int = 1
    api_max: int = 2
    _ollama_active: set[str] = field(default_factory=set)
    _api_active: set[str] = field(default_factory=set)

    def _get(self, slot_type: str) -> tuple[set[str], int]:
        if slot_type == "ollama":
            return self._ollama_active, self.ollama_max
        return self._api_active, self.api_max

    def acquire(self, slot_type: str, run_id: str) -> bool:
        """Try to acquire a slot. Returns True if acquired, False if at capacity."""
        active, maximum = self._get(slot_type)
        if len(active) >= maximum:
            return False
        active.add(run_id)
        return True

    def release(self, slot_type: str, run_id: str) -> None:
        """Release a slot."""
        active, _ = self._get(slot_type)
        active.discard(run_id)

    def is_available(self, slot_type: str) -> bool:
        """Check if a slot is available without acquiring."""
        active, maximum = self._get(slot_type)
        return len(active) < maximum


_STAGE_ORDER = {"ablation": 0, "sensitivity": 1, "scaling": 2}


def sort_by_priority(runs: list[RunEntry]) -> list[RunEntry]:
    """Sort runs by scheduling priority.

    Order: ollama before api, then ablation > sensitivity > scaling,
    then D1 > D2 > D3 (smallest dataset first).
    """
    def key(r: RunEntry) -> tuple[int, int, str]:
        slot_rank = 0 if r.slot_type == "ollama" else 1
        stage_rank = _STAGE_ORDER.get(r.stage, 99)
        return (slot_rank, stage_rank, r.dataset)

    return sorted(runs, key=key)


def generate_runs(profile: dict[str, Any]) -> list[RunEntry]:
    """Expand the full experiment matrix from a profile dict.

    Returns a flat list of RunEntry objects covering ablation, sensitivity,
    and scaling stages for every model allowed by the profile.
    """
    mode = profile["mode"]
    datasets = profile["datasets"]
    seeds = profile["seeds"]
    max_events_global = profile.get("max_events")
    max_event_words_global = profile.get("max_event_words")

    ablation_cfg = profile["ablation"]
    configs = ablation_cfg["configs"]
    baselines = ablation_cfg["baselines"]

    sensitivity_cfg = profile["sensitivity"]
    sweeps = sensitivity_cfg["sweeps"]

    scaling_cfg = profile["scaling"]
    event_counts = scaling_cfg["event_counts"]

    models = profile["llm_models"]
    runs: list[RunEntry] = []

    for model_key, model_info in models.items():
        model_id = model_info["id"]
        slot = _slot_type(model_id)
        stages = get_model_stages(profile, model_key)
        seed_override = get_model_seed_override(profile, model_key)
        model_seeds = seed_override if seed_override is not None else seeds

        for ds in datasets:
            me_override = get_model_max_events_override(profile, model_key, ds)
            me = me_override if me_override is not None else max_events_global
            mew = max_event_words_global

            # -- ablation ----------------------------------------------------
            if "ablation" in stages:
                # Config runs (seeded)
                for cfg in configs:
                    for s in model_seeds:
                        rid = _make_run_id("ablation", ds, cfg, s, model_key)
                        runs.append(RunEntry(
                            run_id=rid,
                            stage="ablation",
                            dataset=ds,
                            model_key=model_key,
                            model_id=model_id,
                            config=cfg,
                            seed=s,
                            status="pending",
                            slot_type=slot,
                            max_events=me,
                            max_event_words=mew,
                            started_at=None,
                            finished_at=None,
                            error=None,
                            result_file=f"results/{mode}/ablation/{ds}_ablation_{model_key}_results.csv",
                            metrics=None,
                        ))
                # Baselines (no seed dimension; use seed=0)
                for bl in baselines:
                    bl_name = f"baseline_{bl}"
                    rid = _make_run_id("ablation", ds, bl_name, 0, model_key)
                    runs.append(RunEntry(
                        run_id=rid,
                        stage="ablation",
                        dataset=ds,
                        model_key=model_key,
                        model_id=model_id,
                        config=bl_name,
                        seed=0,
                        status="pending",
                        slot_type=slot,
                        max_events=me,
                        max_event_words=mew,
                        started_at=None,
                        finished_at=None,
                        error=None,
                        result_file=f"results/{mode}/ablation/{ds}_ablation_{model_key}_results.csv",
                        metrics=None,
                    ))

            # -- sensitivity -------------------------------------------------
            if "sensitivity" in stages:
                for param in sweeps:
                    rid = f"sensitivity__{ds}__sweep_{param}__seed0__{model_key}"
                    runs.append(RunEntry(
                        run_id=rid,
                        stage="sensitivity",
                        dataset=ds,
                        model_key=model_key,
                        model_id=model_id,
                        config=f"sweep_{param}",
                        seed=0,
                        status="pending",
                        slot_type=slot,
                        max_events=me,
                        max_event_words=mew,
                        started_at=None,
                        finished_at=None,
                        error=None,
                        result_file=f"results/{mode}/sensitivity/{model_key}/sensitivity_{param}_{ds}.csv",
                        metrics=None,
                    ))

            # -- scaling -----------------------------------------------------
            if "scaling" in stages and ds in ("D2", "D3"):
                for ec in event_counts:
                    rid = f"scaling__{ds}__scale_events_{ec}__seed0__{model_key}"
                    runs.append(RunEntry(
                        run_id=rid,
                        stage="scaling",
                        dataset=ds,
                        model_key=model_key,
                        model_id=model_id,
                        config=f"scale_events_{ec}",
                        seed=0,
                        status="pending",
                        slot_type=slot,
                        max_events=me,
                        max_event_words=mew,
                        started_at=None,
                        finished_at=None,
                        error=None,
                        result_file=f"results/{mode}/scaling/{model_key}/scaling_events_{ds}.csv",
                        metrics=None,
                    ))

    return runs


# ---------------------------------------------------------------------------
# Migration: mark runs as done if their result files already exist
# ---------------------------------------------------------------------------

def migrate_existing_results(manifest: Manifest) -> int:
    """Scan result files on disk and mark matching runs as 'done'.

    Returns the number of runs migrated. Only marks a run done if its
    result_file exists AND contains at least one data row.
    """
    import csv as csv_mod

    migrated = 0
    for entry in manifest.runs.values():
        if entry.status != "pending":
            continue
        result_path = Path(entry.result_file)
        if not result_path.exists():
            continue
        try:
            with open(result_path) as f:
                reader = csv_mod.DictReader(f)
                rows = list(reader)
            if not rows:
                continue
            # For ablation: check if this specific config+seed has a row
            if entry.stage == "ablation":
                matching = [
                    r for r in rows
                    if r.get("config") == entry.config
                    and str(r.get("seed", "")) == str(entry.seed)
                ]
                if not matching:
                    continue
                # Extract metrics from the matching row
                row = matching[0]
                entry.metrics = {
                    k: float(row[k]) for k in ("f1", "precision", "recall")
                    if k in row
                }
            entry.status = "done"
            migrated += 1
        except Exception:
            continue
    return migrated
