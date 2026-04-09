#!/usr/bin/env python3
"""Live progress monitor for the Neural Router DVC pipeline.

Watches the DVC log file and result directories to display:
  - Per-stage progress bars with individual config×seed detail
  - Overall pipeline progress with ETA
  - Elapsed and estimated remaining time per stage
  - Last 5 meaningful log entries

Usage:
    python scripts/monitor.py                  # default log: logs/dvc_repro.log
    python scripts/monitor.py --log my.log     # custom log path
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_profile, get_model_stages
from src.progress import read_progress

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Stage definitions ────────────────────────────────────────────────────────
# (name, output files, weight, expected_subtasks)
# Smoke test: 3 configs × 1 seed + 2 baselines = 5 subtasks per dataset ablation
# Sensitivity: 4 sweeps × ~5 parameter values = ~20 subtasks
# Scaling: ~7 scale points per dimension

STAGES_FALLBACK = [
    ("fetch-eurovoc",     ["data/eurovoc_labels.csv"],                              1,  127),
    # Ablation (per dataset × model)
    ("ablation-D1-qwen7b",["results/full/ablation/D1_ablation_results.csv"],            15,    5),
    ("ablation-D1-sonnet",["results/full/ablation/D1_ablation_sonnet_results.csv"],     20,    5),
    ("ablation-D2-qwen7b",["results/full/ablation/D2_ablation_qwen7b_results.csv"],    15,    5),
    ("ablation-D2-sonnet",["results/full/ablation/D2_ablation_sonnet_results.csv"],     20,    5),
    ("ablation-D3-qwen7b",["results/full/ablation/D3_ablation_qwen7b_results.csv"],    15,    5),
    ("ablation-D3-sonnet",["results/full/ablation/D3_ablation_sonnet_results.csv"],     20,    5),
    # Ablation - Haiku
    ("ablation-D1-haiku", ["results/full/ablation/D1_ablation_haiku_results.csv"],     10,    5),
    ("ablation-D2-haiku", ["results/full/ablation/D2_ablation_haiku_results.csv"],     10,    5),
    ("ablation-D3-haiku", ["results/full/ablation/D3_ablation_haiku_results.csv"],     10,    5),
    # Sensitivity - Haiku
    ("sensitivity-D1-haiku", ["results/full/sensitivity/haiku/sensitivity_k_D1.csv",
                               "results/full/sensitivity/haiku/sensitivity_tau_D1.csv",
                               "results/full/sensitivity/haiku/sensitivity_kappa_D1.csv",
                               "results/full/sensitivity/haiku/sensitivity_embedding_D1.csv"],  8,  20),
    ("sensitivity-D2-haiku", ["results/full/sensitivity/haiku/sensitivity_k_D2.csv",
                               "results/full/sensitivity/haiku/sensitivity_tau_D2.csv",
                               "results/full/sensitivity/haiku/sensitivity_kappa_D2.csv",
                               "results/full/sensitivity/haiku/sensitivity_embedding_D2.csv"], 10,  20),
    ("sensitivity-D3-haiku", ["results/full/sensitivity/haiku/sensitivity_k_D3.csv",
                               "results/full/sensitivity/haiku/sensitivity_tau_D3.csv",
                               "results/full/sensitivity/haiku/sensitivity_kappa_D3.csv",
                               "results/full/sensitivity/haiku/sensitivity_embedding_D3.csv"],  8,  20),
    # Sensitivity (per dataset × model)
    ("sensitivity-D1-qwen7b", ["results/full/sensitivity/qwen7b/sensitivity_k_D1.csv",
                                "results/full/sensitivity/qwen7b/sensitivity_tau_D1.csv",
                                "results/full/sensitivity/qwen7b/sensitivity_kappa_D1.csv",
                                "results/full/sensitivity/qwen7b/sensitivity_embedding_D1.csv"],  8,  20),
    ("sensitivity-D1-sonnet", ["results/full/sensitivity/sonnet/sensitivity_k_D1.csv",
                                "results/full/sensitivity/sonnet/sensitivity_tau_D1.csv",
                                "results/full/sensitivity/sonnet/sensitivity_kappa_D1.csv",
                                "results/full/sensitivity/sonnet/sensitivity_embedding_D1.csv"], 10,  20),
    ("sensitivity-D2-qwen7b", ["results/full/sensitivity/qwen7b/sensitivity_k_D2.csv",
                                "results/full/sensitivity/qwen7b/sensitivity_tau_D2.csv",
                                "results/full/sensitivity/qwen7b/sensitivity_kappa_D2.csv",
                                "results/full/sensitivity/qwen7b/sensitivity_embedding_D2.csv"], 12,  20),
    ("sensitivity-D2-sonnet", ["results/full/sensitivity/sonnet/sensitivity_k_D2.csv",
                                "results/full/sensitivity/sonnet/sensitivity_tau_D2.csv",
                                "results/full/sensitivity/sonnet/sensitivity_kappa_D2.csv",
                                "results/full/sensitivity/sonnet/sensitivity_embedding_D2.csv"], 15,  20),
    ("sensitivity-D3-qwen7b", ["results/full/sensitivity/qwen7b/sensitivity_k_D3.csv",
                                "results/full/sensitivity/qwen7b/sensitivity_tau_D3.csv",
                                "results/full/sensitivity/qwen7b/sensitivity_kappa_D3.csv",
                                "results/full/sensitivity/qwen7b/sensitivity_embedding_D3.csv"], 10,  20),
    ("sensitivity-D3-sonnet", ["results/full/sensitivity/sonnet/sensitivity_k_D3.csv",
                                "results/full/sensitivity/sonnet/sensitivity_tau_D3.csv",
                                "results/full/sensitivity/sonnet/sensitivity_kappa_D3.csv",
                                "results/full/sensitivity/sonnet/sensitivity_embedding_D3.csv"], 12,  20),
    # Scaling - Haiku
    ("scaling-events-D1-haiku", ["results/full/scaling/haiku/scaling_events_D1.csv"],   4,   7),
    ("scaling-events-D2-haiku", ["results/full/scaling/haiku/scaling_events_D2.csv"],   8,   7),
    ("scaling-events-D3-haiku", ["results/full/scaling/haiku/scaling_events_D3.csv"],   5,   7),
    ("scaling-subs-D2-haiku",   ["results/full/scaling/haiku/scaling_subs_D2.csv"],     8,   6),
    ("scaling-subs-D3-haiku",   ["results/full/scaling/haiku/scaling_subs_D3.csv"],     5,   6),
    # Scaling (per dataset × model × dimension)
    ("scaling-events-D1-qwen7b", ["results/full/scaling/qwen7b/scaling_events_D1.csv"],  4,   7),
    ("scaling-events-D1-sonnet", ["results/full/scaling/sonnet/scaling_events_D1.csv"],   5,   7),
    ("scaling-events-D2-qwen7b", ["results/full/scaling/qwen7b/scaling_events_D2.csv"],  8,   7),
    ("scaling-events-D2-sonnet", ["results/full/scaling/sonnet/scaling_events_D2.csv"],  10,   7),
    ("scaling-subs-D2-qwen7b",   ["results/full/scaling/qwen7b/scaling_subs_D2.csv"],    8,   6),
    ("scaling-subs-D2-sonnet",   ["results/full/scaling/sonnet/scaling_subs_D2.csv"],    10,   6),
    ("scaling-events-D3-qwen7b", ["results/full/scaling/qwen7b/scaling_events_D3.csv"],  6,   7),
    ("scaling-events-D3-sonnet", ["results/full/scaling/sonnet/scaling_events_D3.csv"],   8,   7),
    ("scaling-subs-D3-qwen7b",   ["results/full/scaling/qwen7b/scaling_subs_D3.csv"],    6,   6),
    ("scaling-subs-D3-sonnet",   ["results/full/scaling/sonnet/scaling_subs_D3.csv"],     8,   6),
]

def build_stages(profile: dict) -> list[tuple[str, list[str], int, int]]:
    """Build stage list from params.yaml profile.

    Returns list of (name, output_files, weight, expected_subtasks) tuples.
    """
    mode = profile["mode"]
    p_abl = profile.get("ablation", {})
    datasets = profile.get("datasets", ["D1", "D2", "D3"])

    # Compute expected subtask counts from profile
    n_configs = len(p_abl.get("configs", []))
    n_baselines = len(p_abl.get("baselines", []))
    n_seeds = len(profile.get("seeds", [42]))
    ablation_subtasks = n_configs * n_seeds + n_baselines

    p_sens = profile.get("sensitivity", {})
    sweeps = p_sens.get("sweeps", [])
    sens_subtasks = 0
    for sw in sweeps:
        if sw == "k": sens_subtasks += len(p_sens.get("k_values", []))
        elif sw == "tau": sens_subtasks += len(p_sens.get("tau_values", []))
        elif sw == "kappa": sens_subtasks += len(p_sens.get("kappa_values", []))
        elif sw == "embedding": sens_subtasks += 4  # 4 embedding models

    p_scale = profile.get("scaling", {})
    scale_event_subtasks = len(p_scale.get("event_counts", []))
    scale_subs_subtasks = 6  # typical subscription counts: 5,10,20,50,100,full

    # Weight heuristics by model type
    model_weights = {"qwen7b": 15, "haiku": 10, "sonnet": 20}

    stages = []
    stages.append(("fetch-eurovoc", ["data/eurovoc_labels.csv"], 1, 127))

    llm_models = profile.get("llm_models", {})
    for model_key, model_cfg in llm_models.items():
        allowed = get_model_stages(profile, model_key)
        w = model_weights.get(model_key, 10)

        for ds in datasets:
            if "ablation" in allowed:
                tag = f"{ds}_ablation_{model_key}"
                stages.append((
                    f"ablation-{ds}-{model_key}",
                    [f"results/{mode}/ablation/{tag}_results.csv"],
                    w, ablation_subtasks
                ))

            if "sensitivity" in allowed:
                sens_files = [f"results/{mode}/sensitivity/{model_key}/sensitivity_{sw}_{ds}.csv" for sw in sweeps]
                stages.append((
                    f"sensitivity-{ds}-{model_key}",
                    sens_files,
                    w + 2, sens_subtasks
                ))

            if "scaling" in allowed:
                stages.append((
                    f"scaling-events-{ds}-{model_key}",
                    [f"results/{mode}/scaling/{model_key}/scaling_events_{ds}.csv"],
                    w - 5 if w > 5 else 4, scale_event_subtasks
                ))
                if ds in ("D2", "D3"):  # subscription scaling only for multi-sub datasets
                    stages.append((
                        f"scaling-subs-{ds}-{model_key}",
                        [f"results/{mode}/scaling/{model_key}/scaling_subs_{ds}.csv"],
                        w - 5 if w > 5 else 4, scale_subs_subtasks
                    ))

    return stages


# Patterns that indicate a sub-task completed within a stage
SUBTASK_PATTERNS = [
    re.compile(r"A\d+\s+seed=\d+:"),           # ablation seed result
    re.compile(r"(bm25|sbert|cross_encoder|tfidf|glove|word2vec).*F1="),  # baseline result
    re.compile(r"(k|tau|kappa|embedding_model)=\S+:.*F1="),  # sensitivity result
    re.compile(r"(events|subs)=\d+:.*F1="),     # scaling result
    re.compile(r"\[\d+/\d+\]"),                 # EUROVOC fetch progress
]

# Pattern for granular progress: individual config/baseline completion
DETAIL_PATTERN = re.compile(
    r"(A\d+)\s+seed=(\d+):\s+F1=([\d.]+).*L=([\d.]+)s"
)
BASELINE_PATTERN = re.compile(
    r"Baseline\s+(\w+).*F1=([\d.]+)"
)
# Pattern: "Running A3 on D1 (seed=42, k=19, ...)"
RUNNING_PATTERN = re.compile(
    r"Running\s+(A\d+|Baseline\s+\w+)\s+on\s+(\w+)"
)

# Regex to parse timestamps from log lines: "2026-03-17 05:05:15,142"
TS_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+")


@dataclass
class SubtaskDetail:
    """Tracks an individual config×seed or baseline completion."""
    name: str       # e.g., "A0 seed=42" or "bm25"
    f1: float = 0.0
    latency: float = 0.0
    status: str = "pending"  # pending | running | done


@dataclass
class StageState:
    name: str
    outputs: list[str]
    weight: int
    expected_subtasks: int
    status: str = "pending"       # pending | running | done | cached | failed
    start_time: float | None = None
    end_time: float | None = None
    subtask_count: int = 0
    subtask_details: list[SubtaskDetail] = field(default_factory=list)
    current_subtask: str = ""     # what's currently running

    @property
    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def sub_fraction(self) -> float:
        if self.expected_subtasks <= 0:
            return 0.0
        return min(1.0, self.subtask_count / self.expected_subtasks)

    @property
    def stage_eta(self) -> float:
        """Estimated seconds remaining for this stage."""
        if self.status != "running" or self.subtask_count == 0 or self.start_time is None:
            return -1
        elapsed = time.time() - self.start_time
        rate = elapsed / self.subtask_count
        remaining = self.expected_subtasks - self.subtask_count
        return rate * remaining


def parse_timestamp(line: str) -> float | None:
    """Extract Unix timestamp from a log line. Returns None if no timestamp found."""
    m = TS_PATTERN.search(line)
    if m:
        try:
            dt = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            return dt.timestamp()
        except ValueError:
            pass
    return None


def is_pipeline_process_running() -> bool:
    """Check if DVC or run_experiment process is actually running."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "dvc repro|run_experiment|run_sensitivity|run_scaling"],
            capture_output=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def format_time(seconds: float) -> str:
    if seconds < 0:
        return "--:--"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def progress_bar(fraction: float, width: int = 30) -> str:
    filled = int(width * fraction)
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    return f"[{bar}] {fraction * 100:5.1f}%"


def mini_bar(fraction: float, width: int = 15) -> str:
    filled = int(width * fraction)
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    return f"[{bar}]"


STATUS_ICONS = {
    "pending": "[ ]",
    "running": "[>]",
    "done":    "[+]",
    "cached":  "[=]",
    "failed":  "[!]",
}


def parse_log(log_path: Path, stage_map: dict[str, StageState]) -> list[str]:
    """Parse DVC log for stage lifecycle events and sub-task progress.

    This is idempotent: it re-reads the full log each time and rebuilds
    stage states from scratch (except start_time, which is set once).

    Returns: list of last 5 meaningful log lines for display.
    """
    recent_lines: list[str] = []
    current_stage: str | None = None
    last_ts: float | None = None  # last seen timestamp

    # Reset subtask counts and details (not status or timing)
    for s in stage_map.values():
        s.subtask_count = 0
        s.subtask_details = []
        s.current_subtask = ""

    if not log_path.exists():
        return recent_lines

    with open(log_path, "r", errors="replace") as f:
        for line in f:
            # Track latest timestamp
            ts = parse_timestamp(line)
            if ts:
                last_ts = ts

            # Cached stage (DVC uses both "is cached" and "didn't change, skipping")
            m_cached = re.search(r"Stage '(\S+)' (?:is cached|didn't change)", line)
            if m_cached:
                s = stage_map.get(m_cached.group(1))
                if s:
                    s.status = "cached"
                    s.subtask_count = s.expected_subtasks
                continue

            # Stage start
            m = re.search(r"Running stage '(\S+)':", line)
            if m:
                stage_name = m.group(1)
                # Previous running stage must be done (sequential execution)
                if current_stage and current_stage != stage_name:
                    cs = stage_map.get(current_stage)
                    if cs and cs.status == "running":
                        cs.status = "done"
                        if cs.end_time is None:
                            cs.end_time = last_ts or time.time()
                            cs.subtask_count = cs.expected_subtasks
                current_stage = stage_name
                s = stage_map.get(stage_name)
                if s:
                    # Always reset to running — handles --force re-runs
                    # where a previously "done" stage is relaunched
                    s.status = "running"
                    s.start_time = last_ts or time.time()
                    s.end_time = None
                    s.subtask_count = 0
                    s.subtask_details = []
                continue

            # Lock update = previous stage succeeded
            if "Updating lock file" in line and current_stage:
                s = stage_map.get(current_stage)
                if s and s.status == "running":
                    s.status = "done"
                    if s.end_time is None:
                        s.end_time = last_ts or time.time()
                        s.subtask_count = s.expected_subtasks
                continue

            # Failure
            m2 = re.search(r"failed to reproduce '(\S+)'", line)
            if m2:
                s = stage_map.get(m2.group(1))
                if s:
                    s.status = "failed"
                    if s.end_time is None:
                        s.end_time = last_ts or time.time()
                    if current_stage == m2.group(1):
                        current_stage = None
                continue

            # Track what's currently running (granular)
            if current_stage:
                cs = stage_map.get(current_stage)
                if cs:
                    # "Running A3 on D1 ..."
                    m_running = RUNNING_PATTERN.search(line)
                    if m_running:
                        cs.current_subtask = m_running.group(1)

                    # Completed config: "A0 seed=42: F1=0.6107, I=3, ρ=1.00, L=45.5s"
                    m_detail = DETAIL_PATTERN.search(line)
                    if m_detail:
                        config, seed, f1, latency = m_detail.groups()
                        cs.subtask_details.append(SubtaskDetail(
                            name=f"{config} s={seed}",
                            f1=float(f1),
                            latency=float(latency),
                            status="done",
                        ))

                    # Completed baseline
                    m_base = BASELINE_PATTERN.search(line)
                    if m_base:
                        bname, f1 = m_base.groups()
                        cs.subtask_details.append(SubtaskDetail(
                            name=bname,
                            f1=float(f1),
                            status="done",
                        ))

                    # Sub-task progress count
                    for pattern in SUBTASK_PATTERNS:
                        if pattern.search(line):
                            cs.subtask_count += 1
                            break

            # Collect meaningful log lines (skip LiteLLM noise and ANSI)
            stripped = line.strip()
            if (stripped
                and not stripped.startswith("[92m")
                and "LiteLLM" not in stripped
                and "litellm" not in stripped.lower()
                and "> python" not in stripped
                and "httpx" not in stripped.lower()
                and "resource_tracker" not in stripped):
                clean = re.sub(r"\x1b\[[0-9;]*m", "", stripped)
                # Strip the logger prefix for cleaner display
                clean = re.sub(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+ \[\w+\] \S+: ?", "", clean)
                if clean and len(clean) > 3:
                    recent_lines.append(clean[:100])

    return recent_lines[-5:]


def detect_from_files(states: list[StageState]):
    """Detect stage completion/progress by inspecting output CSV files.

    This enables the monitor to work regardless of whether experiments were
    launched via ``dvc repro`` or manually (e.g., ``nohup python run_experiment.py``).
    For each stage, check if its primary output CSV exists, count data rows,
    and parse per-config details (F1, latency, compression ratio) for the
    fine-grained subtask display.
    """
    import csv as csv_mod

    for s in states:
        # Only check CSV outputs (skip .json companions)
        csvs = [o for o in s.outputs if o.endswith(".csv")]
        if not csvs:
            continue

        all_exist = True
        total_rows = 0
        earliest_mtime = None
        latest_mtime = None
        file_details: list[SubtaskDetail] = []

        for csv_rel in csvs:
            csv_path = PROJECT_ROOT / csv_rel
            if csv_path.exists():
                mtime = csv_path.stat().st_mtime
                if earliest_mtime is None or mtime < earliest_mtime:
                    earliest_mtime = mtime
                if latest_mtime is None or mtime > latest_mtime:
                    latest_mtime = mtime
                try:
                    with open(csv_path) as f:
                        reader = csv_mod.DictReader(f)
                        for row in reader:
                            total_rows += 1
                            # Only parse ablation/sensitivity/scaling CSVs
                            # (they have a "config" or "f1" column)
                            if "config" not in row and "f1" not in row:
                                continue
                            # Build subtask detail from CSV row
                            config = row.get("config", "")
                            seed = row.get("seed", "")
                            try:
                                f1 = float(row.get("f1", 0))
                            except (ValueError, TypeError):
                                f1 = 0.0
                            try:
                                latency = float(row.get("latency_s", 0))
                            except (ValueError, TypeError):
                                latency = 0.0
                            try:
                                rho = float(row.get("compression_ratio", 1))
                            except (ValueError, TypeError):
                                rho = 1.0
                            try:
                                invocations = int(float(row.get("invocations", 0)))
                            except (ValueError, TypeError):
                                invocations = 0

                            # Name: "A0 s=42" or "bm25" for baselines
                            if config.startswith("baseline_"):
                                name = config.replace("baseline_", "")
                            else:
                                name = f"{config} s={seed}"

                            try:
                                cost = float(row.get("cost_per_1k", 0))
                            except (ValueError, TypeError):
                                cost = 0.0
                            try:
                                tok_in = int(float(row.get("tokens_prompt", 0)))
                            except (ValueError, TypeError):
                                tok_in = 0
                            try:
                                tok_out = int(float(row.get("tokens_response", 0)))
                            except (ValueError, TypeError):
                                tok_out = 0

                            detail = SubtaskDetail(
                                name=name, f1=f1, latency=latency, status="done"
                            )
                            # Stash extra info for display
                            detail.rho = rho  # type: ignore[attr-defined]
                            detail.invocations = invocations  # type: ignore[attr-defined]
                            detail.cost = cost  # type: ignore[attr-defined]
                            detail.tok_in = tok_in  # type: ignore[attr-defined]
                            detail.tok_out = tok_out  # type: ignore[attr-defined]
                            file_details.append(detail)
                except Exception:
                    pass
            else:
                all_exist = False

        if not all_exist and total_rows == 0:
            # Log may say "done" from a previous mode (e.g., integration_smoke)
            # but output files for the current mode don't exist yet. Revert.
            if s.status in ("done", "cached") and not s.name.startswith("fetch"):
                s.status = "pending"
                s.subtask_count = 0
                s.subtask_details = []
                s.start_time = None
                s.end_time = None
            continue  # no file evidence yet

        # Only enrich stages that the log parser already identified as running
        # or done. Stages still "pending" per the DVC log may have stale files
        # from previous runs or migrations; DVC will clear those before running.
        if s.status == "pending":
            continue

        # Update subtask details from file (more reliable than log parsing)
        if file_details:
            s.subtask_details = file_details

        # Ablation stages: 3 configs + 2 baselines = 5 rows when complete
        # Sensitivity stages: ~20 rows across 4 CSV files
        # Scaling stages: ~7 rows per CSV
        if total_rows >= s.expected_subtasks:
            if s.status not in ("done", "cached"):
                s.status = "done"
                if earliest_mtime and s.start_time is None:
                    s.start_time = earliest_mtime
                if latest_mtime:
                    s.end_time = latest_mtime
                s.subtask_count = s.expected_subtasks
        elif total_rows > 0:
            s.subtask_count = max(s.subtask_count, total_rows)


def detect_from_progress(states: list[StageState], mode: str):
    """Update stage states from fine-grained progress files.

    These files are written by the ProgressTracker in each script
    and provide within-stage granularity (event-level for ablation,
    parameter-level for sensitivity, scale-point for scaling).
    """
    progress = read_progress(mode)

    for s in states:
        p = progress.get(s.name)
        if not p:
            continue

        updated_at = p.get("updated_at", "")
        # Check if progress file is recent (< 60s old)
        try:
            updated_time = datetime.fromisoformat(updated_at).timestamp()
            age = time.time() - updated_time
        except (ValueError, TypeError):
            age = 999

        if p.get("status") == "done":
            if s.status not in ("done", "cached"):
                s.status = "done"
                s.subtask_count = s.expected_subtasks
            continue

        if p.get("status") == "running" and age < 1200:  # active within last 20 min (Ollama calls can be slow)
            if s.status in ("pending", "running"):
                s.status = "running"
                if s.start_time is None:
                    try:
                        s.start_time = datetime.fromisoformat(p.get("started_at", "")).timestamp()
                    except (ValueError, TypeError):
                        s.start_time = time.time()

            # Update current subtask display
            s.current_subtask = p.get("current_task", "")

            # Update subtask count from progress
            tasks_done = p.get("tasks_done", 0)
            tasks_total = p.get("tasks_total", 0)
            items_done = p.get("items_done", 0)
            items_total = p.get("items_total", 0)

            if tasks_total > 0:
                # Use effective progress (completed tasks + fraction of current)
                if items_total > 0:
                    current_frac = items_done / items_total
                else:
                    current_frac = 0.0
                effective = tasks_done + current_frac
                # Map to expected_subtasks scale
                s.subtask_count = int(effective * s.expected_subtasks / tasks_total)
            elif items_total > 0:
                s.subtask_count = int(items_done * s.expected_subtasks / items_total)

            # Store progress detail for rendering
            s._progress_detail = p.get("detail", "")
            s._progress_eta = p.get("eta_s", -1)
            s._items_done = items_done
            s._items_total = items_total
            s._tasks_done_progress = tasks_done
            s._tasks_total_progress = tasks_total


def render(states: list[StageState], pipeline_start: float, log_path: Path,
           recent_lines: list[str], process_alive: bool):
    """Render the full dashboard to terminal."""
    now = time.time()
    elapsed_total = now - pipeline_start

    n_done = sum(1 for s in states if s.status in ("done", "cached"))
    n_failed = sum(1 for s in states if s.status == "failed")
    n_total = len(states)

    # Overall progress by weight
    total_weight = sum(s.weight for s in states)
    done_weight = sum(s.weight for s in states if s.status in ("done", "cached"))
    running_weight = sum(s.weight * s.sub_fraction for s in states if s.status == "running")
    overall_fraction = (done_weight + running_weight) / total_weight

    # ETA: based on actually-run stages (not cached)
    actual_done_time = sum(s.elapsed for s in states if s.status == "done")
    actual_done_weight = sum(s.weight for s in states if s.status == "done")
    remaining_weight = total_weight - done_weight - running_weight

    if actual_done_weight > 0:
        rate = actual_done_time / actual_done_weight
        eta_seconds = rate * remaining_weight
        # Add running stage's remaining time
        for s in states:
            if s.status == "running" and s.stage_eta > 0:
                eta_seconds = s.stage_eta + rate * (remaining_weight - s.weight * (1 - s.sub_fraction))
                break
    elif any(s.status == "running" for s in states):
        for s in states:
            if s.status == "running" and s.stage_eta > 0:
                stage_total_est = s.elapsed / max(s.sub_fraction, 0.01)
                rate = stage_total_est / s.weight
                eta_seconds = s.stage_eta + rate * (remaining_weight - s.weight * (1 - s.sub_fraction))
                break
        else:
            eta_seconds = -1
    else:
        eta_seconds = -1

    # Clear screen
    sys.stdout.write("\033[2J\033[H")

    print("=" * 78)
    print("  Neural Router Experiment Pipeline")
    print("=" * 78)
    print()
    print(f"  Overall:  {progress_bar(overall_fraction, 40)}  ({n_done}/{n_total} stages)")
    status_line = f"  Elapsed:  {format_time(elapsed_total)}    ETA: {format_time(eta_seconds)}"
    if process_alive:
        status_line += "    * pipeline active"
    else:
        status_line += "    - pipeline stopped"
    print(status_line)
    if n_failed > 0:
        print(f"  \u26A0\uFE0F  {n_failed} stage(s) failed")
    print()

    # Per-stage table
    hdr = f"  {'Stage':<30} {'Status':<9} {'Elapsed':>9}  {'ETA':>9}  {'Progress'}"
    print(hdr)
    print(f"  {'-' * 30} {'-' * 9} {'-' * 9}  {'-' * 9}  {'-' * 22}")

    # Show active/completed stages individually, collapse pending into one line
    pending_count = 0
    for s in states:
        if s.status == "pending":
            pending_count += 1
            continue

        icon = STATUS_ICONS.get(s.status, "?")
        elapsed_str = format_time(s.elapsed) if s.start_time else ""

        if s.status == "running":
            # Use fine-grained item progress if available
            items_done = getattr(s, "_items_done", 0)
            items_total = getattr(s, "_items_total", 0)
            tasks_done_p = 0
            tasks_total_p = 0
            if hasattr(s, "_items_total") and items_total > 0:
                tasks_done_p = getattr(s, "_tasks_done_progress", 0)
                tasks_total_p = getattr(s, "_tasks_total_progress", 0)
                # Compute effective fraction including item-level progress
                if tasks_total_p > 0:
                    frac = (tasks_done_p + items_done / items_total) / tasks_total_p
                else:
                    frac = items_done / items_total
                bar = mini_bar(frac)
                pct = f"{frac * 100:4.0f}%"
                detail = f"{bar} {pct} ({items_done}/{items_total} events, {tasks_done_p}/{tasks_total_p} tasks)"
            else:
                frac = s.sub_fraction
                bar = mini_bar(frac)
                pct = f"{frac * 100:4.0f}%"
                detail = f"{bar} {pct} ({s.subtask_count}/{s.expected_subtasks})"
            # Recompute ETA from elapsed + fraction (more accurate than stale progress file ETA)
            if frac > 0 and s.start_time:
                elapsed_stage = time.time() - s.start_time
                eta_str = format_time(elapsed_stage * (1 - frac) / frac)
            else:
                eta_str = format_time(s.stage_eta)
        elif s.status == "done":
            bar = mini_bar(1.0)
            detail = f"{bar}  100%"
            eta_str = ""
        elif s.status == "cached":
            detail = "cached"
            eta_str = ""
            elapsed_str = ""
        elif s.status == "failed":
            bar = mini_bar(s.sub_fraction)
            pct = f"{s.sub_fraction * 100:4.0f}%"
            detail = f"{bar} {pct} (failed)"
            eta_str = ""
        else:
            detail = ""
            eta_str = ""

        print(f"  {s.name:<30} {icon} {s.status:<6} {elapsed_str:>9}  {eta_str:>9}  {detail}")

    if pending_count > 0:
        print(f"  {'... pending':<30} {STATUS_ICONS['pending']} {'':6} {'':>9}  {'':>9}  {pending_count} stages queued")

    # Granular subtask detail
    # Running stages: full detail.  Completed stages: compact one-line summary.
    print()
    for s in states:
        if not s.subtask_details:
            continue

        if s.status == "running":
            # Full detail for running stage
            print(f"  \u250C\u2500 {s.name}")
            for d in s.subtask_details:
                f1_str = f"F1={d.f1:.4f}" if d.f1 > 0 else "F1=0.000"
                lat_str = f"{d.latency:.0f}s" if d.latency > 0 else ""
                rho = getattr(d, "rho", 1.0)
                rho_str = f"\u03C1={rho:.2f}" if rho != 1.0 else ""
                inv = getattr(d, "invocations", 0)
                inv_str = f"I={inv}" if inv > 0 else ""
                cost = getattr(d, "cost", 0.0)
                cost_str = f"${cost:.4f}/1k" if cost > 0 else ""
                extras = "  ".join(x for x in [inv_str, rho_str, cost_str, lat_str] if x)
                print(f"  \u2502  \u2705 {d.name:<16} {f1_str:>12}  {extras}")
            if s.current_subtask:
                print(f"  \u2502  \U0001F504 {s.current_subtask:<16} running...")
            # Fine-grained progress from progress file
            progress_detail = getattr(s, "_progress_detail", "")
            items_done = getattr(s, "_items_done", 0)
            items_total = getattr(s, "_items_total", 0)
            progress_eta = getattr(s, "_progress_eta", -1)
            if items_total > 0:
                item_bar = mini_bar(items_done / items_total, width=10)
                item_pct = f"{items_done}/{items_total}"
                eta_str = format_time(progress_eta) if progress_eta >= 0 else ""
                print(f"  \u2502  \u23F3 {item_bar} {item_pct}  ETA: {eta_str}  {progress_detail}")
            remaining = s.expected_subtasks - s.subtask_count
            if remaining > 0:
                print(f"  \u2514  {remaining} remaining")
            else:
                print(f"  \u2514\u2500")

        elif s.status in ("done", "cached"):
            # Compact one-line summary: best F1 config + compression
            configs = [d for d in s.subtask_details if not d.name.startswith(("bm25", "sbert", "cross", "tfidf"))]
            baselines = [d for d in s.subtask_details if d.name.startswith(("bm25", "sbert", "cross", "tfidf"))]
            if configs:
                best = max(configs, key=lambda d: d.f1)
                rho = getattr(best, "rho", 1.0)
                rho_str = f" \u03C1={rho:.2f}" if rho != 1.0 else ""
                cost = getattr(best, "cost", 0.0)
                cost_str = f" ${cost:.4f}/1k" if cost > 0 else ""
                parts = [f"best: {best.name} F1={best.f1:.4f}{rho_str}{cost_str}"]
            else:
                parts = []
            if baselines:
                best_bl = max(baselines, key=lambda d: d.f1)
                parts.append(f"best baseline: {best_bl.name} F1={best_bl.f1:.4f}")
            print(f"  \u2500 {s.name}: {' | '.join(parts)}")

    # Recent log entries
    if recent_lines:
        print()
        print("  Recent log:")
        for line in recent_lines:
            print(f"    {line}")

    print()
    print(f"  Log: {log_path}")
    print(f"  Ctrl+C to stop monitoring (pipeline continues in background)")
    print()
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Monitor DVC pipeline progress")
    parser.add_argument("--log", default="logs/dvc_repro.log",
                        help="Primary DVC log file (all dvc_*.log files are also scanned)")
    args = parser.parse_args()

    log_path = PROJECT_ROOT / args.log

    # Load profile to build dynamic stage list
    try:
        profile = load_profile()
    except (FileNotFoundError, ValueError) as e:
        print(f"Warning: Could not load params.yaml ({e}), using fallback stages")
        profile = None

    if profile:
        stage_defs = build_stages(profile)
    else:
        stage_defs = STAGES_FALLBACK  # fallback to hardcoded if params.yaml missing

    states = [StageState(name=n, outputs=o, weight=w, expected_subtasks=st)
              for n, o, w, st in stage_defs]
    stage_map = {s.name: s for s in states}

    # Create log dir if needed
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        log_path.touch()

    pipeline_start = log_path.stat().st_mtime or time.time()

    try:
        while True:
            # Parse ALL dvc_*.log files for stage lifecycle events
            all_recent: list[str] = []
            for lf in sorted(log_path.parent.glob("dvc_*.log")):
                recent = parse_log(lf, stage_map)
                all_recent.extend(recent)
            # Keep only the 5 most recent lines across all logs
            recent_lines = all_recent[-5:] if all_recent else []

            detect_from_files(states)  # override/supplement with file-based detection
            mode = profile["mode"] if profile else "integration_smoke"
            detect_from_progress(states, mode)
            process_alive = is_pipeline_process_running()
            render(states, pipeline_start, log_path, recent_lines, process_alive)

            # Check if pipeline is finished
            if all(s.status in ("done", "cached", "failed") for s in states):
                print("  \u2728 Pipeline complete!")
                break

            # Only exit if no pipeline process AND no running stages with recent activity
            if not process_alive:
                running_count = sum(1 for s in states if s.status == "running")
                started_any = any(s.status != "pending" for s in states)
                if running_count == 0 and started_any:
                    print("  Pipeline process not found and no stages running.")
                    break
                elif started_any:
                    # Process died but stage was running = failure
                    log_age = time.time() - log_path.stat().st_mtime
                    if log_age > 1200:  # 20 min with no process = definitely dead (Ollama calls can take 5+ min each)
                        print("  Pipeline process not found (log inactive for 20min)")
                        break

            time.sleep(5)

    except KeyboardInterrupt:
        print("\n  Monitor stopped. Pipeline may still be running.")
        print(f"  Check log: tail -f {log_path}")


if __name__ == "__main__":
    main()
