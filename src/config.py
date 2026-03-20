"""Experiment configuration loader.

Reads params.yaml and resolves the active mode profile.  CLI arguments
in the run_* scripts override profile values; this module provides the
defaults so that dvc.yaml commands stay minimal.

Usage::

    from src.config import load_profile

    profile = load_profile()          # reads params.yaml, resolves active mode
    profile = load_profile(mode="full")  # override mode

    # Access values
    profile["seeds"]                   # [42] or [42, 123, ...]
    profile["ablation"]["configs"]     # ["A0", "A3", "A6"] or all
    profile["max_events"]             # 100 or None
    profile["sensitivity"]["k_values"] # [1, 5, 10, ...]
    profile["scaling"]["event_counts"] # [50, 100, ...]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_PARAMS_FILE = "params.yaml"


def load_params(params_path: str | Path | None = None) -> dict[str, Any]:
    """Load raw params.yaml."""
    path = Path(params_path or _PARAMS_FILE)
    if not path.exists():
        # Try relative to repo root (when called from scripts/)
        alt = Path(__file__).resolve().parent.parent / _PARAMS_FILE
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(f"params.yaml not found at {path} or {alt}")
    with open(path) as f:
        return yaml.safe_load(f)


def load_profile(
    mode: str | None = None,
    params_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load the active mode profile from params.yaml.

    Args:
        mode: Override the mode from params.yaml.  If None, reads the
              ``mode`` key from the file.
        params_path: Path to params.yaml.  Defaults to repo root.

    Returns:
        Dict with profile values (seeds, ablation, sensitivity, scaling, etc.)
        plus top-level ``mode`` and ``llm_models`` keys.
    """
    params = load_params(params_path)
    active_mode = mode or params.get("mode", "integration_smoke")
    profiles = params.get("profiles", {})

    if active_mode not in profiles:
        available = list(profiles.keys())
        raise ValueError(
            f"Unknown mode '{active_mode}'. Available: {available}"
        )

    profile = dict(profiles[active_mode])
    profile["mode"] = active_mode
    profile["llm_models"] = params.get("llm_models", {})
    profile["defaults"] = params.get("defaults", {})

    logger.info(f"Loaded profile: mode={active_mode}")
    return profile


def get_model_stages(
    profile: dict[str, Any],
    model_key: str,
) -> list[str]:
    """Return the list of stages a model should run in the active mode.

    Checks ``stages_by_mode`` first (per-mode scoping), then falls back
    to ``stages`` (same for all modes).
    """
    model_cfg = profile["llm_models"].get(model_key, {})
    mode = profile["mode"]

    # Per-mode scoping (e.g., sonnet has different stages per mode)
    by_mode = model_cfg.get("stages_by_mode", {})
    if mode in by_mode:
        entry = by_mode[mode]
        if isinstance(entry, dict):
            return entry.get("stages", [])
        return entry  # plain list

    # Global stages (same for all modes)
    return model_cfg.get("stages", [])


def get_model_seed_override(
    profile: dict[str, Any],
    model_key: str,
) -> list[int] | None:
    """Return seed override for a model in the active mode, or None."""
    model_cfg = profile["llm_models"].get(model_key, {})
    mode = profile["mode"]

    # Check per-mode override first
    by_mode = model_cfg.get("stages_by_mode", {})
    if mode in by_mode:
        entry = by_mode[mode]
        if isinstance(entry, dict):
            override = entry.get("seed_override")
            if override is not None:
                return override

    # Check top-level model override
    override = model_cfg.get("seed_override")
    if override is not None:
        return override

    return None


def get_model_max_events_override(
    profile: dict[str, Any],
    model_key: str,
    dataset: str,
) -> int | None:
    """Return per-dataset max_events override for a model, or None."""
    model_cfg = profile["llm_models"].get(model_key, {})
    overrides = model_cfg.get("max_events_override", {})
    if overrides and dataset in overrides:
        return overrides[dataset]
    return None


_MODEL_OVERRIDES: dict[str, dict[str, object]] = {
    "qwen7b": {
        "batch_size": 10,
        "llm_timeout": 900,
        "checkpoint_interval": 5,
        "max_parallel": 1,
        "use_async": False,
    },
    "haiku": {
        "batch_size": None,
        "llm_timeout": None,
        "checkpoint_interval": 25,
        "max_parallel": 2,
        "use_async": True,
    },
    "sonnet": {
        "batch_size": None,
        "llm_timeout": None,
        "checkpoint_interval": 50,
        "max_parallel": 4,
        "use_async": True,
    },
}

_DEFAULT_OVERRIDES: dict[str, object] = {
    "batch_size": None,
    "llm_timeout": None,
    "checkpoint_interval": 50,
    "max_parallel": 4,
    "use_async": True,
}


def get_model_overrides(model_key: str) -> dict:
    """Return model-specific execution overrides.

    Returns dict with keys: batch_size, llm_timeout, checkpoint_interval,
    max_parallel, use_async.
    """
    return dict(_MODEL_OVERRIDES.get(model_key, _DEFAULT_OVERRIDES))


def output_dir_for_mode(
    base: str | Path,
    mode: str | None = None,
    stage: str | None = None,
) -> Path:
    """Build output directory path: ``base/mode[/stage]``.

    Args:
        base: Results root (typically "results").
        mode: Active mode name.  If None, omits mode prefix.
        stage: Stage subdirectory (e.g., "ablation", "sensitivity/qwen7b").

    Returns:
        Path object for the output directory.
    """
    p = Path(base)
    if mode:
        p = p / mode
    if stage:
        p = p / stage
    return p
