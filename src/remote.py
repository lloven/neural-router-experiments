"""Remote VM configuration and health check helpers.

Supports loading configuration from a YAML file, environment variables,
or sensible defaults. Environment variables (NROUTER_*) override YAML values.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class RemoteConfig:
    """Configuration for the remote GPU VM."""

    ssh_host: str = "nrouter-vm"
    remote_dir: str = "~/neural-router"
    venv_path: str = "~/neural-router/.venv"
    ollama_model: str = "qwen2.5:7b"
    ollama_embed_model: str = "nomic-embed-text"


# --- Environment variable mapping -------------------------------------------

_ENV_MAP = {
    "NROUTER_SSH_HOST": "ssh_host",
    "NROUTER_REMOTE_DIR": "remote_dir",
    "NROUTER_VENV_PATH": "venv_path",
    "NROUTER_OLLAMA_MODEL": "ollama_model",
    "NROUTER_OLLAMA_EMBED_MODEL": "ollama_embed_model",
}


def load_remote_config(
    config_path: Optional[Path | str] = None,
) -> RemoteConfig:
    """Load RemoteConfig from YAML file and/or environment variables.

    Priority: env vars > YAML > defaults.
    """
    values: dict[str, str] = {}

    # 1. Load from YAML if available
    if config_path is not None:
        path = Path(config_path)
        if path.exists():
            import yaml

            with open(path) as f:
                yaml_data = yaml.safe_load(f) or {}
            for key in RemoteConfig.__dataclass_fields__:
                if key in yaml_data:
                    values[key] = yaml_data[key]

    # 2. Override with environment variables
    for env_key, field_name in _ENV_MAP.items():
        env_val = os.environ.get(env_key)
        if env_val is not None:
            values[field_name] = env_val

    return RemoteConfig(**values)


def check_ssh_connection(config: RemoteConfig) -> bool:
    """Test SSH connectivity to the remote VM. Returns True on success."""
    result = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
         config.ssh_host, "echo", "ok"],
        capture_output=True, text=True, timeout=10,
    )
    return result.returncode == 0


def check_ollama_health(config: RemoteConfig) -> dict:
    """Check if Ollama is running on the VM and required models are available.

    Returns dict with:
      - 'available' (bool): True if the LLM model is available.
      - 'embed_available' (bool): True if the embeddings model is available.
      - 'models' (list of model names): all models found on the VM.
      - 'error' (str, optional): error message if Ollama is unreachable.
    """
    result = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=5", config.ssh_host, "ollama", "list"],
        capture_output=True, text=True, timeout=15,
    )
    if result.returncode != 0:
        return {
            "available": False,
            "embed_available": False,
            "models": [],
            "error": result.stderr.strip(),
        }

    models = []
    for line in result.stdout.strip().splitlines()[1:]:  # skip header
        parts = line.split()
        if parts:
            models.append(parts[0])

    return {
        "available": config.ollama_model in models,
        "embed_available": config.ollama_embed_model in models,
        "models": models,
    }
