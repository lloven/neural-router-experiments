"""RED tests for src/remote.py — RemoteConfig and SSH helpers.

TDD Phase: RED. These tests define the desired behavior for remote VM
configuration loading and health checking.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


class TestRemoteConfigLoading:
    """RemoteConfig must load from YAML file or environment variables."""

    def test_loads_from_yaml(self, tmp_path):
        """RemoteConfig loads ssh_host, remote_dir, etc. from a YAML file."""
        from src.remote import RemoteConfig, load_remote_config

        yaml_content = (
            "ssh_host: remote-host\n"
            "remote_dir: ~/neural-router\n"
            "venv_path: ~/neural-router/.venv\n"
            "ollama_model: qwen2.5:7b\n"
        )
        cfg_file = tmp_path / "remote.yaml"
        cfg_file.write_text(yaml_content)

        config = load_remote_config(cfg_file)
        assert isinstance(config, RemoteConfig)
        assert config.ssh_host == "remote-host"
        assert config.remote_dir == "~/neural-router"
        assert config.venv_path == "~/neural-router/.venv"
        assert config.ollama_model == "qwen2.5:7b"

    def test_loads_from_env_vars(self):
        """RemoteConfig falls back to environment variables when no file given."""
        from src.remote import RemoteConfig, load_remote_config

        env = {
            "NROUTER_SSH_HOST": "test-vm",
            "NROUTER_REMOTE_DIR": "/home/test/router",
            "NROUTER_VENV_PATH": "/home/test/router/.venv",
            "NROUTER_OLLAMA_MODEL": "qwen2.5:7b",
        }
        with patch.dict(os.environ, env, clear=False):
            config = load_remote_config(config_path=None)
        assert config.ssh_host == "test-vm"
        assert config.remote_dir == "/home/test/router"

    def test_env_overrides_yaml(self, tmp_path):
        """Environment variables take precedence over YAML values."""
        from src.remote import load_remote_config

        yaml_content = "ssh_host: yaml-host\nremote_dir: ~/yaml-dir\n"
        cfg_file = tmp_path / "remote.yaml"
        cfg_file.write_text(yaml_content)

        with patch.dict(os.environ, {"NROUTER_SSH_HOST": "env-host"}, clear=False):
            config = load_remote_config(cfg_file)
        assert config.ssh_host == "env-host"  # env wins
        assert config.remote_dir == "~/yaml-dir"  # yaml used when no env

    def test_missing_file_and_no_env_uses_defaults(self):
        """When no file and no env vars, use sensible defaults."""
        from src.remote import load_remote_config

        # Clear any NROUTER_ env vars
        clean_env = {k: v for k, v in os.environ.items() if not k.startswith("NROUTER_")}
        with patch.dict(os.environ, clean_env, clear=True):
            config = load_remote_config(config_path=None)
        assert config.ssh_host == "remote-host"  # default
        assert config.remote_dir == "~/neural-router"  # default

    def test_dataclass_fields(self):
        """RemoteConfig has all required fields."""
        from src.remote import RemoteConfig

        rc = RemoteConfig(
            ssh_host="vm",
            remote_dir="~/dir",
            venv_path="~/dir/.venv",
            ollama_model="qwen2.5:7b",
        )
        assert rc.ssh_host == "vm"
        assert rc.remote_dir == "~/dir"
        assert rc.venv_path == "~/dir/.venv"
        assert rc.ollama_model == "qwen2.5:7b"


class TestRemoteHealthChecks:
    """SSH and Ollama health check helpers."""

    def test_check_ssh_connection_success(self):
        """check_ssh returns True when SSH succeeds."""
        from src.remote import RemoteConfig, check_ssh_connection

        config = RemoteConfig(
            ssh_host="test-vm", remote_dir="~", venv_path="~/.venv",
            ollama_model="qwen2.5:7b",
        )
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert check_ssh_connection(config) is True
            mock_run.assert_called_once()
            # Should use ssh with a simple command like 'echo ok'
            args = mock_run.call_args[0][0]
            assert "ssh" in args[0] or args[0] == "ssh"

    def test_check_ssh_connection_failure(self):
        """check_ssh returns False when SSH fails."""
        from src.remote import RemoteConfig, check_ssh_connection

        config = RemoteConfig(
            ssh_host="bad-vm", remote_dir="~", venv_path="~/.venv",
            ollama_model="qwen2.5:7b",
        )
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=255)
            assert check_ssh_connection(config) is False

    def test_check_ollama_health_parses_model(self):
        """check_ollama_health returns dict with model availability."""
        from src.remote import RemoteConfig, check_ollama_health

        config = RemoteConfig(
            ssh_host="test-vm", remote_dir="~", venv_path="~/.venv",
            ollama_model="qwen2.5:7b",
        )
        ollama_output = "NAME            ID          SIZE    MODIFIED\nqwen2.5:7b     abc123      4.4 GB  2 days ago\n"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=ollama_output)
            result = check_ollama_health(config)
            assert result["available"] is True
            assert "qwen2.5:7b" in result["models"]
