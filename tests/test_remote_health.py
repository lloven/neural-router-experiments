"""Tests for remote health check with embeddings model verification.

TDD RED phase: defines expected behavior for the enhanced health check
that verifies both LLM and embeddings models are available on the VM.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock
import subprocess

import pytest

from src.remote import RemoteConfig, check_ollama_health


class TestOllamaHealthEmbeddingsModel:
    """check_ollama_health must verify both LLM and embeddings models."""

    def _make_config(self, **kwargs) -> RemoteConfig:
        defaults = {
            "ollama_model": "qwen2.5:7b",
            "ollama_embed_model": "nomic-embed-text",
        }
        defaults.update(kwargs)
        return RemoteConfig(**defaults)

    def test_health_check_includes_embed_model_status(self):
        """Health check result must include 'embed_available' key."""
        config = self._make_config()

        ollama_list_output = (
            "NAME              ID           SIZE    MODIFIED\n"
            "qwen2.5:7b        abc123       4.7 GB  2 days ago\n"
            "nomic-embed-text  def456       274 MB  2 days ago\n"
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=ollama_list_output,
                stderr="",
            )
            result = check_ollama_health(config)

        assert "embed_available" in result
        assert result["embed_available"] is True

    def test_health_check_fails_when_embed_model_missing(self):
        """embed_available should be False when nomic-embed-text not pulled."""
        config = self._make_config()

        ollama_list_output = (
            "NAME              ID           SIZE    MODIFIED\n"
            "qwen2.5:7b        abc123       4.7 GB  2 days ago\n"
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=ollama_list_output,
                stderr="",
            )
            result = check_ollama_health(config)

        assert result["embed_available"] is False

    def test_health_check_both_models_available(self):
        """When both LLM and embed models present, both flags True."""
        config = self._make_config()

        ollama_list_output = (
            "NAME              ID           SIZE    MODIFIED\n"
            "qwen2.5:7b        abc123       4.7 GB  2 days ago\n"
            "nomic-embed-text  def456       274 MB  2 days ago\n"
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=ollama_list_output,
                stderr="",
            )
            result = check_ollama_health(config)

        assert result["available"] is True
        assert result["embed_available"] is True

    def test_health_check_ollama_down(self):
        """When Ollama is unreachable, both flags False."""
        config = self._make_config()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout="",
                stderr="Connection refused",
            )
            result = check_ollama_health(config)

        assert result["available"] is False
        assert result.get("embed_available", False) is False

    def test_config_has_embed_model_field(self):
        """RemoteConfig must have ollama_embed_model field."""
        config = RemoteConfig()
        assert hasattr(config, "ollama_embed_model")
        assert config.ollama_embed_model == "nomic-embed-text"

    def test_env_var_overrides_embed_model(self):
        """NROUTER_OLLAMA_EMBED_MODEL env var should override default."""
        from src.remote import load_remote_config

        with patch.dict("os.environ", {"NROUTER_OLLAMA_EMBED_MODEL": "mxbai-embed-large"}):
            config = load_remote_config()
            assert config.ollama_embed_model == "mxbai-embed-large"
