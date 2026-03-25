"""Tests for Ollama-based embeddings backend.

TDD RED phase: these tests define the expected interface and behavior
for the OllamaEmbeddings class before implementation exists.

Tests cover:
  1. encode() returns list-of-float-lists (as np.ndarray)
  2. Embedding dimension matches expected (768 for nomic-embed-text)
  3. Batch encoding (multiple texts at once)
  4. Fallback to sentence-transformers when EMBEDDINGS_BACKEND=sentence-transformers
  5. Error handling when Ollama is unreachable
  6. Integration: router works with OllamaEmbeddings
  7. CRITICAL: importing embeddings module does NOT import torch
  8. Factory function creates correct backend based on env var
  9. Disk caching works with OllamaEmbeddings
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Test 7 (CRITICAL): No torch import when using ollama backend
# ---------------------------------------------------------------------------

class TestNoTorchImport:
    """Importing the embeddings module with EMBEDDINGS_BACKEND=ollama must
    NOT trigger a torch import.  This is the primary motivation for the
    entire refactor (GPU VM NFS load cascade).
    """

    def test_no_torch_import_with_ollama_backend(self):
        """Importing embeddings module must not trigger torch import
        when EMBEDDINGS_BACKEND=ollama."""
        # Remove torch and sentence_transformers from sys.modules
        torch_modules = [k for k in sys.modules if k.startswith("torch")]
        st_modules = [k for k in sys.modules if k.startswith("sentence_transformers")]
        saved = {}
        for k in torch_modules + st_modules:
            saved[k] = sys.modules.pop(k)

        try:
            # Also remove our module so it gets re-imported fresh
            emb_modules = [k for k in sys.modules if "embeddings" in k and "neural" in k.lower() or k == "src.embeddings"]
            for k in list(emb_modules):
                if k in sys.modules:
                    sys.modules.pop(k)

            with patch.dict(os.environ, {"EMBEDDINGS_BACKEND": "ollama"}):
                # Re-import the module
                import src.embeddings
                importlib.reload(src.embeddings)

                # Verify torch was NOT imported as a side effect
                assert "torch" not in sys.modules, (
                    "torch was imported as side effect of 'import src.embeddings' "
                    "with EMBEDDINGS_BACKEND=ollama"
                )
        finally:
            # Restore original modules
            sys.modules.update(saved)


# ---------------------------------------------------------------------------
# Test 8: Factory function
# ---------------------------------------------------------------------------

class TestEmbeddingFactory:
    """The create_embedding_model() factory must return the correct backend
    based on the EMBEDDINGS_BACKEND environment variable.
    """

    def test_factory_returns_ollama_by_default(self):
        """Default backend (no env var or EMBEDDINGS_BACKEND=ollama)
        should return OllamaEmbeddings."""
        from src.embeddings import create_embedding_model, OllamaEmbeddings

        with patch.dict(os.environ, {"EMBEDDINGS_BACKEND": "ollama"}):
            model = create_embedding_model()
            assert isinstance(model, OllamaEmbeddings)

    @pytest.mark.skipif(
        not importlib.util.find_spec("sentence_transformers"),
        reason="sentence-transformers not installed"
    )
    def test_factory_returns_sentence_transformers_when_requested(self):
        """EMBEDDINGS_BACKEND=sentence-transformers should return the
        original EmbeddingModel (sentence-transformers based)."""
        from src.embeddings import create_embedding_model, EmbeddingModel

        with patch.dict(os.environ, {"EMBEDDINGS_BACKEND": "sentence-transformers"}):
            # Mock the SentenceTransformer import so this test doesn't need torch
            mock_st = MagicMock()
            mock_st.get_sentence_embedding_dimension.return_value = 384
            with patch("sentence_transformers.SentenceTransformer", return_value=mock_st):
                model = create_embedding_model()
                assert isinstance(model, EmbeddingModel)

    def test_factory_default_is_ollama(self):
        """When EMBEDDINGS_BACKEND is not set at all, default to ollama."""
        from src.embeddings import create_embedding_model, OllamaEmbeddings

        with patch.dict(os.environ, {}, clear=False):
            # Remove EMBEDDINGS_BACKEND if present
            env = os.environ.copy()
            env.pop("EMBEDDINGS_BACKEND", None)
            with patch.dict(os.environ, env, clear=True):
                model = create_embedding_model()
                assert isinstance(model, OllamaEmbeddings)


# ---------------------------------------------------------------------------
# Tests 1-3: OllamaEmbeddings.encode() interface (mocked HTTP)
# ---------------------------------------------------------------------------

def _make_ollama_response(embedding: list[float]) -> dict:
    """Helper: create a mock Ollama /api/embed response."""
    return {"model": "nomic-embed-text", "embeddings": [embedding]}


class TestOllamaEmbeddingsEncode:
    """Test encoding via OllamaEmbeddings with mocked HTTP responses."""

    def _make_model(self, **kwargs):
        from src.embeddings import OllamaEmbeddings
        return OllamaEmbeddings(**kwargs)

    def test_encode_returns_ndarray(self):
        """encode() must return a numpy ndarray of shape (n_texts, dim)."""
        model = self._make_model()
        fake_embedding = [0.1] * 768

        with patch("httpx.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = _make_ollama_response(fake_embedding)
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            result = model.encode(["hello world"])

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 768)

    def test_encode_dimension_768(self):
        """nomic-embed-text produces 768-dim embeddings."""
        model = self._make_model(model="nomic-embed-text")
        fake_embedding = list(np.random.randn(768).tolist())

        with patch("httpx.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = _make_ollama_response(fake_embedding)
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            result = model.encode(["test text"])

        assert result.shape[1] == 768

    def test_encode_batch_multiple_texts(self):
        """Batch encoding of N texts returns (N, dim) array."""
        model = self._make_model()
        texts = ["first", "second", "third"]

        def side_effect(url, **kwargs):
            mock_resp = MagicMock()
            mock_resp.json.return_value = _make_ollama_response(
                list(np.random.randn(768).tolist())
            )
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        with patch("httpx.post", side_effect=side_effect):
            result = model.encode(texts)

        assert result.shape == (3, 768)

    def test_encode_normalizes_by_default(self):
        """When normalize=True (default), output vectors should have unit L2 norm."""
        model = self._make_model()
        raw = np.random.randn(768).tolist()

        with patch("httpx.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = _make_ollama_response(raw)
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            result = model.encode(["text"], normalize=True)

        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 5: Error handling
# ---------------------------------------------------------------------------

class TestOllamaEmbeddingsErrors:
    """Error handling when Ollama is unreachable or returns errors."""

    def test_connection_error_raises(self):
        """ConnectionError when Ollama server is down."""
        from src.embeddings import OllamaEmbeddings
        import httpx

        model = OllamaEmbeddings(base_url="http://localhost:99999")

        with patch("httpx.post", side_effect=httpx.ConnectError("Connection refused")):
            with pytest.raises(httpx.ConnectError):
                model.encode(["test"])

    def test_http_error_raises(self):
        """HTTP 500 from Ollama should raise."""
        from src.embeddings import OllamaEmbeddings
        import httpx

        model = OllamaEmbeddings()

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=MagicMock()
        )

        with patch("httpx.post", return_value=mock_resp):
            with pytest.raises(httpx.HTTPStatusError):
                model.encode(["test"])

    def test_model_not_found_error(self):
        """Ollama returns error when model not pulled."""
        from src.embeddings import OllamaEmbeddings

        model = OllamaEmbeddings(model="nonexistent-model")

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("model 'nonexistent-model' not found")

        with patch("httpx.post", return_value=mock_resp):
            with pytest.raises(Exception, match="not found"):
                model.encode(["test"])


# ---------------------------------------------------------------------------
# Test 9: Disk caching with OllamaEmbeddings
# ---------------------------------------------------------------------------

class TestOllamaEmbeddingsCaching:
    """OllamaEmbeddings must support the same disk caching as EmbeddingModel."""

    def test_cache_hit_skips_http_call(self, tmp_path):
        """Second encode() with same texts should use cache, not HTTP."""
        from src.embeddings import OllamaEmbeddings

        model = OllamaEmbeddings(cache_dir=str(tmp_path))
        fake_embedding = list(np.random.randn(768).tolist())

        with patch("httpx.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = _make_ollama_response(fake_embedding)
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            # First call: hits Ollama
            result1 = model.encode(["cached text"])
            assert mock_post.call_count == 1

            # Second call: should use cache
            result2 = model.encode(["cached text"])
            assert mock_post.call_count == 1  # no additional HTTP call

        np.testing.assert_array_equal(result1, result2)

    def test_different_texts_miss_cache(self, tmp_path):
        """Different texts should not use cached results."""
        from src.embeddings import OllamaEmbeddings

        model = OllamaEmbeddings(cache_dir=str(tmp_path))

        call_count = 0

        def side_effect(url, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_resp = MagicMock()
            mock_resp.json.return_value = _make_ollama_response(
                list(np.random.randn(768).tolist())
            )
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        with patch("httpx.post", side_effect=side_effect):
            model.encode(["text A"])
            model.encode(["text B"])

        assert call_count == 2


# ---------------------------------------------------------------------------
# Test 6: Integration with router (using mocked embeddings)
# ---------------------------------------------------------------------------

class TestRouterIntegration:
    """The router must work with OllamaEmbeddings as a drop-in replacement."""

    def test_router_accepts_ollama_embeddings(self):
        """NeuralRouter.__init__ accepts OllamaEmbeddings instance."""
        from src.embeddings import OllamaEmbeddings
        from src.router import NeuralRouter, RouterConfig
        from src.llm import DryRunLLMClient

        model = OllamaEmbeddings()
        config = RouterConfig()
        llm = DryRunLLMClient()

        router = NeuralRouter(config=config, llm_client=llm, embedding_model=model)
        assert router.embedder is model

    def test_ollama_embeddings_has_dimension_property(self):
        """OllamaEmbeddings must expose .dimension for compatibility."""
        from src.embeddings import OllamaEmbeddings

        model = OllamaEmbeddings(model="nomic-embed-text")
        assert model.dimension == 768

    def test_ollama_embeddings_has_model_name(self):
        """OllamaEmbeddings must expose .model_name for cache key compat."""
        from src.embeddings import OllamaEmbeddings

        model = OllamaEmbeddings(model="nomic-embed-text")
        assert model.model_name == "nomic-embed-text"
