"""Embedding model abstraction for Neural Router.

Supports two backends controlled by the EMBEDDINGS_BACKEND environment variable:

  - ``ollama`` (default): calls the Ollama HTTP API (``/api/embed``).
    No torch import. Safe on GPU VMs with NFS home directories.
  - ``sentence-transformers``: uses the sentence-transformers library
    (imports torch). For local development on machines with working CUDA.

Both backends share the same interface (encode, dimension, model_name)
and support disk caching to ``.npy`` files keyed by
(model_name, text_content_hash).

The ``EMBEDDING_MODELS`` dict lists the four models compared in the
embedding sensitivity experiment (Section 4.6).

**Dimension note:** nomic-embed-text (Ollama) produces 768-dim embeddings.
all-MiniLM-L6-v2 (sentence-transformers) produces 384-dim embeddings.
Cached embeddings from one backend are NOT compatible with the other.
Use ONE model consistently per experiment run.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Known dimensions for Ollama embedding models (avoids a probe call at init)
_OLLAMA_EMBED_DIMENSIONS: dict[str, int] = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
    "snowflake-arctic-embed": 1024,
}


class OllamaEmbeddings:
    """Embedding model backed by the Ollama HTTP API.

    Calls ``POST /api/embed`` for each text (or batch). Does NOT import
    torch or sentence-transformers, making it safe on NFS-home GPU VMs.

    Args:
        model: Ollama model name (e.g., "nomic-embed-text").
        base_url: Ollama server URL.
        cache_dir: Parent directory for the embedding cache (optional).
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        cache_dir: Optional[str] = None,
    ):
        self.model_name = model
        self.base_url = base_url.rstrip("/")
        self._dimension = _OLLAMA_EMBED_DIMENSIONS.get(model, 768)
        self._cache_dir = Path(cache_dir) / "embedding_cache" if cache_dir else None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode texts into embeddings via Ollama API.

        Args:
            texts: list of text strings.
            batch_size: ignored (kept for interface compatibility).
            show_progress: ignored (kept for interface compatibility).
            normalize: L2-normalize embeddings (default True).

        Returns:
            numpy array of shape (len(texts), dimension).
        """
        import json as _json
        import urllib.request

        # Try loading from cache
        cache_key = self._cache_key(texts, normalize)
        if cache_key:
            cached = self._load_cache(cache_key)
            if cached is not None:
                logger.info(f"Loaded {len(texts)} embeddings from cache")
                return cached

        embeddings = []
        for text in texts:
            # Use urllib instead of httpx — httpx triggers vfork cascade
            # on LXD containers with cgroup CPU limits.
            data = _json.dumps({"model": self.model_name, "input": text}).encode()
            req = urllib.request.Request(
                f"{self.base_url}/api/embed",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                embedding = _json.loads(resp.read())["embeddings"][0]
            embeddings.append(embedding)

        result = np.array(embeddings, dtype=np.float32)

        if normalize:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # avoid division by zero
            result = result / norms

        # Save to cache
        if cache_key:
            self._save_cache(cache_key, result)

        return result

    def _cache_key(self, texts: list[str], normalize: bool) -> Optional[str]:
        """Generate a cache key from model name, normalization flag, and text content."""
        if not self._cache_dir:
            return None
        content = f"{self.model_name}|norm={normalize}|" + "\n".join(texts)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _load_cache(self, key: str) -> Optional[np.ndarray]:
        """Load cached embeddings from disk."""
        path = self._cache_dir / f"{key}.npy"
        if path.exists():
            try:
                return np.load(path)
            except Exception:
                return None
        return None

    def _save_cache(self, key: str, embeddings: np.ndarray) -> None:
        """Save embeddings to disk cache."""
        path = self._cache_dir / f"{key}.npy"
        try:
            np.save(path, embeddings)
            logger.debug(f"Cached {embeddings.shape[0]} embeddings to {path}")
        except Exception as e:
            logger.warning(f"Failed to cache embeddings: {e}")


class EmbeddingModel:
    """Sentence-transformer embedding model with optional disk caching.

    WARNING: This backend imports torch, which causes NFS load cascades on
    GPU VMs with NFS home directories. Use OllamaEmbeddings on such machines.

    Caching saves embeddings to ``.npy`` files keyed by a SHA-256 hash of
    (model_name, normalization flag, concatenated texts).

    Args:
        model_name: HuggingFace model identifier (e.g., "all-MiniLM-L6-v2").
        device: Torch device string (None for auto-detect).
        cache_dir: Parent directory for the embedding cache.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._cache_dir = Path(cache_dir) / "embedding_cache" if cache_dir else None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self._dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self._dimension}")

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode texts into embeddings, using disk cache when available.

        Args:
            texts: list of text strings
            batch_size: encoding batch size
            show_progress: show tqdm progress bar
            normalize: L2-normalize embeddings (for cosine similarity via dot product)

        Returns:
            numpy array of shape (len(texts), dimension)
        """
        # Try loading from cache
        cache_key = self._cache_key(texts, normalize)
        if cache_key:
            cached = self._load_cache(cache_key)
            if cached is not None:
                logger.info(f"Loaded {len(texts)} embeddings from cache")
                return cached

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
        )
        result = np.array(embeddings)

        # Save to cache
        if cache_key:
            self._save_cache(cache_key, result)

        return result

    def _cache_key(self, texts: list[str], normalize: bool) -> Optional[str]:
        """Generate a cache key from model name, normalization flag, and text content."""
        if not self._cache_dir:
            return None
        content = f"{self.model_name}|norm={normalize}|" + "\n".join(texts)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _load_cache(self, key: str) -> Optional[np.ndarray]:
        """Load cached embeddings from disk."""
        path = self._cache_dir / f"{key}.npy"
        if path.exists():
            try:
                return np.load(path)
            except Exception:
                return None
        return None

    def _save_cache(self, key: str, embeddings: np.ndarray) -> None:
        """Save embeddings to disk cache."""
        path = self._cache_dir / f"{key}.npy"
        try:
            np.save(path, embeddings)
            logger.debug(f"Cached {embeddings.shape[0]} embeddings to {path}")
        except Exception as e:
            logger.warning(f"Failed to cache embeddings: {e}")


def create_embedding_model(
    cache_dir: Optional[str] = None,
    **kwargs,
) -> OllamaEmbeddings | EmbeddingModel:
    """Factory function to create the correct embedding backend.

    Controlled by the ``EMBEDDINGS_BACKEND`` environment variable:
      - ``ollama`` (default): returns OllamaEmbeddings
      - ``sentence-transformers``: returns EmbeddingModel (imports torch)

    Args:
        cache_dir: Parent directory for embedding cache.
        **kwargs: Passed to the backend constructor.

    Returns:
        An embedding model instance with encode(), dimension, and model_name.
    """
    backend = os.environ.get("EMBEDDINGS_BACKEND", "ollama").lower()

    if backend == "sentence-transformers":
        logger.info("Using sentence-transformers embedding backend")
        return EmbeddingModel(cache_dir=cache_dir, **kwargs)
    else:
        logger.info("Using Ollama embedding backend")
        return OllamaEmbeddings(cache_dir=cache_dir, **kwargs)


# Models compared in the embedding sensitivity experiment (Section 4.6, Fig. 6d)
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "e5-large-v2": "intfloat/e5-large-v2",
    "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
    "nomic-embed-text": "ollama/nomic-embed-text",
}
