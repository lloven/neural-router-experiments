"""Embedding model abstraction for Neural Router.

Wraps sentence-transformers for subscription/event embedding and
cosine similarity computation. Supports disk caching to avoid
recomputation across seeds and runs.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Sentence-transformer embedding model with optional disk caching.

    Caching saves embeddings to `.npy` files keyed by (model_name, text_hash).
    This avoids recomputing embeddings when only the random seed changes
    (e.g., across k-means reruns), saving 5-30s per invocation depending
    on corpus size.
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
        """Generate a cache key from model name and text content hash."""
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


# Models used in the embedding sensitivity experiment
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "e5-large-v2": "intfloat/e5-large-v2",
    "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
}
