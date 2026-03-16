"""Embedding model abstraction for Neural Router.

Wraps sentence-transformers for subscription/event embedding and
cosine similarity computation.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Sentence-transformer embedding model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
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
        """Encode texts into embeddings.

        Args:
            texts: list of text strings
            batch_size: encoding batch size
            show_progress: show tqdm progress bar
            normalize: L2-normalize embeddings (for cosine similarity via dot product)

        Returns:
            numpy array of shape (len(texts), dimension)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
        )
        return np.array(embeddings)


# Models used in the embedding sensitivity experiment
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "e5-large-v2": "intfloat/e5-large-v2",
    "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
}
