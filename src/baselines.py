"""Baseline matching methods for comparison with the Neural Router.

Six baselines:
  1. BM25 (keyword matching)
  2. Sentence-BERT cosine similarity
  3. Cross-encoder reranker
  4. GloVe cosine similarity
  5. TF-IDF cosine similarity
  6. Word2Vec cosine similarity
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data import Dataset, Event, Subscription
from .router import MatchResult

logger = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    """Result from a baseline method."""
    method: str
    matches: list[MatchResult]
    latency_s: float


def run_baseline(
    method: str,
    dataset: Dataset,
    kappa: int = 3,
) -> BaselineResult:
    """Run a baseline matching method on a dataset."""
    dispatch = {
        "bm25": _run_bm25,
        "sbert": _run_sbert_cosine,
        "cross_encoder": _run_cross_encoder,
        "glove": _run_glove,
        "tfidf": _run_tfidf,
        "word2vec": _run_word2vec,
    }

    if method not in dispatch:
        raise ValueError(f"Unknown baseline: {method}. Available: {list(dispatch.keys())}")

    t0 = time.time()
    matches = dispatch[method](dataset, kappa)
    latency = time.time() - t0

    return BaselineResult(method=method, matches=matches, latency_s=latency)


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

def _run_bm25(dataset: Dataset, kappa: int) -> list[MatchResult]:
    """BM25 keyword matching."""
    from rank_bm25 import BM25Okapi

    # Tokenize subscription descriptions
    sub_texts = [s.description.lower().split() for s in dataset.subscriptions]
    bm25 = BM25Okapi(sub_texts)

    results = []
    for event in dataset.events:
        query = event.text.lower().split()
        scores = bm25.get_scores(query)
        top_indices = np.argsort(scores)[::-1][:kappa]
        matched_ids = [dataset.subscriptions[i].id for i in top_indices if scores[i] > 0]
        # Pad to kappa if fewer matches
        if len(matched_ids) < kappa:
            for i in top_indices:
                if dataset.subscriptions[i].id not in matched_ids:
                    matched_ids.append(dataset.subscriptions[i].id)
                if len(matched_ids) >= kappa:
                    break
        results.append(MatchResult(event_id=event.id, matched_subscription_ids=matched_ids[:kappa]))

    return results


# ---------------------------------------------------------------------------
# Sentence-BERT cosine
# ---------------------------------------------------------------------------

def _run_sbert_cosine(dataset: Dataset, kappa: int) -> list[MatchResult]:
    """Sentence-BERT cosine similarity (same model as Neural Router default)."""
    from .embeddings import EmbeddingModel

    model = EmbeddingModel("all-MiniLM-L6-v2")

    sub_texts = [s.description for s in dataset.subscriptions]
    sub_embeddings = model.encode(sub_texts)

    event_texts = [e.text for e in dataset.events]
    event_embeddings = model.encode(event_texts, show_progress=True)

    similarities = cosine_similarity(event_embeddings, sub_embeddings)

    results = []
    for i, event in enumerate(dataset.events):
        top_indices = np.argsort(similarities[i])[::-1][:kappa]
        matched_ids = [dataset.subscriptions[j].id for j in top_indices]
        results.append(MatchResult(event_id=event.id, matched_subscription_ids=matched_ids))

    return results


# ---------------------------------------------------------------------------
# Cross-encoder reranker
# ---------------------------------------------------------------------------

def _run_cross_encoder(dataset: Dataset, kappa: int) -> list[MatchResult]:
    """Cross-encoder pairwise scoring (ms-marco-MiniLM-L-6-v2)."""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    results = []
    for event in dataset.events:
        pairs = [(event.text, s.description) for s in dataset.subscriptions]
        scores = model.predict(pairs, show_progress_bar=False)
        top_indices = np.argsort(scores)[::-1][:kappa]
        matched_ids = [dataset.subscriptions[j].id for j in top_indices]
        results.append(MatchResult(event_id=event.id, matched_subscription_ids=matched_ids))

    return results


# ---------------------------------------------------------------------------
# TF-IDF cosine
# ---------------------------------------------------------------------------

def _run_tfidf(dataset: Dataset, kappa: int) -> list[MatchResult]:
    """TF-IDF cosine similarity."""
    sub_texts = [s.description for s in dataset.subscriptions]
    event_texts = [e.text for e in dataset.events]

    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
    all_texts = sub_texts + event_texts
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    sub_vectors = tfidf_matrix[:len(sub_texts)]
    event_vectors = tfidf_matrix[len(sub_texts):]

    similarities = cosine_similarity(event_vectors, sub_vectors)

    results = []
    for i, event in enumerate(dataset.events):
        top_indices = np.argsort(similarities[i].toarray().flatten() if hasattr(similarities[i], 'toarray') else similarities[i])[::-1][:kappa]
        matched_ids = [dataset.subscriptions[j].id for j in top_indices]
        results.append(MatchResult(event_id=event.id, matched_subscription_ids=matched_ids))

    return results


# ---------------------------------------------------------------------------
# GloVe cosine
# ---------------------------------------------------------------------------

def _run_glove(dataset: Dataset, kappa: int) -> list[MatchResult]:
    """GloVe average embedding cosine similarity.

    Uses pre-trained GloVe vectors (via gensim or a local file).
    Falls back to random embeddings if GloVe is not available.
    """
    try:
        import gensim.downloader as api
        glove = api.load("glove-wiki-gigaword-100")
        embed_fn = lambda text: _avg_word_vectors(text, glove, 100)
    except Exception as e:
        logger.warning(f"GloVe not available ({e}), using random embeddings as placeholder")
        rng = np.random.RandomState(42)
        embed_fn = lambda text: rng.randn(100)

    sub_embeddings = np.array([embed_fn(s.description) for s in dataset.subscriptions])
    event_embeddings = np.array([embed_fn(e.text) for e in dataset.events])

    # Normalize
    sub_norms = np.linalg.norm(sub_embeddings, axis=1, keepdims=True)
    sub_norms[sub_norms == 0] = 1
    sub_embeddings = sub_embeddings / sub_norms

    event_norms = np.linalg.norm(event_embeddings, axis=1, keepdims=True)
    event_norms[event_norms == 0] = 1
    event_embeddings = event_embeddings / event_norms

    similarities = cosine_similarity(event_embeddings, sub_embeddings)

    results = []
    for i, event in enumerate(dataset.events):
        top_indices = np.argsort(similarities[i])[::-1][:kappa]
        matched_ids = [dataset.subscriptions[j].id for j in top_indices]
        results.append(MatchResult(event_id=event.id, matched_subscription_ids=matched_ids))

    return results


# ---------------------------------------------------------------------------
# Word2Vec cosine
# ---------------------------------------------------------------------------

def _run_word2vec(dataset: Dataset, kappa: int) -> list[MatchResult]:
    """Word2Vec average embedding cosine similarity."""
    try:
        import gensim.downloader as api
        w2v = api.load("word2vec-google-news-300")
        embed_fn = lambda text: _avg_word_vectors(text, w2v, 300)
    except Exception as e:
        logger.warning(f"Word2Vec not available ({e}), using random embeddings as placeholder")
        rng = np.random.RandomState(42)
        embed_fn = lambda text: rng.randn(300)

    sub_embeddings = np.array([embed_fn(s.description) for s in dataset.subscriptions])
    event_embeddings = np.array([embed_fn(e.text) for e in dataset.events])

    sub_norms = np.linalg.norm(sub_embeddings, axis=1, keepdims=True)
    sub_norms[sub_norms == 0] = 1
    sub_embeddings = sub_embeddings / sub_norms

    event_norms = np.linalg.norm(event_embeddings, axis=1, keepdims=True)
    event_norms[event_norms == 0] = 1
    event_embeddings = event_embeddings / event_norms

    similarities = cosine_similarity(event_embeddings, sub_embeddings)

    results = []
    for i, event in enumerate(dataset.events):
        top_indices = np.argsort(similarities[i])[::-1][:kappa]
        matched_ids = [dataset.subscriptions[j].id for j in top_indices]
        results.append(MatchResult(event_id=event.id, matched_subscription_ids=matched_ids))

    return results


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _avg_word_vectors(text: str, model, dim: int) -> np.ndarray:
    """Compute average word vector for a text."""
    words = text.lower().split()
    vectors = []
    for word in words:
        if word in model:
            vectors.append(model[word])
    if not vectors:
        return np.zeros(dim)
    return np.mean(vectors, axis=0)
