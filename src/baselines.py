"""Baseline matching methods for comparison with the Neural Router (Section 4.4).

Six non-LLM baselines that match events to subscriptions using different
text similarity approaches. These serve as reference points in the ablation
study (Table 3 in the paper):

  1. BM25          -- sparse keyword matching (rank_bm25 library)
  2. Sentence-BERT -- dense cosine similarity (same model as Neural Router)
  3. Cross-encoder -- pairwise reranker (ms-marco-MiniLM-L-6-v2)
  4. TF-IDF        -- sparse bag-of-words cosine similarity
  5. GloVe         -- average word-vector cosine similarity (100d)
  6. Word2Vec      -- average word-vector cosine similarity (300d)

All baselines return the top-kappa subscriptions per event.
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
    """Result from a baseline method.

    Attributes:
        method: Baseline identifier (e.g., "bm25", "sbert").
        matches: One MatchResult per event.
        latency_s: Wall-clock time for the entire matching run.
    """
    method: str
    matches: list[MatchResult]
    latency_s: float


def run_baseline(
    method: str,
    dataset: Dataset,
    kappa: int = 3,
) -> BaselineResult:
    """Run a baseline matching method on a dataset.

    Args:
        method: One of "bm25", "sbert", "cross_encoder", "glove", "tfidf",
            "word2vec".
        dataset: Loaded dataset with events and subscriptions.
        kappa: Number of top matches to return per event.

    Returns:
        BaselineResult with per-event matches and wall-clock latency.

    Raises:
        ValueError: If `method` is not a recognised baseline name.
    """
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
    """BM25 keyword matching (Okapi BM25).

    Tokenises subscription descriptions as the "document corpus" and each
    event text as a query. Returns top-kappa subscriptions by BM25 score,
    padding with highest-scoring non-zero entries if fewer than kappa match.
    """
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
    """Sentence-BERT cosine similarity using all-MiniLM-L6-v2.

    Uses the same embedding model as the Neural Router's default, but
    without clustering or LLM matching. Pure embedding-space retrieval.
    """
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
    """Cross-encoder pairwise scoring (ms-marco-MiniLM-L-6-v2).

    Scores every (event, subscription) pair jointly, making it O(|M| * |S|)
    in forward passes. More accurate than bi-encoder retrieval but much
    slower for large subscription sets.
    """
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
    """TF-IDF cosine similarity.

    Fits a joint vocabulary over subscriptions and events, then ranks
    subscriptions by cosine similarity in the sparse TF-IDF space.
    """
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
    """GloVe average-word-embedding cosine similarity (100d).

    Downloads glove-wiki-gigaword-100 via gensim's API. Falls back to
    random embeddings if GloVe is unavailable (results will be meaningless
    but the pipeline will not crash).
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
    """Word2Vec average-word-embedding cosine similarity (300d).

    Downloads word2vec-google-news-300 via gensim's API. Falls back to
    random embeddings if Word2Vec is unavailable.
    """
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
    """Compute average word vector for a text.

    Words not in the model's vocabulary are silently skipped. If no words
    are found, returns a zero vector.

    Args:
        text: Input text string.
        model: Gensim KeyedVectors model (GloVe or Word2Vec).
        dim: Embedding dimensionality (must match the model).

    Returns:
        Mean word vector of shape (dim,), or zeros if no vocabulary overlap.
    """
    words = text.lower().split()
    vectors = []
    for word in words:
        if word in model:
            vectors.append(model[word])
    if not vectors:
        return np.zeros(dim)
    return np.mean(vectors, axis=0)
