"""TDD for matched-pair LLM-call caching.

Audit found that round_robin (calibration-free) showed F1 deltas of -0.12
between independently-submitted SLURM jobs, which is impossible from
calibration restriction (round_robin doesn't see calibration). The variance
was dominated by LLM nondeterminism between separate ollama processes.

To enable matched-pair perturbation analysis, LLMClient gains an optional
on-disk response cache keyed by (model, prompt_hash). The cache is a
JSONL file; the second run with the same (model, prompt) returns the cached
LLMResponse without invoking the API. This neutralises LLM-nondeterminism
across SLURM jobs so that baseline-vs-perturbed comparisons measure the
intended treatment.

Lessons applied:
  L31 — TDD: every cache feature gets a failing test before implementation.
  L51 — Failure must propagate: a corrupted cache entry must raise, not
        silently return stale data.
  L65 — Statistical theatre: stochasticity sources must be traceable.
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


def _make_fake_litellm_response(text: str, prompt_tokens: int = 10, response_tokens: int = 5):
    """Return a MagicMock that mimics a litellm.completion() return value."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = text
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = response_tokens
    resp.model_dump = lambda: {"text": text}
    return resp


def test_llm_client_accepts_cache_path(tmp_path):
    """LLMClient must accept a cache_path: Path | str | None parameter."""
    from src.llm import LLMClient
    cache = tmp_path / "llm_cache.jsonl"
    c = LLMClient(model="ollama/qwen2.5:7b", cache_path=cache)
    assert c.cache_path == cache


def test_cache_miss_calls_api_and_writes_entry(tmp_path):
    """First call with a fresh cache must invoke the API and write to disk."""
    from src.llm import LLMClient
    cache = tmp_path / "llm_cache.jsonl"
    c = LLMClient(model="ollama/qwen2.5:7b", cache_path=cache)

    with patch("litellm.completion") as mock_completion:
        mock_completion.return_value = _make_fake_litellm_response("hello")
        r = c.invoke("test prompt")
        assert r.text == "hello"
        assert mock_completion.call_count == 1, "cache miss must call API"

    assert cache.exists(), "cache file must be created on miss"
    lines = cache.read_text().splitlines()
    assert len(lines) == 1, "exactly one cache entry expected"
    entry = json.loads(lines[0])
    assert entry["model"] == "ollama/qwen2.5:7b"
    assert entry["text"] == "hello"
    assert "prompt_hash" in entry


def test_cache_hit_skips_api_call(tmp_path):
    """Second call with same (model, prompt) must hit cache and skip API."""
    from src.llm import LLMClient
    cache = tmp_path / "llm_cache.jsonl"

    # First call (miss) populates cache
    c1 = LLMClient(model="ollama/qwen2.5:7b", cache_path=cache)
    with patch("litellm.completion") as mock_completion:
        mock_completion.return_value = _make_fake_litellm_response("cached_response")
        c1.invoke("identical prompt")
        assert mock_completion.call_count == 1

    # Second call (hit) — fresh client, fresh API mock that should NOT be called
    c2 = LLMClient(model="ollama/qwen2.5:7b", cache_path=cache)
    with patch("litellm.completion") as mock_completion:
        r = c2.invoke("identical prompt")
        assert r.text == "cached_response", "must return cached text"
        assert mock_completion.call_count == 0, (
            "cache hit must NOT call API; matched-pair semantics require this"
        )


def test_cache_keys_on_model_and_prompt(tmp_path):
    """Different (model, prompt) pairs must produce different cache entries."""
    from src.llm import LLMClient
    cache = tmp_path / "llm_cache.jsonl"

    c_7b = LLMClient(model="ollama/qwen2.5:7b", cache_path=cache)
    c_32b = LLMClient(model="ollama/qwen2.5:32b", cache_path=cache)

    with patch("litellm.completion") as mock:
        mock.return_value = _make_fake_litellm_response("from_7b")
        c_7b.invoke("p1")
        mock.return_value = _make_fake_litellm_response("from_32b")
        c_32b.invoke("p1")  # same prompt, different model
        mock.return_value = _make_fake_litellm_response("from_7b_p2")
        c_7b.invoke("p2")  # same model, different prompt

    lines = cache.read_text().splitlines()
    assert len(lines) == 3, f"expected 3 distinct cache entries, got {len(lines)}"

    # Verify the keys differ
    with patch("litellm.completion") as mock:
        # Each replay must hit cache (mock should never be called)
        r = c_7b.invoke("p1")
        assert r.text == "from_7b"
        r = c_32b.invoke("p1")
        assert r.text == "from_32b"
        r = c_7b.invoke("p2")
        assert r.text == "from_7b_p2"
        assert mock.call_count == 0, "all three replays must hit cache"


def test_no_cache_path_means_no_cache(tmp_path):
    """Without cache_path, repeated calls must invoke the API every time."""
    from src.llm import LLMClient
    c = LLMClient(model="ollama/qwen2.5:7b")  # no cache_path
    assert c.cache_path is None

    with patch("litellm.completion") as mock_completion:
        mock_completion.return_value = _make_fake_litellm_response("uncached")
        c.invoke("p")
        c.invoke("p")
        assert mock_completion.call_count == 2, (
            "without cache_path, both calls must hit the API"
        )


def test_cache_preserves_token_counts_and_latency_replay(tmp_path):
    """Cached LLMResponse must round-trip prompt_tokens, response_tokens,
    and a non-negative latency_s (for downstream cost/latency accounting)."""
    from src.llm import LLMClient
    cache = tmp_path / "llm_cache.jsonl"

    c1 = LLMClient(model="ollama/qwen2.5:7b", cache_path=cache)
    with patch("litellm.completion") as mock:
        mock.return_value = _make_fake_litellm_response("xyz", prompt_tokens=42, response_tokens=17)
        r1 = c1.invoke("prompt")

    c2 = LLMClient(model="ollama/qwen2.5:7b", cache_path=cache)
    with patch("litellm.completion") as mock:
        r2 = c2.invoke("prompt")
        assert mock.call_count == 0, "cache hit"
    assert r2.prompt_tokens == 42
    assert r2.response_tokens == 17
    assert r2.latency_s >= 0.0
    assert r2.text == "xyz"
    assert r2.model == "ollama/qwen2.5:7b"


def test_corrupt_cache_entry_raises(tmp_path):
    """L51: a corrupted cache file (malformed JSONL) must raise on first
    read, not silently return stale or wrong data."""
    from src.llm import LLMClient
    cache = tmp_path / "llm_cache.jsonl"
    cache.write_text("this is not json\n")
    c = LLMClient(model="ollama/qwen2.5:7b", cache_path=cache)
    # Trigger cache read by calling invoke. The cache implementation must
    # raise on corrupt JSONL rather than mask the bug.
    with pytest.raises((ValueError, json.JSONDecodeError)):
        c.invoke("anything")
