"""Synchronous LLM client abstraction for the Neural Router.

Uses LiteLLM for provider-agnostic LLM access, supporting OpenAI, Azure,
local models via vLLM/Ollama, and other providers. This module provides:

  LLMResponse     -- dataclass wrapping a single LLM completion result
  LLMClient       -- synchronous client with usage tracking and cost estimation
  DryRunLLMClient -- mock client that simulates calls without hitting any API
                     (useful for cost estimation, pipeline validation, and tests)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from a single LLM invocation.

    Attributes:
        text: The generated completion text.
        prompt_tokens: Number of tokens in the prompt (from API usage data).
        response_tokens: Number of tokens in the completion.
        latency_s: Wall-clock time for this call in seconds.
        model: Model identifier that produced the response.
        raw_response: Full provider response dict (for debugging).
    """
    text: str
    prompt_tokens: int
    response_tokens: int
    latency_s: float
    model: str
    raw_response: Optional[dict] = None


class LLMClient:
    """Synchronous LLM client using LiteLLM.

    Tracks cumulative invocation count, token usage, latency, and estimated
    monetary cost across all calls. Call ``reset_stats()`` between experiment
    runs to isolate per-run accounting.

    Args:
        model: LiteLLM model identifier (e.g., "gpt-4o-mini", "ollama/llama3").
        temperature: Sampling temperature (0.0 for deterministic).
        max_tokens: Maximum tokens in the generated completion.
        api_key: Optional API key override (otherwise uses env vars).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
        cache_path: Optional["Path | str"] = None,
    ):
        self.model = model
        self.timeout = timeout  # per-request timeout in seconds (None = litellm default)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._api_key = api_key
        # Optional on-disk LLM-call cache for matched-pair experiments.
        # Keyed by (model, prompt_hash). When set, repeated invoke() calls
        # with the same (model, prompt) return the cached response without
        # invoking the API. This neutralises LLM-nondeterminism between
        # SLURM jobs so baseline-vs-perturbed comparisons measure the
        # intended treatment, not GPU/Ollama jitter.
        from pathlib import Path
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self._cache: Optional[dict] = None  # lazy-loaded
        self._total_invocations = 0
        self._total_prompt_tokens = 0
        self._total_response_tokens = 0
        self._total_latency = 0.0

    def _cache_key(self, prompt: str) -> str:
        """Stable hash of (model, prompt) — sha1 hex of UTF-8 bytes."""
        import hashlib
        h = hashlib.sha1()
        h.update(self.model.encode("utf-8"))
        h.update(b"\x00")
        h.update(prompt.encode("utf-8"))
        return h.hexdigest()

    def _load_cache(self) -> dict:
        """Read the JSONL cache file into an in-memory dict (model_prompt_hash → entry).
        A corrupt cache file raises rather than silently masking stale data.
        """
        import json
        if self.cache_path is None:
            return {}
        if not self.cache_path.exists():
            return {}
        cache: dict = {}
        for lineno, line in enumerate(self.cache_path.read_text().splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)  # propagate JSONDecodeError rather than silently swallowing
            cache[entry["prompt_hash"]] = entry
        return cache

    def _write_cache_entry(self, entry: dict) -> None:
        """Append a single JSONL entry to the cache file."""
        import json
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def cache_lookup(self, prompt: str) -> Optional[LLMResponse]:
        """Public cache lookup. Returns None on miss; LLMResponse on hit.

        Used by both the sync invoke() path and the router's async path
        (src/router.py:_async_match_all_clusters) so the LLM-call cache
        protects determinism regardless of the execution model.
        """
        if self.cache_path is None:
            return None
        if self._cache is None:
            self._cache = self._load_cache()
        entry = self._cache.get(self._cache_key(prompt))
        if entry is None:
            return None
        return LLMResponse(
            text=entry["text"],
            prompt_tokens=entry["prompt_tokens"],
            response_tokens=entry["response_tokens"],
            latency_s=entry.get("latency_s", 0.0),
            model=entry["model"],
            raw_response=None,
        )

    def cache_write(
        self,
        prompt: str,
        text: str,
        prompt_tokens: int,
        response_tokens: int,
        latency_s: float,
    ) -> None:
        """Public cache writer. No-op when cache_path is None."""
        if self.cache_path is None:
            return
        entry = {
            "model": self.model,
            "prompt_hash": self._cache_key(prompt),
            "text": text,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "latency_s": latency_s,
        }
        self._write_cache_entry(entry)
        if self._cache is not None:
            self._cache[entry["prompt_hash"]] = entry

    def invoke(self, prompt: str, max_retries: int = 5) -> LLMResponse:
        """Invoke the LLM with a single-turn prompt.

        Args:
            prompt: The user-message content to send.
            max_retries: Max retries on rate-limit or overloaded errors.

        Returns:
            An LLMResponse with the completion text and usage metadata.

        Raises:
            Exception: Propagated from LiteLLM on API errors.
        """
        import litellm

        # Cache check before API. cache_lookup is shared with the
        # router's async path — sync and async share one cache.
        cached = self.cache_lookup(prompt)
        if cached is not None:
            self._total_invocations += 1
            self._total_prompt_tokens += cached.prompt_tokens
            self._total_response_tokens += cached.response_tokens
            return cached

        for attempt in range(max_retries):
            t0 = time.time()
            try:
                kwargs = dict(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    api_key=self._api_key,
                )
                if self.timeout is not None:
                    kwargs["timeout"] = self.timeout
                response = litellm.completion(**kwargs)

                text = response.choices[0].message.content or ""
                prompt_tokens = response.usage.prompt_tokens if response.usage else 0
                response_tokens = response.usage.completion_tokens if response.usage else 0
                latency = time.time() - t0

                self._total_invocations += 1
                self._total_prompt_tokens += prompt_tokens
                self._total_response_tokens += response_tokens
                self._total_latency += latency

                # Write cache entry via the shared public helper so
                # subsequent matched-pair runs (sync OR async) hit cache.
                self.cache_write(prompt, text, prompt_tokens, response_tokens, latency)

                return LLMResponse(
                    text=text,
                    prompt_tokens=prompt_tokens,
                    response_tokens=response_tokens,
                    latency_s=latency,
                    model=self.model,
                    raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
                )

            except Exception as e:
                latency = time.time() - t0
                err_str = str(e).lower()
                # Fatal errors (billing/auth) should never be retried
                fatal_patterns = ["usage limit", "invalid_api_key", "authentication", "permission", "not_found_error"]
                if any(p in err_str for p in fatal_patterns):
                    logger.error(f"Fatal API error (not retrying): {e}")
                    raise
                is_retryable = "rate_limit" in err_str or "429" in str(e) or "overloaded" in err_str
                if is_retryable and attempt < max_retries - 1:
                    wait = 2 ** attempt * 5  # 5s, 10s, 20s, 40s
                    logger.warning(f"Retryable error, waiting {wait}s (attempt {attempt+1}/{max_retries}): {e}")
                    time.sleep(wait)
                    continue
                logger.error(f"LLM invocation failed after {latency:.2f}s: {e}")
                raise

    @property
    def total_invocations(self) -> int:
        return self._total_invocations

    @property
    def total_cost(self) -> float:
        """Estimate cumulative cost based on GPT-4o-mini pricing.

        Pricing as of March 2026: $0.15/1M input tokens, $0.60/1M output tokens.
        """
        input_cost = self._total_prompt_tokens * 0.15 / 1_000_000
        output_cost = self._total_response_tokens * 0.60 / 1_000_000
        return input_cost + output_cost

    def reset_stats(self) -> None:
        self._total_invocations = 0
        self._total_prompt_tokens = 0
        self._total_response_tokens = 0
        self._total_latency = 0.0

    def stats_summary(self) -> str:
        return (
            f"LLM Stats: {self._total_invocations} invocations, "
            f"{self._total_prompt_tokens} prompt tokens, "
            f"{self._total_response_tokens} response tokens, "
            f"${self.total_cost:.4f} est. cost, "
            f"{self._total_latency:.1f}s total latency"
        )


class DryRunLLMClient(LLMClient):
    """Mock LLM client that returns empty JSON without making API calls.

    Useful for:
      - Validating the pipeline end-to-end without spending tokens.
      - Estimating invocation counts and token budgets via RouterStats.
      - Unit-testing prompt construction and response parsing.

    All calls are logged (first 200 chars) in ``call_log`` for inspection.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._call_log: list[str] = []

    def invoke(self, prompt: str) -> LLMResponse:
        """Return a simulated empty JSON response (no API call).

        Token counts are estimated from prompt length (4 chars per token).

        Args:
            prompt: The prompt that would be sent to the LLM.

        Returns:
            LLMResponse with text='{}' and estimated token counts.
        """
        t0 = time.time()
        self._call_log.append(prompt[:200])  # log first 200 chars

        # Estimate token counts from prompt length
        prompt_tokens = len(prompt) // 4
        response_tokens = 100  # placeholder

        self._total_invocations += 1
        self._total_prompt_tokens += prompt_tokens
        self._total_response_tokens += response_tokens
        self._total_latency += time.time() - t0

        # Return empty but valid JSON
        return LLMResponse(
            text='{}',
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            latency_s=time.time() - t0,
            model=f"dry-run({self.model})",
        )

    @property
    def call_log(self) -> list[str]:
        return self._call_log
