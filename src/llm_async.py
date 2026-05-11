"""Async LLM client for parallel invocations.

Uses asyncio + litellm's async API to run multiple LLM invocations
concurrently, dramatically reducing wall-clock time when matching
events across multiple clusters.

The concurrency degree is bounded by a semaphore (max_concurrent) to
stay within provider rate limits. This module is complementary to the
inline async path in ``router.py._run_async_matching()``, which uses
litellm directly; AsyncLLMClient provides a standalone reusable client
with its own usage tracking.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

from .llm import LLMResponse

logger = logging.getLogger(__name__)


class AsyncLLMClient:
    """Async LLM client using litellm's async completion.

    Provides both async (``invoke_async``, ``invoke_batch_async``) and sync
    (``invoke_batch_sync``) interfaces, all backed by asyncio concurrency.

    Args:
        model: LiteLLM model identifier.
        temperature: Sampling temperature (0.0 for deterministic).
        max_tokens: Maximum tokens in the generated completion.
        api_key: Optional API key override.
        max_concurrent: Maximum number of concurrent LLM calls (semaphore bound).
        rate_limit_rpm: Informational rate limit (not enforced in code yet).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        api_key: Optional[str] = None,
        max_concurrent: int = 10,
        rate_limit_rpm: int = 500,
        cache_path: Optional["Path | str"] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._api_key = api_key
        self._max_concurrent = max_concurrent
        self._rate_limit_rpm = rate_limit_rpm
        self._semaphore: Optional[asyncio.Semaphore] = None
        # Shared LLM-call cache (sync + async). The cache file format
        # is identical to LLMClient's (JSONL, sha1(model || prompt) keying)
        # so sync and async clients can read each other's writes — single
        # source of truth for matched-pair determinism.
        from pathlib import Path
        self.cache_path = Path(cache_path) if cache_path is not None else None
        # Reuse LLMClient's cache helpers via composition rather than
        # duplicating logic (DRY): a hidden LLMClient-shaped object holds
        # the cache and exposes cache_lookup / cache_write.
        from .llm import LLMClient
        self._cache_helper = LLMClient(model=model, cache_path=cache_path)
        self._total_invocations = 0
        self._total_prompt_tokens = 0
        self._total_response_tokens = 0
        self._total_latency = 0.0

    def cache_lookup(self, prompt: str):
        """Delegate to the underlying LLMClient cache helper."""
        return self._cache_helper.cache_lookup(prompt)

    def cache_write(self, prompt: str, text: str, prompt_tokens: int,
                    response_tokens: int, latency_s: float) -> None:
        """Delegate to the underlying LLMClient cache helper."""
        self._cache_helper.cache_write(prompt, text, prompt_tokens, response_tokens, latency_s)

    async def invoke_async(self, prompt: str) -> LLMResponse:
        """Invoke the LLM asynchronously (single prompt).

        Acquires the concurrency semaphore before calling the API.

        Args:
            prompt: The user-message content to send.

        Returns:
            An LLMResponse with the completion text and usage metadata.
        """
        import litellm

        # Cache lookup before semaphore acquisition.
        cached = self.cache_lookup(prompt)
        if cached is not None:
            self._total_invocations += 1
            self._total_prompt_tokens += cached.prompt_tokens
            self._total_response_tokens += cached.response_tokens
            return cached

        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrent)

        async with self._semaphore:
            t0 = time.time()
            try:
                response = await litellm.acompletion(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    api_key=self._api_key,
                )

                text = response.choices[0].message.content or ""
                prompt_tokens = response.usage.prompt_tokens if response.usage else 0
                response_tokens = response.usage.completion_tokens if response.usage else 0
                latency = time.time() - t0

                self._total_invocations += 1
                self._total_prompt_tokens += prompt_tokens
                self._total_response_tokens += response_tokens
                self._total_latency += latency

                # Write through to the shared cache for matched-pair runs.
                self.cache_write(prompt, text, prompt_tokens, response_tokens, latency)

                return LLMResponse(
                    text=text,
                    prompt_tokens=prompt_tokens,
                    response_tokens=response_tokens,
                    latency_s=latency,
                    model=self.model,
                )

            except Exception as e:
                latency = time.time() - t0
                logger.error(f"Async LLM invocation failed after {latency:.2f}s: {e}")
                raise

    async def invoke_batch_async(self, prompts: list[str]) -> list[LLMResponse]:
        """Invoke multiple prompts concurrently via asyncio.gather.

        Respects the max_concurrent semaphore to avoid rate limiting.
        Failed calls are returned as exceptions in the result list.

        Args:
            prompts: List of prompt strings to invoke in parallel.

        Returns:
            List of LLMResponse objects (or BaseException for failed calls).
        """
        tasks = [self.invoke_async(p) for p in prompts]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def invoke_batch_sync(self, prompts: list[str]) -> list[LLMResponse]:
        """Synchronous wrapper for batch invocation.

        Creates or reuses an event loop to run async calls. If an event loop
        is already running (e.g., inside Jupyter), falls back to executing
        in a separate thread via ThreadPoolExecutor.

        Args:
            prompts: List of prompt strings to invoke in parallel.

        Returns:
            List of LLMResponse objects (or BaseException for failed calls).
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in an async context, use a new loop in a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.invoke_batch_async(prompts))
                    return future.result()
            else:
                return loop.run_until_complete(self.invoke_batch_async(prompts))
        except RuntimeError:
            return asyncio.run(self.invoke_batch_async(prompts))

    @property
    def total_invocations(self) -> int:
        return self._total_invocations

    @property
    def total_cost(self) -> float:
        """Estimate cumulative cost (GPT-4o-mini pricing, March 2026)."""
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
            f"AsyncLLM Stats: {self._total_invocations} invocations, "
            f"{self._total_prompt_tokens} prompt tokens, "
            f"{self._total_response_tokens} response tokens, "
            f"${self.total_cost:.4f} est. cost, "
            f"{self._total_latency:.1f}s total latency"
        )
