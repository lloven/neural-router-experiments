"""Async LLM client for parallel invocations.

Uses asyncio + litellm's async API to run multiple LLM invocations
concurrently, dramatically reducing wall-clock time when matching
events across multiple clusters.
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
    """Async LLM client using litellm's async completion."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        api_key: Optional[str] = None,
        max_concurrent: int = 10,
        rate_limit_rpm: int = 500,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._api_key = api_key
        self._max_concurrent = max_concurrent
        self._rate_limit_rpm = rate_limit_rpm
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._total_invocations = 0
        self._total_prompt_tokens = 0
        self._total_response_tokens = 0
        self._total_latency = 0.0

    async def invoke_async(self, prompt: str) -> LLMResponse:
        """Invoke the LLM asynchronously."""
        import litellm

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
        """Invoke multiple prompts concurrently.

        Respects the max_concurrent semaphore to avoid rate limiting.
        """
        tasks = [self.invoke_async(p) for p in prompts]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def invoke_batch_sync(self, prompts: list[str]) -> list[LLMResponse]:
        """Synchronous wrapper for batch invocation.

        Creates or reuses an event loop to run async calls.
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
