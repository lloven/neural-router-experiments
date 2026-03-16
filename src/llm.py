"""LLM client abstraction for Neural Router.

Uses LiteLLM for provider-agnostic LLM access. Supports OpenAI, Azure,
local models via vLLM/Ollama, etc.
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
    """Response from an LLM invocation."""
    text: str
    prompt_tokens: int
    response_tokens: int
    latency_s: float
    model: str
    raw_response: Optional[dict] = None


class LLMClient:
    """LLM client using LiteLLM."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._api_key = api_key
        self._total_invocations = 0
        self._total_prompt_tokens = 0
        self._total_response_tokens = 0
        self._total_latency = 0.0

    def invoke(self, prompt: str) -> LLMResponse:
        """Invoke the LLM with a prompt and return the response."""
        import litellm

        t0 = time.time()
        try:
            response = litellm.completion(
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
                raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
            )

        except Exception as e:
            latency = time.time() - t0
            logger.error(f"LLM invocation failed after {latency:.2f}s: {e}")
            raise

    @property
    def total_invocations(self) -> int:
        return self._total_invocations

    @property
    def total_cost(self) -> float:
        """Estimate cost based on GPT-4o-mini pricing (as of March 2026)."""
        # $0.15/1M input tokens, $0.60/1M output tokens
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
    """LLM client that simulates responses without API calls.

    Useful for testing the pipeline, validating configs, and estimating
    costs before running with real LLM calls.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._call_log: list[str] = []

    def invoke(self, prompt: str) -> LLMResponse:
        """Return a simulated empty response."""
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
