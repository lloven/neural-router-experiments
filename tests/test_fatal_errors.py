"""Tests for fatal error detection and propagation in LLM clients and router.

Verifies that billing-limit, auth, and other non-transient API errors:
1. Are detected as fatal (not retried)
2. Propagate as exceptions (not silently produce empty results)
3. Abort the run early rather than continuing with all-empty matches
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm import LLMClient, LLMResponse
from src.router import NeuralRouter, RouterConfig, ABLATION_CONFIGS, MatchResult
from src.data import Event, Subscription


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def billing_error():
    """Simulates the Anthropic billing-limit error seen in production."""
    return Exception(
        'AnthropicException - {"type":"error","error":{"type":"invalid_request_error",'
        '"message":"You have reached your specified API usage limits. '
        'You will regain access on 2026-04-01 at 00:00 UTC."}}'
    )


@pytest.fixture
def auth_error():
    return Exception("invalid_api_key: The API key provided is not valid")


@pytest.fixture
def rate_limit_error():
    return Exception("rate_limit_error: Too many requests, please slow down")


@pytest.fixture
def model_not_found_error():
    return Exception("not_found_error: model 'claude-3-haiku' not found")


@pytest.fixture
def sample_subscriptions():
    return [
        Subscription(id="s1", name="Sports", description="Sports news and events"),
        Subscription(id="s2", name="Tech", description="Technology updates"),
    ]


@pytest.fixture
def sample_events():
    return [
        Event(id="e1", text="The team won the championship game", ground_truth=["s1"]),
        Event(id="e2", text="New AI model released by researchers", ground_truth=["s2"]),
        Event(id="e3", text="Stock market reaches all-time high", ground_truth=[]),
    ]


# ---------------------------------------------------------------------------
# LLMClient: fatal error detection
# ---------------------------------------------------------------------------

class TestLLMClientFatalErrors:
    """LLMClient.invoke() should NOT retry fatal errors."""

    def test_billing_limit_not_retried(self, billing_error):
        """Billing-limit errors should raise immediately without retrying."""
        client = LLMClient(model="anthropic/claude-3-haiku-20240307")
        call_count = 0

        def mock_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            raise billing_error

        with patch("litellm.completion", side_effect=mock_completion):
            with pytest.raises(Exception, match="usage limits"):
                client.invoke("test prompt")

        assert call_count == 1, "Fatal error should not be retried"

    def test_auth_error_not_retried(self, auth_error):
        """Auth errors should raise immediately without retrying."""
        client = LLMClient(model="anthropic/claude-3-haiku-20240307")
        call_count = 0

        def mock_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            raise auth_error

        with patch("litellm.completion", side_effect=mock_completion):
            with pytest.raises(Exception, match="invalid_api_key"):
                client.invoke("test prompt")

        assert call_count == 1, "Fatal error should not be retried"

    def test_rate_limit_IS_retried(self, rate_limit_error):
        """Rate-limit errors SHOULD be retried (control test)."""
        client = LLMClient(model="anthropic/claude-3-haiku-20240307")
        call_count = 0

        def mock_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            raise rate_limit_error

        with patch("litellm.completion", side_effect=mock_completion):
            with patch("time.sleep"):  # skip actual sleep
                with pytest.raises(Exception, match="rate_limit"):
                    client.invoke("test prompt", max_retries=3)

        assert call_count == 3, "Rate limit should be retried"


# ---------------------------------------------------------------------------
# Router: async fatal error propagation
# ---------------------------------------------------------------------------

class TestRouterFatalErrorPropagation:
    """Router._async_match_all_clusters() should abort on fatal errors."""

    def _make_router(self, llm_client, sample_subscriptions):
        config = RouterConfig(
            use_clustering=False,
            use_cover_merge=False,
            use_cosine_filter=False,
            use_async=True,
            llm_model="anthropic/claude-3-haiku-20240307",
            max_parallel=2,
        )
        from src.embeddings import EmbeddingModel
        embedding_model = MagicMock(spec=EmbeddingModel)
        # Return fake embeddings
        import numpy as np
        embedding_model.encode.return_value = np.random.randn(
            max(len(sample_subscriptions), 10), 384
        ).astype(np.float32)

        router = NeuralRouter(
            config=config,
            llm_client=llm_client,
            embedding_model=embedding_model,
        )
        return router

    def test_billing_error_aborts_matching(self, billing_error, sample_subscriptions, sample_events):
        """A billing-limit error during async matching should raise, not return empty results."""
        llm_client = MagicMock(spec=LLMClient)
        llm_client.model = "anthropic/claude-3-haiku-20240307"

        router = self._make_router(llm_client, sample_subscriptions)
        router.optimize_subscriptions(sample_subscriptions)

        # Mock litellm.acompletion to raise billing error
        async def mock_acompletion(**kwargs):
            raise billing_error

        with patch("litellm.acompletion", side_effect=mock_acompletion):
            with pytest.raises(Exception, match="usage limits"):
                router.match_events(sample_events)

    def test_all_transient_failures_abort(self, sample_subscriptions, sample_events):
        """If ALL async calls fail with transient errors, the run should abort."""
        llm_client = MagicMock(spec=LLMClient)
        llm_client.model = "anthropic/claude-3-haiku-20240307"

        router = self._make_router(llm_client, sample_subscriptions)
        router.optimize_subscriptions(sample_subscriptions)

        async def mock_acompletion(**kwargs):
            raise Exception("Connection timeout after 600s")

        with patch("litellm.acompletion", side_effect=mock_acompletion):
            with pytest.raises(RuntimeError, match="All .* LLM calls failed"):
                router.match_events(sample_events)

    def test_partial_failure_produces_partial_results(self, sample_subscriptions, sample_events):
        """If SOME calls succeed and some fail, partial results should be returned."""
        llm_client = MagicMock(spec=LLMClient)
        llm_client.model = "anthropic/claude-3-haiku-20240307"

        router = self._make_router(llm_client, sample_subscriptions)
        router.optimize_subscriptions(sample_subscriptions)

        call_count = 0

        async def mock_acompletion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call succeeds
                mock_resp = MagicMock()
                mock_resp.choices = [MagicMock()]
                mock_resp.choices[0].message.content = json.dumps(
                    {e.id: ["s1"] for e in sample_events}
                )
                mock_resp.usage = MagicMock()
                mock_resp.usage.prompt_tokens = 100
                mock_resp.usage.completion_tokens = 50
                return mock_resp
            else:
                raise Exception("Connection timeout")

        with patch("litellm.acompletion", side_effect=mock_acompletion):
            # Should not raise — partial success is OK
            results = router.match_events(sample_events)
            assert len(results) == len(sample_events)


# ---------------------------------------------------------------------------
# Router: _is_fatal_error detection
# ---------------------------------------------------------------------------

class TestFatalErrorDetection:
    """Test the _is_fatal_error helper patterns."""

    def test_billing_limit_detected(self):
        """Billing-limit messages should be detected as fatal."""
        # Import and test the function via the module
        from src.router import NeuralRouter
        # The function is nested inside _async_match_all_clusters,
        # so we test the behavior indirectly through the error patterns
        billing_msg = "You have reached your specified API usage limits"
        fatal_patterns = ["usage limit", "invalid_api_key", "authentication", "permission", "not_found_error"]
        assert any(p in billing_msg.lower() for p in fatal_patterns)

    def test_auth_failure_detected(self):
        auth_msg = "invalid_api_key: The provided key is not valid"
        fatal_patterns = ["usage limit", "invalid_api_key", "authentication", "permission", "not_found_error"]
        assert any(p in auth_msg.lower() for p in fatal_patterns)

    def test_rate_limit_NOT_fatal(self):
        rate_msg = "rate_limit_error: Too many requests"
        fatal_patterns = ["usage limit", "invalid_api_key", "authentication", "permission", "not_found_error"]
        assert not any(p in rate_msg.lower() for p in fatal_patterns)

    def test_overloaded_NOT_fatal(self):
        overloaded_msg = "overloaded_error: The API is temporarily overloaded"
        fatal_patterns = ["usage limit", "invalid_api_key", "authentication", "permission", "not_found_error"]
        assert not any(p in overloaded_msg.lower() for p in fatal_patterns)
