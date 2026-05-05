"""Tests for permissive parsing of LLM match responses.

The strict `json.loads(text.strip())` approach in the original code fails
silently on common verbose-LLM output patterns:

  * Llama 3.1 / Mistral often emit prose before/after the JSON, e.g.
        "Here's my analysis:\n```json\n{...}\n```\nHope this helps!"
  * Models sometimes omit the markdown fences but include a trailing note.
  * Models sometimes prefix with a "Sure, here are the matches:" line.

The permissive parser must extract the FIRST {...} block from the response
text and parse that, while preserving the existing happy-path behaviour
for clean JSON outputs (Qwen, Haiku, Sonnet — already verified).

TDD RED phase: tests written before implementation.
"""

from __future__ import annotations

import json

import pytest

from src.data import Event
from src.router import (
    NeuralRouter,
    RouterConfig,
    ABLATION_CONFIGS,
)
from src.llm import DryRunLLMClient


def _make_router() -> NeuralRouter:
    """Return a router instance suitable for parser-only testing."""
    config = RouterConfig(**ABLATION_CONFIGS["A0"].__dict__)
    config.llm_model = "ollama/qwen2.5:7b"  # any backend; we never invoke it
    llm = DryRunLLMClient(model=config.llm_model)
    # Embedder not needed for parser tests; pass None — _parse_match_response
    # only uses self for nothing.
    return NeuralRouter(config=config, llm_client=llm, embedding_model=None)


# ---------------------------------------------------------------------------
# Happy-path: existing behaviour must not regress
# ---------------------------------------------------------------------------


def test_parse_clean_json_unchanged():
    """Pure JSON (Qwen / Haiku / Sonnet behaviour) keeps working."""
    router = _make_router()
    events = [Event(id="e1", text="t", ground_truth=[]),
              Event(id="e2", text="t", ground_truth=[])]
    raw = '{"e1": ["s1", "s2"], "e2": ["s3"]}'

    results = router._parse_match_response(raw, events)
    assert {r.event_id: r.matched_subscription_ids for r in results} == {
        "e1": ["s1", "s2"],
        "e2": ["s3"],
    }


def test_parse_markdown_fenced_json_unchanged():
    """Markdown-fenced JSON (existing handling) keeps working."""
    router = _make_router()
    events = [Event(id="e1", text="t", ground_truth=[])]
    raw = '```json\n{"e1": ["s1"]}\n```'

    results = router._parse_match_response(raw, events)
    assert results[0].matched_subscription_ids == ["s1"]


# ---------------------------------------------------------------------------
# Permissive cases: prose before/after JSON (the Llama 3.1 failure mode)
# ---------------------------------------------------------------------------


def test_parse_with_prose_before():
    """Llama-style: 'Sure, here are the matches:' prefix before JSON."""
    router = _make_router()
    events = [Event(id="e1", text="t", ground_truth=[])]
    raw = "Sure, here are the matches you requested:\n\n" \
          '{"e1": ["s1", "s2"]}'

    results = router._parse_match_response(raw, events)
    assert results[0].matched_subscription_ids == ["s1", "s2"]


def test_parse_with_prose_after():
    """Llama-style: explanatory paragraph appended after JSON."""
    router = _make_router()
    events = [Event(id="e1", text="t", ground_truth=[])]
    raw = '{"e1": ["s1"]}\n\nI matched event e1 with subscription s1 ' \
          'because the topics align.'

    results = router._parse_match_response(raw, events)
    assert results[0].matched_subscription_ids == ["s1"]


def test_parse_with_prose_around_fenced_json():
    """Common Llama wrapping: prose + fenced JSON + prose."""
    router = _make_router()
    events = [Event(id="e1", text="t", ground_truth=[]),
              Event(id="e2", text="t", ground_truth=[])]
    raw = (
        "Here is my analysis of the events:\n\n"
        "```json\n"
        '{"e1": ["s1"], "e2": ["s2", "s3"]}\n'
        "```\n\n"
        "Hope this helps!"
    )

    results = router._parse_match_response(raw, events)
    out = {r.event_id: r.matched_subscription_ids for r in results}
    assert out == {"e1": ["s1"], "e2": ["s2", "s3"]}


def test_parse_handles_braces_in_inner_strings():
    """Don't be fooled by literal `{` `}` characters appearing inside string
    values — extract the OUTER object, not the first one we trip over.
    """
    router = _make_router()
    events = [Event(id="e1", text="t", ground_truth=[])]
    raw = (
        '{"e1": ["s1"], "_note": "the JSON contains balanced { and } chars"}'
    )

    results = router._parse_match_response(raw, events)
    assert results[0].matched_subscription_ids == ["s1"]


# ---------------------------------------------------------------------------
# Failure modes that must STILL raise (so the async path's empty-fallback
# kicks in): genuinely malformed payloads.
# ---------------------------------------------------------------------------


def test_parse_genuine_garbage_raises():
    """No JSON object anywhere in the text → raise."""
    router = _make_router()
    events = [Event(id="e1", text="t", ground_truth=[])]
    raw = "Sorry, I cannot help with that request."

    with pytest.raises((json.JSONDecodeError, ValueError)):
        router._parse_match_response(raw, events)


def test_parse_truncated_json_raises():
    """Partial JSON (LLM cut off mid-output) → raise."""
    router = _make_router()
    events = [Event(id="e1", text="t", ground_truth=[])]
    raw = '{"e1": ["s1", "s2'

    with pytest.raises((json.JSONDecodeError, ValueError)):
        router._parse_match_response(raw, events)
