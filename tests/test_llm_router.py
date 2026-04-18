"""Tests for the LLM router. Uses mocks — does not call real APIs."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_task_type_enum_complete():
    """Every TaskType must have a routing entry and temperature."""
    from src.llm.router import TASK_ROUTING, TASK_TEMPERATURE, TaskType
    for task in TaskType:
        assert task in TASK_ROUTING, f"Missing routing for {task}"
        assert task in TASK_TEMPERATURE, f"Missing temperature for {task}"
        assert len(TASK_ROUTING[task]) > 0, f"Empty routing for {task}"


def test_router_picks_groq_fast_for_extraction():
    """Skill extraction should prefer the fast Groq model."""
    from src.llm.router import LLMRouter, TaskType
    router = LLMRouter()
    # Only run this assertion if Groq is configured — skip gracefully otherwise
    if "groq_fast" not in router._providers:
        pytest.skip("Groq not configured; routing test requires at least one provider")
    picked = router.pick_provider(TaskType.SKILL_EXTRACTION)
    # Should be groq with the fast model (contains 'instant')
    assert "groq" in picked.lower() and "instant" in picked.lower()


def test_router_picks_gemini_for_long_context():
    """Report synthesis prefers Gemini (1M TPM)."""
    from src.llm.router import LLMRouter, TaskType
    router = LLMRouter()
    if "gemini" not in router._providers:
        pytest.skip("Gemini not configured")
    picked = router.pick_provider(TaskType.REPORT_SYNTHESIS)
    assert "gemini" in picked.lower()


def test_router_falls_back_on_rate_limit():
    """If groq_fast rate-limits, should try next provider."""
    from src.llm.providers import RateLimitError
    from src.llm.router import LLMRouter, TaskType
    router = LLMRouter()

    # Force groq_fast to raise RateLimitError
    if "groq_fast" in router._providers:
        mock_groq = MagicMock()
        mock_groq.is_available.return_value = True
        mock_groq.complete.side_effect = RateLimitError("429")
        mock_groq.name = "groq:mocked"
        router._providers["groq_fast"] = mock_groq

    # Mock gemini to succeed
    if "gemini" in router._providers:
        mock_gem = MagicMock()
        mock_gem.is_available.return_value = True
        mock_gem.complete.return_value = "fallback worked"
        mock_gem.name = "gemini:mocked"
        router._providers["gemini"] = mock_gem

        result = router.complete(
            task=TaskType.SKILL_EXTRACTION,
            messages=[{"role": "user", "content": "test"}],
        )
        assert result == "fallback worked"
        mock_groq.complete.assert_called_once()
        mock_gem.complete.assert_called_once()
    else:
        pytest.skip("Gemini not configured; can't test fallback")


def test_available_providers_returns_list():
    from src.llm.router import LLMRouter
    router = LLMRouter()
    avail = router.available_providers()
    assert isinstance(avail, list)
    # At minimum, router should know about its registered providers
    assert len(router._providers) > 0