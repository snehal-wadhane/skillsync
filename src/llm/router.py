"""Task-aware LLM router.

Different tasks have different quality/volume/latency tradeoffs. The router
maps a `TaskType` to the right provider+model combination.

Usage:
    router = LLMRouter()
    text = router.complete(
        task=TaskType.SKILL_EXTRACTION,
        messages=[{"role": "user", "content": "..."}],
    )

The router automatically falls back from heavy→fast→offline if the primary
provider is rate-limited or unavailable.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from loguru import logger

from src.config import get_settings
from src.llm.providers import (
    BaseProvider,
    GeminiProvider,
    GroqProvider,
    LLMError,
    OllamaProvider,
    RateLimitError,
)


class TaskType(str, Enum):
    """Types of LLM tasks in SkillSync. Each maps to a specific provider profile."""

    # High-volume, short-output tasks — use fastest cheap model
    SKILL_EXTRACTION = "skill_extraction"           # Extract skills from a chunk of text
    CLASSIFICATION = "classification"                # Classify into a small label set
    METADATA_EXTRACTION = "metadata_extraction"      # Pull course code, credits, etc. from syllabus

    # Mid-complexity reasoning
    BLOOM_CLASSIFICATION = "bloom_classification"    # Ambiguous Bloom's verb cases
    SKILL_ALIASING = "skill_aliasing"                # "Is 'ML' the same as 'Machine Learning'?"

    # High-complexity reasoning — use heavy model
    GAP_ANALYSIS = "gap_analysis"                    # Reasoning over retrieved context
    MULTI_AGENT_DEBATE = "multi_agent_debate"        # Agent arguments
    RECOMMENDATION = "recommendation"                # Curriculum change proposals
    CO_PO_MAPPING = "co_po_mapping"                  # Map COs to POs with justification

    # Long-context tasks — use Gemini (1M TPM)
    REPORT_SYNTHESIS = "report_synthesis"            # Assemble final report
    LONG_DOC_SUMMARY = "long_doc_summary"            # Summarize a full industry report


# ---------------------------------------------------------------------------
# Task → provider routing
# ---------------------------------------------------------------------------
# Each task lists providers in priority order. Router tries them in turn.
TASK_ROUTING: dict[TaskType, list[str]] = {
    TaskType.SKILL_EXTRACTION:     ["groq_fast", "gemini", "ollama"],
    TaskType.CLASSIFICATION:       ["groq_fast", "gemini", "ollama"],
    TaskType.METADATA_EXTRACTION:  ["groq_fast", "gemini", "ollama"],
    TaskType.BLOOM_CLASSIFICATION: ["groq_fast", "gemini", "ollama"],
    TaskType.SKILL_ALIASING:       ["groq_fast", "gemini", "ollama"],
    TaskType.GAP_ANALYSIS:         ["groq_heavy", "gemini", "ollama"],
    TaskType.MULTI_AGENT_DEBATE:   ["groq_heavy", "gemini", "ollama"],
    TaskType.RECOMMENDATION:       ["groq_heavy", "gemini", "ollama"],
    TaskType.CO_PO_MAPPING:        ["groq_heavy", "gemini", "ollama"],
    TaskType.REPORT_SYNTHESIS:     ["gemini", "groq_heavy", "ollama"],  # Gemini first — long context
    TaskType.LONG_DOC_SUMMARY:     ["gemini", "groq_heavy", "ollama"],
}

# Suggested temperature per task (lower = more deterministic)
TASK_TEMPERATURE: dict[TaskType, float] = {
    TaskType.SKILL_EXTRACTION:     0.0,
    TaskType.CLASSIFICATION:       0.0,
    TaskType.METADATA_EXTRACTION:  0.0,
    TaskType.BLOOM_CLASSIFICATION: 0.0,
    TaskType.SKILL_ALIASING:       0.0,
    TaskType.GAP_ANALYSIS:         0.2,
    TaskType.MULTI_AGENT_DEBATE:   0.5,  # Want some diversity between agents
    TaskType.RECOMMENDATION:       0.3,
    TaskType.CO_PO_MAPPING:        0.1,
    TaskType.REPORT_SYNTHESIS:     0.3,
    TaskType.LONG_DOC_SUMMARY:     0.2,
}


class LLMRouter:
    """Routes tasks to appropriate LLM providers with automatic fallback."""

    def __init__(self) -> None:
        settings = get_settings()
        self._providers: dict[str, BaseProvider] = {}

        # Instantiate providers lazily — but track which are configured
        if settings.groq_configured:
            self._providers["groq_heavy"] = GroqProvider(model=settings.groq_model_heavy)
            self._providers["groq_fast"] = GroqProvider(model=settings.groq_model_fast)
        if settings.gemini_configured:
            self._providers["gemini"] = GeminiProvider(model=settings.gemini_model)
        # Ollama always registered — is_available() determines if it actually works
        self._providers["ollama"] = OllamaProvider()

        logger.info(f"LLM router initialized with providers: {list(self._providers.keys())}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def pick_provider(self, task: TaskType) -> str:
        """Return the name of the provider that would be used for this task.

        Useful for debugging / observability. Does not actually call the LLM.
        """
        for name in TASK_ROUTING[task]:
            if name in self._providers and self._providers[name].is_available():
                return self._providers[name].name
        raise LLMError(
            f"No available provider for task {task}. "
            f"Check your .env — set GROQ_API_KEY at minimum."
        )

    def complete(
        self,
        task: TaskType,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """Run completion, falling back through providers on rate limit / error."""
        temp = temperature if temperature is not None else TASK_TEMPERATURE[task]
        priority = TASK_ROUTING[task]
        last_error: Exception | None = None

        for name in priority:
            if name not in self._providers:
                continue
            provider = self._providers[name]
            if not provider.is_available():
                logger.debug(f"Provider {name} not available for task {task}, skipping")
                continue
            try:
                logger.debug(f"Task {task.value} → {provider.name}")
                return provider.complete(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temp,
                    **kwargs,
                )
            except RateLimitError as e:
                logger.warning(f"{provider.name} rate-limited, falling back. Err: {e}")
                last_error = e
                continue
            except LLMError as e:
                logger.error(f"{provider.name} failed: {e}")
                last_error = e
                continue

        raise LLMError(
            f"All providers failed for task {task}. Last error: {last_error}"
        )

    def available_providers(self) -> list[str]:
        """List providers that can currently be called."""
        return [p.name for p in self._providers.values() if p.is_available()]