"""Thin wrappers around each LLM provider so the router can treat them uniformly.

Each provider implements a common interface:
    - .complete(messages, **kwargs) -> str    (returns text completion)
    - .name: str                              (identifier, e.g. 'groq-70b')
    - .is_available() -> bool                 (can we actually call this?)
"""
from __future__ import annotations

import abc
from typing import Any

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.config import get_settings


class LLMError(Exception):
    """Base error for LLM provider failures."""


class RateLimitError(LLMError):
    """Raised when we hit a provider's rate limit."""


class BaseProvider(abc.ABC):
    """Common interface all LLM providers must satisfy."""

    name: str = "base"

    @abc.abstractmethod
    def complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> str:
        """Return text completion for the given chat messages."""

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True if this provider can be called right now."""


# ---------------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------------
class GroqProvider(BaseProvider):
    """Groq LPU inference — primary cloud LLM."""

    def __init__(self, model: str | None = None) -> None:
        from groq import Groq
        settings = get_settings()
        self.model = model or settings.groq_model_heavy
        self.name = f"groq:{self.model}"
        self._client = Groq(api_key=settings.groq_api_key)
        self._configured = settings.groq_configured

    def is_available(self) -> bool:
        return self._configured

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(RateLimitError),
        reraise=True,
    )
    def complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            msg = str(e).lower()
            if "rate" in msg or "429" in msg:
                logger.warning(f"Groq rate limit hit: {e}")
                raise RateLimitError(str(e)) from e
            logger.error(f"Groq error: {e}")
            raise LLMError(str(e)) from e


# ---------------------------------------------------------------------------
# Gemini (Google AI Studio)
# ---------------------------------------------------------------------------
class GeminiProvider(BaseProvider):
    """Google Gemini — secondary cloud LLM, good for long context."""

    def __init__(self, model: str | None = None) -> None:
        import google.generativeai as genai
        settings = get_settings()
        self.model = model or settings.gemini_model
        self.name = f"gemini:{self.model}"
        self._configured = settings.gemini_configured
        if self._configured:
            genai.configure(api_key=settings.google_api_key)
            self._model = genai.GenerativeModel(self.model)
        else:
            self._model = None

    def is_available(self) -> bool:
        return self._configured

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(RateLimitError),
        reraise=True,
    )
    def complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> str:
        if self._model is None:
            raise LLMError("Gemini not configured")
        # Convert OpenAI-style messages to Gemini format
        prompt_parts: list[str] = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                prompt_parts.append(f"[System instructions]\n{content}")
            elif role == "user":
                prompt_parts.append(f"[User]\n{content}")
            elif role == "assistant":
                prompt_parts.append(f"[Assistant]\n{content}")
        prompt = "\n\n".join(prompt_parts)
        try:
            import google.generativeai as genai
            resp = self._model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                ),
            )
            return resp.text or ""
        except Exception as e:
            msg = str(e).lower()
            if "quota" in msg or "rate" in msg or "429" in msg:
                logger.warning(f"Gemini rate limit hit: {e}")
                raise RateLimitError(str(e)) from e
            logger.error(f"Gemini error: {e}")
            raise LLMError(str(e)) from e


# ---------------------------------------------------------------------------
# Ollama (offline fallback)
# ---------------------------------------------------------------------------
class OllamaProvider(BaseProvider):
    """Local Ollama — offline demo fallback. Tiny model, low quality, no internet needed."""

    def __init__(self, model: str | None = None) -> None:
        settings = get_settings()
        self.model = model or settings.ollama_model
        self.name = f"ollama:{self.model}"
        self._host = settings.ollama_host

    def is_available(self) -> bool:
        """Check if Ollama is reachable. Doesn't validate the model is pulled."""
        import httpx
        try:
            r = httpx.get(f"{self._host}/api/tags", timeout=2.0)
            return r.status_code == 200
        except Exception:  # noqa: BLE001
            return False

    def complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> str:
        import ollama
        try:
            client = ollama.Client(host=self._host)
            resp = client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            )
            return resp["message"]["content"]
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            raise LLMError(str(e)) from e