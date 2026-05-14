"""
Async LLM client wrapper with retry logic.

Uses OpenAI-compatible client pointed at OpenRouter.
Provides specialised methods for chat, emotion analysis, profile extraction, etc.
"""

import asyncio
import json
import logging
import re
from openai import AsyncOpenAI

from config import (
    OPENROUTER_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_RETRY_ATTEMPTS,
    LLM_RETRY_BASE_DELAY,
)

logger = logging.getLogger(__name__)


class LLMClient:
    """Async LLM client with retry logic and structured output helpers."""

    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=LLM_BASE_URL,
        )

    async def _call_with_retry(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Make an LLM call with exponential backoff retry."""
        max_tokens = max_tokens or LLM_MAX_TOKENS
        temperature = temperature if temperature is not None else LLM_TEMPERATURE

        last_error = None
        for attempt in range(LLM_RETRY_ATTEMPTS):
            try:
                response = await self._client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                    temperature=temperature,
                )
                content = response.choices[0].message.content
                if content is None:
                    content = ""
                return content
            except Exception as exc:
                last_error = exc
                delay = LLM_RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt + 1, LLM_RETRY_ATTEMPTS, exc, delay,
                )
                await asyncio.sleep(delay)

        logger.error("LLM call failed after %d attempts: %s", LLM_RETRY_ATTEMPTS, last_error)
        raise last_error  # type: ignore[misc]

    # ── Chat ─────────────────────────────────────────────────────────────────────

    async def chat(
        self, system_prompt: str, user_message: str, max_tokens: int | None = None
    ) -> str:
        """Main conversational LLM call."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        return await self._call_with_retry(messages, max_tokens=max_tokens)

    # ── Structured JSON calls ────────────────────────────────────────────────────

    async def _call_json(
        self, prompt: str, max_tokens: int = 400, temperature: float = 0.3
    ) -> dict | None:
        """Make an LLM call expecting JSON output. Returns parsed dict or None."""
        messages = [
            {
                "role": "system",
                "content": "You are a precise JSON-only assistant. Respond with ONLY valid JSON, no markdown fences, no explanation.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            raw = await self._call_with_retry(messages, max_tokens=max_tokens, temperature=temperature)
            # Strip markdown code fences if present
            cleaned = raw.strip()
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse JSON from LLM response: %s\nRaw: %s", exc, raw[:200])
            return None
        except Exception as exc:
            logger.error("JSON LLM call failed: %s", exc)
            return None

    async def analyze_emotion(self, prompt: str) -> dict | None:
        """Analyze emotion from a formatted prompt. Returns structured result."""
        return await self._call_json(prompt, max_tokens=300, temperature=0.2)

    async def extract_profile(self, prompt: str) -> dict | None:
        """Extract profile updates from a formatted prompt."""
        return await self._call_json(prompt, max_tokens=500, temperature=0.2)

    async def generate_summary(self, prompt: str) -> str:
        """Generate a summary from a formatted prompt."""
        messages = [
            {
                "role": "system",
                "content": "You are a concise, insightful summarizer. Write naturally and specifically.",
            },
            {"role": "user", "content": prompt},
        ]
        return await self._call_with_retry(messages, max_tokens=600, temperature=0.5)


# ── Module-level singleton ───────────────────────────────────────────────────────

_llm_instance: LLMClient | None = None


def get_llm() -> LLMClient:
    """Get the singleton LLMClient instance."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMClient()
    return _llm_instance
