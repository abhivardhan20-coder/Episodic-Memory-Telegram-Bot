"""
Utility classes and functions for the AI Diary Companion.
Includes LLM client with retry logic and summarization helpers.
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from openai import AsyncOpenAI

from app.config import (
    OPENROUTER_API_KEY, LLM_BASE_URL, LLM_MODEL, 
    LLM_MAX_TOKENS, LLM_TEMPERATURE, LLM_RETRY_ATTEMPTS, 
    LLM_RETRY_BASE_DELAY
)
from app.prompts import DAILY_SUMMARY_PROMPT, WEEKLY_SUMMARY_PROMPT, MONTHLY_SUMMARY_PROMPT

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
                return response.choices[0].message.content or ""
            except Exception as exc:
                last_error = exc
                delay = LLM_RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning("LLM call failed (attempt %d/%d): %s. Retrying...", attempt + 1, LLM_RETRY_ATTEMPTS, exc)
                await asyncio.sleep(delay)

        logger.error("LLM call failed after %d attempts: %s", LLM_RETRY_ATTEMPTS, last_error)
        raise last_error

    async def chat(self, system_prompt: str, user_message: str, max_tokens: int | None = None) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        return await self._call_with_retry(messages, max_tokens=max_tokens)

    async def _call_json(self, prompt: str, max_tokens: int = 400, temperature: float = 0.3) -> dict | None:
        messages = [
            {"role": "system", "content": "You are a precise JSON-only assistant. Respond with ONLY valid JSON, no markdown fences."},
            {"role": "user", "content": prompt},
        ]
        try:
            raw = await self._call_with_retry(messages, max_tokens=max_tokens, temperature=temperature)
            cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            cleaned = re.sub(r"\s*```$", "", cleaned)
            return json.loads(cleaned)
        except Exception as exc:
            logger.warning("Failed to parse JSON: %s", exc)
            return None

    async def analyze_emotion(self, prompt: str) -> dict | None:
        return await self._call_json(prompt, max_tokens=300, temperature=0.2)

    async def extract_profile(self, prompt: str) -> dict | None:
        return await self._call_json(prompt, max_tokens=500, temperature=0.2)

    async def generate_summary(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are a concise, insightful summarizer."},
            {"role": "user", "content": prompt},
        ]
        return await self._call_with_retry(messages, max_tokens=600, temperature=0.5)

_llm_instance: LLMClient | None = None

def get_llm() -> LLMClient:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMClient()
    return _llm_instance

# --- Summarization Logic (Moved from summarizer.py) ---

def format_episodes_for_summary(episodes: list[dict]) -> str:
    if not episodes: return "(no conversations)"
    lines = []
    for ep in episodes:
        ts = ep.get("timestamp", "")[:16].replace("T", " ")
        emotion = ep.get("detected_emotion", "")
        lines.append(f"[{ts}] [{emotion}]\n  User: {ep.get('user_message')}\n  Assistant: {ep.get('bot_response')[:200]}")
    return "\n".join(lines)

async def check_and_generate_summaries(user_id: int):
    from app.database import get_db
    db = get_db()
    llm = get_llm()
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")

    # Daily Summary
    existing_daily = await db.get_recent_summaries(user_id, "daily", limit=1)
    if not existing_daily or existing_daily[0].get("period_start", "")[:10] < yesterday:
        start, end = f"{yesterday}T00:00:00", f"{yesterday}T23:59:59"
        episodes = await db.get_episodes_by_date_range(user_id, start, end)
        if episodes:
            text = format_episodes_for_summary(episodes)
            prompt = DAILY_SUMMARY_PROMPT.format(date=yesterday, episodes=text)
            summary = await llm.generate_summary(prompt)
            await db.save_summary(user_id, "daily", start, end, summary)
            logger.info("Daily summary generated for user %d", user_id)

    # Weekly/Monthly can be added here if needed, following the same pattern
