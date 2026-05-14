"""
Layered memory engine for the AI Diary Companion.

Replaces the old JSON-file-based episodic memory with SQLite-backed
unlimited storage. Provides the bridge between raw database operations
and the higher-level retrieval/analysis engines.
"""

import logging
from datetime import datetime

from database import get_db

logger = logging.getLogger(__name__)


# ── Episode Operations ───────────────────────────────────────────────────────────

async def save_episode(
    user_id: int,
    user_message: str,
    bot_response: str,
    detected_emotion: str | None = None,
    emotion_confidence: float | None = None,
    secondary_emotion: str | None = None,
    topics: list[str] | None = None,
    is_diary_entry: bool = False,
) -> int:
    """
    Save a conversation episode to permanent storage.
    Returns the episode ID.
    """
    db = get_db()
    episode_id = await db.save_episode(
        user_id=user_id,
        user_message=user_message,
        bot_response=bot_response,
        detected_emotion=detected_emotion,
        emotion_confidence=emotion_confidence,
        secondary_emotion=secondary_emotion,
        topics=topics,
        is_diary_entry=is_diary_entry,
    )
    logger.debug(
        "Saved episode %d for user %d (emotion=%s, topics=%s)",
        episode_id, user_id, detected_emotion, topics,
    )
    return episode_id


async def get_recent_episodes(user_id: int, limit: int = 5) -> list[dict]:
    """Fetch the N most recent episodes in chronological order."""
    db = get_db()
    return await db.get_recent_episodes(user_id, limit)


async def get_episodes_by_emotion(user_id: int, emotion: str, limit: int = 5) -> list[dict]:
    """Fetch episodes where the user was feeling a specific emotion."""
    db = get_db()
    return await db.get_episodes_by_emotion(user_id, emotion, limit)


async def search_episodes(user_id: int, query: str, limit: int = 10) -> list[dict]:
    """Search across all episodes for a user by keyword."""
    db = get_db()
    return await db.search_episodes(user_id, query, limit)


async def get_episodes_by_date_range(user_id: int, start: str, end: str) -> list[dict]:
    """Fetch episodes within a date range."""
    db = get_db()
    return await db.get_episodes_by_date_range(user_id, start, end)


async def get_episode_count(user_id: int) -> int:
    """Total stored episodes for a user."""
    db = get_db()
    return await db.get_episode_count(user_id)


async def get_oldest_episode(user_id: int) -> dict | None:
    """Get the very first episode ever recorded for a user."""
    db = get_db()
    return await db.get_oldest_episode(user_id)


async def get_emotion_counts(user_id: int, days: int = 30) -> list[dict]:
    """Get emotion distribution over the last N days."""
    db = get_db()
    return await db.get_emotion_counts(user_id, days)


async def get_all_episodes(user_id: int) -> list[dict]:
    """Fetch ALL episodes for a user (for export)."""
    db = get_db()
    return await db.get_all_episodes(user_id)


async def delete_all_episodes(user_id: int) -> None:
    """Delete all data for a user. USE WITH CAUTION."""
    db = get_db()
    await db.delete_all_user_data(user_id)
    logger.info("Deleted all data for user %d", user_id)


# ── Memory Summary ───────────────────────────────────────────────────────────────

async def get_memory_summary(user_id: int) -> str:
    """
    Returns a human-readable summary of the user's memory stats.
    Used by the /memory command.
    """
    db = get_db()

    count = await db.get_episode_count(user_id)
    if count == 0:
        return "No memories stored yet. Start chatting and I'll remember everything!"

    oldest = await db.get_oldest_episode(user_id)
    oldest_date = oldest["timestamp"][:10] if oldest else "unknown"

    # Get schedule info
    schedule = await db.get_schedule(user_id)
    streak = schedule.get("streak_count", 0) if schedule else 0
    last_diary = schedule.get("last_diary_date", "never") if schedule else "never"

    # Get user settings
    user = await db.get_user(user_id)
    reminder_time = user.get("reminder_time", "22:00") if user else "22:00"
    reminder_enabled = user.get("reminder_enabled", 1) if user else 1

    # Get emotion stats
    emotions = await db.get_emotion_counts(user_id, days=30)
    top_emotions = ", ".join(
        f"{e['detected_emotion']} ({e['count']})" for e in emotions[:3]
    ) if emotions else "not enough data yet"

    lines = [
        f"📚 Total memories: {count}",
        f"📅 Oldest memory: {oldest_date}",
        f"📝 Last diary entry: {last_diary}",
        f"🔥 Current streak: {streak} day{'s' if streak != 1 else ''}",
        f"⏰ Daily reminder: {reminder_time} ({'on' if reminder_enabled else 'off'})",
        f"💭 Top emotions (30d): {top_emotions}",
    ]
    return "\n".join(lines)
