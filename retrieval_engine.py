"""
Hybrid context retrieval engine for the AI Diary Companion.

Builds the optimal context window for each LLM call by pulling from
multiple memory layers: recent episodes, semantic profile, summaries,
emotionally relevant memories, and topic-matched episodes.
"""

import logging
import re

from config import (
    RECENT_EPISODES_COUNT,
    TOPIC_MATCH_COUNT,
    EMOTION_MATCH_COUNT,
    SUMMARY_COUNT,
    MAX_CONTEXT_CHARS,
)
from database import get_db
from memory_engine import get_recent_episodes, search_episodes, get_episodes_by_emotion
from semantic_profile import get_profile, profile_to_context_string
from emotion_engine import get_emotional_trends

logger = logging.getLogger(__name__)


async def build_context(user_id: int, current_message: str) -> str:
    """
    Assemble the full context window from all memory layers.

    Priority ordering (highest to lowest):
    1. Semantic profile (always included if exists)
    2. Recent episodes (last N conversations)
    3. Recent diary entries (high priority — dedicated reflections)
    4. Topic-matched historical episodes
    5. Emotionally relevant memories
    6. Recent summaries
    7. Emotional trends

    The output is a single string injected into the LLM system prompt.
    """
    sections: list[str] = []

    # ── 1. Semantic Profile ──────────────────────────────────────────────────────
    try:
        profile = await get_profile(user_id)
        profile_str = profile_to_context_string(profile)
        if profile_str:
            sections.append(profile_str)
    except Exception as exc:
        logger.warning("Failed to load semantic profile: %s", exc)

    # ── 2. Recent Episodes ──────────────────────────────────────────────────────
    try:
        recent = await get_recent_episodes(user_id, limit=RECENT_EPISODES_COUNT)
        if recent:
            recent_text = _format_episodes("RECENT CONVERSATIONS", recent)
            sections.append(recent_text)
    except Exception as exc:
        logger.warning("Failed to load recent episodes: %s", exc)

    # ── 3. Recent Diary Entries ──────────────────────────────────────────────────
    try:
        db = get_db()
        diary_entries = await db.get_recent_diary_entries(user_id, limit=3)
        if diary_entries:
            diary_text = _format_diary_entries("RECENT DIARY ENTRIES", diary_entries)
            sections.append(diary_text)

        # Also include high-importance diary entries not already shown
        important_entries = await db.get_important_diary_entries(user_id, limit=3)
        seen_diary_ids = {e.get("id") for e in diary_entries}
        extra_important = [
            e for e in important_entries
            if e.get("id") not in seen_diary_ids
        ]
        if extra_important:
            imp_text = _format_diary_entries(
                "IMPORTANT PAST DIARY ENTRIES", extra_important
            )
            sections.append(imp_text)
    except Exception as exc:
        logger.warning("Failed to load diary entries: %s", exc)

    # ── 4. Topic-Matched Episodes ────────────────────────────────────────────────
    try:
        keywords = _extract_keywords(current_message)
        if keywords:
            topic_episodes: list[dict] = []
            seen_ids = {ep.get("id") for ep in (recent if recent else [])}

            for kw in keywords[:3]:  # limit keyword queries
                matches = await search_episodes(user_id, kw, limit=TOPIC_MATCH_COUNT)
                for m in matches:
                    if m.get("id") not in seen_ids:
                        topic_episodes.append(m)
                        seen_ids.add(m.get("id"))

            if topic_episodes:
                # Deduplicate and limit
                topic_episodes = topic_episodes[:TOPIC_MATCH_COUNT]
                topic_text = _format_episodes(
                    "RELEVANT PAST CONVERSATIONS (matched by topic)", topic_episodes
                )
                sections.append(topic_text)
    except Exception as exc:
        logger.warning("Failed to load topic-matched episodes: %s", exc)

    # ── 5. Emotionally Relevant Memories ─────────────────────────────────────────
    try:
        # Detect if the current message has emotional keywords
        emotion_kw = _detect_emotion_keywords(current_message)
        if emotion_kw:
            seen_ids_emo = {ep.get("id") for ep in (recent if recent else [])}
            emotion_eps = await get_episodes_by_emotion(
                user_id, emotion_kw, limit=EMOTION_MATCH_COUNT
            )
            emotion_eps = [e for e in emotion_eps if e.get("id") not in seen_ids_emo]

            if emotion_eps:
                emo_text = _format_episodes(
                    f"PAST MOMENTS WHEN USER FELT {emotion_kw.upper()}", emotion_eps
                )
                sections.append(emo_text)
    except Exception as exc:
        logger.warning("Failed to load emotion-matched episodes: %s", exc)

    # ── 6. Recent Summaries ──────────────────────────────────────────────────────
    try:
        db = get_db()
        daily_summaries = await db.get_recent_summaries(user_id, "daily", limit=SUMMARY_COUNT)
        weekly_summaries = await db.get_recent_summaries(user_id, "weekly", limit=1)

        summary_parts = []
        for s in daily_summaries:
            date = s.get("period_start", "")[:10]
            summary_parts.append(f"[{date}] {s.get('content', '')}")
        for s in weekly_summaries:
            date = s.get("period_start", "")[:10]
            summary_parts.append(f"[Week of {date}] {s.get('content', '')}")

        if summary_parts:
            summary_text = "=== RECENT LIFE SUMMARIES ===\n" + "\n\n".join(summary_parts) + "\n=== END SUMMARIES ==="
            sections.append(summary_text)
    except Exception as exc:
        logger.warning("Failed to load summaries: %s", exc)

    # ── 7. Emotional Trends ──────────────────────────────────────────────────────
    try:
        trends = await get_emotional_trends(user_id, days=14)
        if trends.get("total_entries", 0) > 3:
            sections.append(f"=== EMOTIONAL TREND ===\n{trends['trend_summary']}\n=== END TREND ===")
    except Exception as exc:
        logger.warning("Failed to load emotional trends: %s", exc)

    # ── Assemble and Truncate ────────────────────────────────────────────────────
    if not sections:
        return ""

    full_context = "\n\n".join(sections)

    # Truncate if exceeding limit (cut from the end, keeping profile + recent)
    if len(full_context) > MAX_CONTEXT_CHARS:
        full_context = full_context[:MAX_CONTEXT_CHARS] + "\n... (older context truncated)"

    return full_context


# ── Helper Functions ─────────────────────────────────────────────────────────────

def _format_episodes(title: str, episodes: list[dict]) -> str:
    """Format a list of episodes into a labelled context block."""
    lines = [f"=== {title} ==="]
    for ep in episodes:
        ts = ep.get("timestamp", "")[:16].replace("T", " ")
        emotion = ep.get("detected_emotion", "")
        emotion_tag = f" [{emotion}]" if emotion else ""
        lines.append(f"[{ts}]{emotion_tag}")
        lines.append(f"User: {ep.get('user_message', '')}")
        lines.append(f"You: {ep.get('bot_response', '')[:300]}")
        lines.append("")
    lines.append(f"=== END {title.split('(')[0].strip()} ===")
    return "\n".join(lines)


def _format_diary_entries(title: str, entries: list[dict]) -> str:
    """Format a list of diary entries into a labelled context block."""
    lines = [f"=== {title} ==="]
    for entry in entries:
        date = entry.get("created_at", "")[:10]
        emotions = entry.get("detected_emotions", "")
        importance = entry.get("importance_score", 0.5)
        stars = "★" * max(1, round((importance or 0.5) * 5))
        summary = entry.get("ai_summary", "")
        raw = entry.get("raw_text", "")[:200]

        entry_title = entry.get("title") or "Diary Entry"
        lines.append(f"[{date}] {entry_title} ({emotions}) {stars}")
        if summary:
            lines.append(f"Summary: {summary}")
        else:
            lines.append(f"Entry: {raw}")

        goals = entry.get("extracted_goals", [])
        if goals and isinstance(goals, list):
            lines.append(f"Goals mentioned: {', '.join(goals)}")
        stressors = entry.get("extracted_stressors", [])
        if stressors and isinstance(stressors, list):
            lines.append(f"Stressors: {', '.join(stressors)}")

        lines.append("")
    lines.append(f"=== END {title.split('(')[0].strip()} ===")
    return "\n".join(lines)


def _extract_keywords(message: str) -> list[str]:
    """Extract meaningful keywords from a message for topic matching."""
    # Remove common stop words and short words
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "shall",
        "it", "its", "i", "me", "my", "we", "our", "you", "your",
        "he", "she", "they", "them", "this", "that", "these", "those",
        "in", "on", "at", "to", "for", "of", "with", "by", "from",
        "and", "or", "but", "not", "so", "if", "then", "than",
        "what", "how", "when", "where", "why", "who", "which",
        "about", "just", "really", "very", "much", "some", "any",
        "all", "more", "most", "other", "no", "yes", "up", "out",
        "now", "here", "there", "also", "like", "get", "got", "go",
        "going", "went", "come", "came", "make", "made", "take",
        "took", "know", "knew", "think", "thought", "feel", "felt",
        "want", "need", "tell", "told", "say", "said", "see", "saw",
        "been", "being", "thing", "things", "way", "day", "time",
        "today", "yesterday", "tomorrow", "don", "didn", "doesn",
        "won", "wouldn", "couldn", "shouldn", "haven", "hasn", "hadn",
    }

    words = re.findall(r"\b[a-zA-Z]{3,}\b", message.lower())
    keywords = [w for w in words if w not in stop_words]

    # Return unique keywords, max 5
    seen = set()
    unique = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)
    return unique[:5]


# Emotion keyword mapping for matching current message to historical emotions
_EMOTION_KEYWORDS: dict[str, list[str]] = {
    "anxious": ["anxious", "anxiety", "nervous", "worried", "worry", "panic", "fear"],
    "sad": ["sad", "depressed", "upset", "unhappy", "crying", "tears", "grief", "loss"],
    "stressed": ["stressed", "stress", "pressure", "overwhelmed", "burnout", "burnt"],
    "angry": ["angry", "frustrated", "annoyed", "irritated", "furious", "rage"],
    "lonely": ["lonely", "alone", "isolated", "nobody", "no one"],
    "happy": ["happy", "joy", "joyful", "excited", "great", "wonderful", "amazing"],
    "tired": ["tired", "exhausted", "fatigue", "sleepy", "drained", "worn"],
    "hopeful": ["hopeful", "hope", "optimistic", "looking forward", "positive"],
    "proud": ["proud", "accomplished", "achievement", "succeeded", "nailed"],
    "grateful": ["grateful", "thankful", "blessed", "appreciate"],
}


def _detect_emotion_keywords(message: str) -> str | None:
    """Detect if the message contains emotion-related keywords."""
    msg_lower = message.lower()
    best_match: str | None = None
    best_count = 0

    for emotion, keywords in _EMOTION_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in msg_lower)
        if count > best_count:
            best_count = count
            best_match = emotion

    return best_match if best_count > 0 else None
