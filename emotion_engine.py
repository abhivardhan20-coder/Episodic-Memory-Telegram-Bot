"""
Emotion detection and pattern analysis engine.

After each conversation exchange, this module analyses the emotional tone,
extracts topics, and detects recurring patterns over time.
"""

import logging
from collections import Counter

from database import get_db
from llm_client import get_llm
from prompts import EMOTION_ANALYSIS_PROMPT

logger = logging.getLogger(__name__)

# Valid emotion labels
VALID_EMOTIONS = {
    "happy", "sad", "anxious", "angry", "excited", "stressed",
    "grateful", "neutral", "proud", "lonely", "hopeful", "frustrated",
    "calm", "overwhelmed", "nostalgic", "confused", "motivated", "tired",
}


async def analyze_emotion(
    user_message: str, bot_response: str
) -> dict:
    """
    Analyze the emotional tone of a conversation exchange.

    Returns:
        {
            "emotion": str,
            "confidence": float,
            "secondary_emotion": str | None,
            "topics": list[str],
        }
    """
    prompt = EMOTION_ANALYSIS_PROMPT.format(
        user_message=user_message,
        bot_response=bot_response,
    )

    llm = get_llm()
    result = await llm.analyze_emotion(prompt)

    if result is None:
        logger.warning("Emotion analysis returned None, defaulting to neutral")
        return {
            "emotion": "neutral",
            "confidence": 0.5,
            "secondary_emotion": None,
            "topics": [],
        }

    # Validate and normalize
    emotion = result.get("emotion", "neutral").lower().strip()
    if emotion not in VALID_EMOTIONS:
        emotion = "neutral"

    secondary = result.get("secondary_emotion")
    if secondary:
        secondary = secondary.lower().strip()
        if secondary not in VALID_EMOTIONS or secondary == "null":
            secondary = None

    confidence = result.get("confidence", 0.5)
    if not isinstance(confidence, (int, float)):
        confidence = 0.5
    confidence = max(0.0, min(1.0, float(confidence)))

    topics = result.get("topics", [])
    if not isinstance(topics, list):
        topics = []
    topics = [str(t).strip() for t in topics if t]

    return {
        "emotion": emotion,
        "confidence": confidence,
        "secondary_emotion": secondary,
        "topics": topics,
    }


async def get_emotional_trends(user_id: int, days: int = 30) -> dict:
    """
    Get emotional trends over the last N days.

    Returns:
        {
            "top_emotions": [("emotion", count), ...],
            "total_entries": int,
            "dominant_emotion": str,
            "trend_summary": str,
        }
    """
    db = get_db()
    emotion_counts = await db.get_emotion_counts(user_id, days)

    if not emotion_counts:
        return {
            "top_emotions": [],
            "total_entries": 0,
            "dominant_emotion": "neutral",
            "trend_summary": "Not enough data to identify emotional trends yet.",
        }

    total = sum(e["count"] for e in emotion_counts)
    top = [(e["detected_emotion"], e["count"]) for e in emotion_counts[:5]]
    dominant = emotion_counts[0]["detected_emotion"]

    # Build a human-readable summary
    parts = []
    for emotion, count in top[:3]:
        pct = round(count / total * 100)
        parts.append(f"{emotion} ({pct}%)")

    trend_summary = f"Over the last {days} days ({total} entries): " + ", ".join(parts)

    return {
        "top_emotions": top,
        "total_entries": total,
        "dominant_emotion": dominant,
        "trend_summary": trend_summary,
    }


async def detect_patterns(user_id: int) -> dict:
    """
    Detect recurring emotional and behavioral patterns.

    Returns a dict with identified patterns.
    """
    db = get_db()

    # Get all recent episodes with emotions
    episodes = await db.get_recent_episodes(user_id, limit=50)

    if len(episodes) < 5:
        return {
            "recurring_emotions": [],
            "frequent_topics": [],
            "patterns": [],
        }

    # Count emotions
    emotion_counter = Counter()
    topic_counter = Counter()

    for ep in episodes:
        if ep.get("detected_emotion"):
            emotion_counter[ep["detected_emotion"]] += 1
        topics = ep.get("topics", [])
        if isinstance(topics, list):
            for t in topics:
                topic_counter[t.lower()] += 1

    # Identify recurring emotions (>20% frequency)
    total_with_emotion = sum(emotion_counter.values())
    recurring_emotions = [
        emotion for emotion, count in emotion_counter.most_common()
        if count / max(total_with_emotion, 1) > 0.2
    ]

    # Identify frequent topics (appearing 3+ times)
    frequent_topics = [
        topic for topic, count in topic_counter.most_common(10)
        if count >= 3
    ]

    # Simple pattern detection
    patterns = []
    if "stressed" in recurring_emotions and "anxious" in recurring_emotions:
        patterns.append("Frequently experiencing stress and anxiety together")
    if "lonely" in recurring_emotions:
        patterns.append("Recurring feelings of loneliness")
    if "motivated" in recurring_emotions and "tired" in recurring_emotions:
        patterns.append("Cycle between high motivation and fatigue")
    if emotion_counter.get("happy", 0) > emotion_counter.get("sad", 0) * 2:
        patterns.append("Generally positive emotional baseline")
    elif emotion_counter.get("sad", 0) > emotion_counter.get("happy", 0) * 2:
        patterns.append("Persistent low mood pattern")

    return {
        "recurring_emotions": recurring_emotions,
        "frequent_topics": frequent_topics,
        "patterns": patterns,
    }
