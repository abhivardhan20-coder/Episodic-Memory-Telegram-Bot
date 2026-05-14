"""
Diary engine for the AI Diary Companion.

Manages structured diary entries — separate from conversations.
Provides deep AI analysis, importance scoring, and pattern detection.
Each entry is analyzed for emotions, goals, stressors, personality signals,
and behavioral patterns, then stored permanently in the diary_entries table.
"""

import json
import logging
from datetime import datetime

from database import get_db
from llm_client import get_llm
from semantic_profile import get_profile, update_profile_from_conversation, profile_to_context_string
from prompts import DIARY_ANALYSIS_PROMPT, DIARY_FOLLOWUP_PROMPT

logger = logging.getLogger(__name__)


async def process_diary_entry(user_id: int, raw_text: str) -> dict:
    """
    Process a new diary entry through the full analysis pipeline.

    1. Save raw entry immediately (for zero data loss)
    2. Run deep AI analysis
    3. Update the entry with analysis results
    4. Generate a reflective follow-up response
    5. Update semantic profile from diary content

    Returns:
        {
            "entry_id": int,
            "analysis": dict,
            "followup": str,
        }
    """
    db = get_db()

    # Step 1: Save raw entry immediately (crash-safe)
    entry_id = await db.save_diary_entry(user_id=user_id, raw_text=raw_text)
    logger.info("Saved raw diary entry %d for user %d", entry_id, user_id)

    # Step 2: Deep AI analysis
    analysis = await _analyze_diary_entry(user_id, raw_text)

    # Step 3: Update entry with analysis results
    if analysis:
        await db.update_diary_entry(
            entry_id,
            title=analysis.get("title"),
            detected_emotions=analysis.get("detected_emotions"),
            emotion_confidence=analysis.get("emotion_confidence"),
            extracted_goals=analysis.get("extracted_goals"),
            extracted_stressors=analysis.get("extracted_stressors"),
            extracted_relationships=analysis.get("extracted_relationships"),
            extracted_topics=analysis.get("extracted_topics"),
            personality_signals=analysis.get("personality_signals"),
            behavioral_patterns=analysis.get("behavioral_patterns"),
            ai_summary=analysis.get("ai_summary"),
            importance_score=analysis.get("importance_score", 0.5),
        )
        logger.info(
            "Updated diary entry %d with analysis (importance=%.2f, emotions=%s)",
            entry_id,
            analysis.get("importance_score", 0.5),
            analysis.get("detected_emotions"),
        )
    else:
        analysis = {}

    # Step 4: Generate reflective follow-up
    followup = await _generate_diary_followup(user_id, raw_text, analysis)

    # Step 5: Update the entry with the follow-up
    if followup:
        await db.update_diary_entry(entry_id, ai_followup=followup)

    # Step 6: Update semantic profile (use diary text as user message, followup as bot response)
    try:
        await update_profile_from_conversation(
            user_id, raw_text, followup or "Diary entry recorded."
        )
    except Exception as exc:
        logger.warning("Profile update from diary failed: %s", exc)

    return {
        "entry_id": entry_id,
        "analysis": analysis,
        "followup": followup or "Your diary entry has been saved. Thank you for sharing. 🌙",
    }


async def _analyze_diary_entry(user_id: int, diary_text: str) -> dict | None:
    """Run deep AI analysis on a diary entry."""
    profile = await get_profile(user_id)
    profile_str = profile_to_context_string(profile)

    prompt = DIARY_ANALYSIS_PROMPT.format(
        diary_text=diary_text,
        profile=profile_str or "(new user, no profile yet)",
    )

    llm = get_llm()
    result = await llm.analyze_emotion(prompt)  # uses _call_json internally

    if result is None:
        logger.warning("Diary analysis returned None for user %d", user_id)
        return None

    # Validate and normalize importance score
    importance = result.get("importance_score", 0.5)
    if not isinstance(importance, (int, float)):
        importance = 0.5
    result["importance_score"] = max(0.0, min(1.0, float(importance)))

    # Validate emotion confidence
    confidence = result.get("emotion_confidence", 0.5)
    if not isinstance(confidence, (int, float)):
        confidence = 0.5
    result["emotion_confidence"] = max(0.0, min(1.0, float(confidence)))

    # Ensure list fields are lists
    for field in (
        "extracted_goals", "extracted_stressors", "extracted_relationships",
        "extracted_topics", "personality_signals", "behavioral_patterns",
    ):
        val = result.get(field)
        if val is None:
            result[field] = []
        elif isinstance(val, str):
            result[field] = [val]
        elif not isinstance(val, list):
            result[field] = []

    return result


async def _generate_diary_followup(
    user_id: int, diary_text: str, analysis: dict
) -> str | None:
    """Generate a warm, reflective follow-up response to a diary entry."""
    profile = await get_profile(user_id)
    profile_str = profile_to_context_string(profile)

    emotions = analysis.get("detected_emotions", "not analyzed")
    topics = ", ".join(analysis.get("extracted_topics", [])) or "general reflection"
    stressors = ", ".join(analysis.get("extracted_stressors", [])) or "none detected"
    goals = ", ".join(analysis.get("extracted_goals", [])) or "none mentioned"

    prompt = DIARY_FOLLOWUP_PROMPT.format(
        diary_text=diary_text,
        emotions=emotions,
        topics=topics,
        stressors=stressors,
        goals=goals,
        profile=profile_str or "(new user)",
    )

    llm = get_llm()
    try:
        return await llm.chat(
            "You are a deeply empathetic AI diary companion. Respond with warmth and insight.",
            prompt,
            max_tokens=300,
        )
    except Exception as exc:
        logger.error("Diary followup generation failed: %s", exc)
        return None


async def get_diary_summary(user_id: int) -> str:
    """
    Get a summary of the user's diary statistics.
    Used by /memory and other commands.
    """
    db = get_db()
    count = await db.get_diary_entry_count(user_id)

    if count == 0:
        return "No diary entries yet. Use /diary to write your first one!"

    latest = await db.get_latest_diary_entry(user_id)
    latest_date = latest.get("created_at", "")[:10] if latest else "unknown"
    latest_emotions = latest.get("detected_emotions", "unknown") if latest else "unknown"

    # Get high-importance entries count
    important = await db.get_important_diary_entries(user_id, limit=100)
    high_importance_count = sum(
        1 for e in important if (e.get("importance_score") or 0) >= 0.7
    )

    lines = [
        f"📓 Total diary entries: {count}",
        f"📅 Latest entry: {latest_date}",
        f"💭 Latest mood: {latest_emotions}",
        f"⭐ High-importance entries: {high_importance_count}",
    ]
    return "\n".join(lines)


async def format_diary_entry_display(entry: dict) -> str:
    """Format a diary entry for display in Telegram."""
    date = entry.get("created_at", "")[:10]
    title = entry.get("title") or "Untitled Entry"
    emotions = entry.get("detected_emotions") or "not analyzed"
    importance = entry.get("importance_score") or 0.5
    summary = entry.get("ai_summary") or "(no summary)"
    raw_text = entry.get("raw_text", "")

    # Importance star rating
    stars = "⭐" * max(1, round(importance * 5))

    # Truncate raw text for display
    preview = raw_text[:300] + "..." if len(raw_text) > 300 else raw_text

    lines = [
        f"📓 {title}",
        f"📅 {date} | {emotions} | {stars}",
        "",
        f"📝 {preview}",
        "",
        f"💭 {summary}",
    ]

    goals = entry.get("extracted_goals", [])
    if goals:
        lines.append(f"🎯 Goals: {', '.join(goals)}")

    stressors = entry.get("extracted_stressors", [])
    if stressors:
        lines.append(f"⚡ Stressors: {', '.join(stressors)}")

    return "\n".join(lines)
