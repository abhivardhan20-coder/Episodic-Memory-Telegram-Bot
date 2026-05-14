"""
Rolling summary generation for the AI Diary Companion.

Generates daily, weekly, and monthly summaries from conversation episodes.
Summaries capture major events, emotional trends, and goal progress.
"""

import logging
from datetime import datetime, timedelta

from database import get_db
from llm_client import get_llm
from prompts import DAILY_SUMMARY_PROMPT, WEEKLY_SUMMARY_PROMPT, MONTHLY_SUMMARY_PROMPT

logger = logging.getLogger(__name__)


def _format_episodes_for_summary(episodes: list[dict]) -> str:
    """Format a list of episodes into a readable string for the LLM."""
    if not episodes:
        return "(no conversations)"

    lines = []
    for ep in episodes:
        ts = ep.get("timestamp", "")[:16].replace("T", " ")
        emotion = ep.get("detected_emotion", "")
        emotion_str = f" [{emotion}]" if emotion else ""
        lines.append(f"[{ts}]{emotion_str}")
        lines.append(f"  User: {ep.get('user_message', '')}")
        lines.append(f"  Assistant: {ep.get('bot_response', '')[:200]}")
        lines.append("")

    return "\n".join(lines)


async def generate_daily_summary(user_id: int, date: str | None = None) -> str | None:
    """
    Generate a daily summary for a specific date.

    Args:
        user_id: Telegram user ID
        date: ISO date string (YYYY-MM-DD). Defaults to today.

    Returns:
        The generated summary text, or None if no episodes exist for that day.
    """
    db = get_db()

    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    # Fetch episodes for that day
    start = f"{date}T00:00:00"
    end = f"{date}T23:59:59"
    episodes = await db.get_episodes_by_date_range(user_id, start, end)

    if not episodes:
        logger.debug("No episodes for user %d on %s, skipping daily summary", user_id, date)
        return None

    # Format and send to LLM
    episodes_text = _format_episodes_for_summary(episodes)
    prompt = DAILY_SUMMARY_PROMPT.format(date=date, episodes=episodes_text)

    llm = get_llm()
    summary = await llm.generate_summary(prompt)

    # Store the summary
    await db.save_summary(
        user_id=user_id,
        summary_type="daily",
        period_start=start,
        period_end=end,
        content=summary,
        emotional_trends=_extract_emotion_counts(episodes),
        key_events=None,
    )

    logger.info("Generated daily summary for user %d on %s", user_id, date)
    return summary


async def generate_weekly_summary(user_id: int, week_start: str | None = None) -> str | None:
    """
    Generate a weekly summary from daily summaries.

    Args:
        user_id: Telegram user ID
        week_start: ISO date string for Monday of the week. Defaults to current week.
    """
    db = get_db()

    if week_start is None:
        today = datetime.now()
        monday = today - timedelta(days=today.weekday())
        week_start = monday.strftime("%Y-%m-%d")

    week_end_dt = datetime.strptime(week_start, "%Y-%m-%d") + timedelta(days=6)
    week_end = week_end_dt.strftime("%Y-%m-%d")

    # Fetch daily summaries for this week
    daily_summaries = await db.get_recent_summaries(user_id, "daily", limit=7)

    # Filter to this week's summaries
    relevant = [
        s for s in daily_summaries
        if s.get("period_start", "")[:10] >= week_start
        and s.get("period_start", "")[:10] <= week_end
    ]

    if not relevant:
        logger.debug("No daily summaries for user %d in week %s", user_id, week_start)
        return None

    summaries_text = "\n\n".join(
        f"[{s.get('period_start', '')[:10]}] {s.get('content', '')}"
        for s in relevant
    )

    prompt = WEEKLY_SUMMARY_PROMPT.format(daily_summaries=summaries_text)

    llm = get_llm()
    summary = await llm.generate_summary(prompt)

    await db.save_summary(
        user_id=user_id,
        summary_type="weekly",
        period_start=f"{week_start}T00:00:00",
        period_end=f"{week_end}T23:59:59",
        content=summary,
    )

    logger.info("Generated weekly summary for user %d (week of %s)", user_id, week_start)
    return summary


async def generate_monthly_summary(user_id: int, year_month: str | None = None) -> str | None:
    """
    Generate a monthly summary from weekly summaries.

    Args:
        user_id: Telegram user ID
        year_month: Format "YYYY-MM". Defaults to current month.
    """
    db = get_db()

    if year_month is None:
        year_month = datetime.now().strftime("%Y-%m")

    # Fetch weekly summaries that fall in this month
    weekly_summaries = await db.get_recent_summaries(user_id, "weekly", limit=5)

    relevant = [
        s for s in weekly_summaries
        if s.get("period_start", "")[:7] == year_month
    ]

    if not relevant:
        logger.debug("No weekly summaries for user %d in %s", user_id, year_month)
        return None

    summaries_text = "\n\n".join(
        f"[Week of {s.get('period_start', '')[:10]}] {s.get('content', '')}"
        for s in relevant
    )

    prompt = MONTHLY_SUMMARY_PROMPT.format(weekly_summaries=summaries_text)

    llm = get_llm()
    summary = await llm.generate_summary(prompt)

    month_start = f"{year_month}-01T00:00:00"
    # Approximate month end
    month_end = f"{year_month}-28T23:59:59"

    await db.save_summary(
        user_id=user_id,
        summary_type="monthly",
        period_start=month_start,
        period_end=month_end,
        content=summary,
    )

    logger.info("Generated monthly summary for user %d (%s)", user_id, year_month)
    return summary


async def check_and_generate_summaries(user_id: int) -> None:
    """
    Check if any summaries need to be generated and create them.
    Called periodically or after diary check-ins.
    """
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")

    # Generate yesterday's daily summary if it doesn't exist
    db = get_db()
    existing_daily = await db.get_recent_summaries(user_id, "daily", limit=1)
    last_daily_date = (
        existing_daily[0].get("period_start", "")[:10]
        if existing_daily
        else ""
    )

    if last_daily_date < yesterday:
        await generate_daily_summary(user_id, yesterday)

    # Generate weekly summary on Mondays
    if now.weekday() == 0:
        last_monday = (now - timedelta(days=7)).strftime("%Y-%m-%d")
        existing_weekly = await db.get_recent_summaries(user_id, "weekly", limit=1)
        last_weekly = (
            existing_weekly[0].get("period_start", "")[:10]
            if existing_weekly
            else ""
        )
        if last_weekly < last_monday:
            await generate_weekly_summary(user_id, last_monday)

    # Generate monthly summary on the 1st
    if now.day == 1:
        last_month = (now - timedelta(days=1)).strftime("%Y-%m")
        existing_monthly = await db.get_recent_summaries(user_id, "monthly", limit=1)
        last_monthly = (
            existing_monthly[0].get("period_start", "")[:7]
            if existing_monthly
            else ""
        )
        if last_monthly < last_month:
            await generate_monthly_summary(user_id, last_month)


def _extract_emotion_counts(episodes: list[dict]) -> dict:
    """Extract emotion frequency from a list of episodes."""
    counts: dict[str, int] = {}
    for ep in episodes:
        emotion = ep.get("detected_emotion")
        if emotion:
            counts[emotion] = counts.get(emotion, 0) + 1
    return counts
