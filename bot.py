"""
AI Diary Companion — Telegram Bot

A persistent AI diary that remembers everything, provides emotionally
aware advice, and proactively encourages reflective journaling.

Main entry point: handles all commands, message routing, and lifecycle.
"""

import asyncio
import json
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path

from telegram import Update
from telegram.constants import ChatAction
from telegram.error import TelegramError
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from config import (
    TELEGRAM_BOT_TOKEN,
    DEFAULT_TIMEZONE,
    DEFAULT_REMINDER_TIME,
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_DIR,
    EXPORT_DIR,
)
from database import get_db
from llm_client import get_llm
from memory_engine import (
    save_episode,
    get_memory_summary,
    search_episodes,
    get_all_episodes,
    delete_all_episodes,
    get_episode_count,
)
from semantic_profile import (
    get_profile,
    update_profile_from_conversation,
    profile_to_context_string,
)
from emotion_engine import analyze_emotion, get_emotional_trends, detect_patterns
from retrieval_engine import build_context
from scheduler import setup_scheduler, schedule_user_reminder, shutdown_scheduler
from summarizer import check_and_generate_summaries
from diary_engine import process_diary_entry, get_diary_summary, format_diary_entry_display
from prompts import (
    SYSTEM_PROMPT, SUMMARY_COMMAND_PROMPT, SEARCH_RESULT_PROMPT,
    DIARY_ENTRY_INTRO, MOOD_SUMMARY_PROMPT, TIMELINE_PROMPT,
)

# ── Logging Setup ────────────────────────────────────────────────────────────────

logging.basicConfig(
    format=LOG_FORMAT,
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "bot.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ── Command Handlers ─────────────────────────────────────────────────────────────

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/start — Welcome and introduce diary companion features."""
    user = update.effective_user
    db = get_db()

    # Check if the user already exists in the database
    existing_user = await db.get_user(user.id)
    is_first_time = existing_user is None

    await db.ensure_user(user.id, user.username, user.first_name)

    # Set up default reminder schedule
    await schedule_user_reminder(
        context.application, user.id, DEFAULT_REMINDER_TIME, DEFAULT_TIMEZONE
    )

    if is_first_time:
        # First-time user — ask for their name
        context.user_data["awaiting_name"] = True
        await update.message.reply_text(
            "Hey there! 🌙\n\n"
            "Welcome to your AI Diary Companion — I'm a journal that listens, "
            "remembers, and grows with you.\n\n"
            "Before we begin, what should I call you?"
        )
    else:
        # Returning user
        name = user.first_name or "there"
        # Try to get name from semantic profile
        profile = await get_profile(user.id)
        if profile.get("name"):
            name = profile["name"]
        await _send_welcome_message(update, name)


async def memory_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/memory — Show memory statistics."""
    user_id = update.effective_user.id
    db = get_db()
    await db.ensure_user(user_id, update.effective_user.username, update.effective_user.first_name)

    conv_summary = await get_memory_summary(user_id)
    diary_summary = await get_diary_summary(user_id)
    await update.message.reply_text(
        f"🧠 Your Memory Dashboard\n\n{conv_summary}\n\n{diary_summary}"
    )


async def clear_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/clear — Two-step confirmation before wiping all data."""
    user_id = update.effective_user.id

    # Check if this is the confirmation step
    if context.user_data.get("clear_pending"):
        context.user_data["clear_pending"] = False
        count = await get_episode_count(user_id)
        await delete_all_episodes(user_id)
        await update.message.reply_text(
            f"Done. I've forgotten everything — {count} memories erased. 🧹\n"
            "Fresh start! Use /start to set up again."
        )
    else:
        count = await get_episode_count(user_id)
        context.user_data["clear_pending"] = True
        await update.message.reply_text(
            f"⚠️ This will permanently delete {count} memories.\n\n"
            "Are you sure? Send /clear again to confirm, "
            "or send anything else to cancel."
        )


async def settime_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/settime HH:MM — Configure daily reminder time."""
    user_id = update.effective_user.id
    db = get_db()
    await db.ensure_user(user_id, update.effective_user.username, update.effective_user.first_name)

    args = context.args
    if not args:
        user = await db.get_user(user_id)
        current = user.get("reminder_time", DEFAULT_REMINDER_TIME) if user else DEFAULT_REMINDER_TIME
        await update.message.reply_text(
            f"⏰ Current reminder time: {current}\n\n"
            "Usage: /settime HH:MM\n"
            "Example: /settime 21:30"
        )
        return

    time_str = args[0].strip()

    # Validate format
    try:
        parts = time_str.split(":")
        hour = int(parts[0])
        minute = int(parts[1])
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError
        time_str = f"{hour:02d}:{minute:02d}"
    except (ValueError, IndexError):
        await update.message.reply_text(
            "❌ Invalid time format. Use HH:MM (24-hour).\n"
            "Example: /settime 22:00"
        )
        return

    user = await db.get_user(user_id)
    timezone = user.get("timezone", DEFAULT_TIMEZONE) if user else DEFAULT_TIMEZONE

    await schedule_user_reminder(context.application, user_id, time_str, timezone)

    await update.message.reply_text(
        f"✅ Daily reminder set to {time_str} ({timezone}).\n"
        "I'll check in with you every day at that time. 🌙"
    )


async def summary_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/summary — Generate a personal life summary."""
    user_id = update.effective_user.id
    db = get_db()
    await db.ensure_user(user_id, update.effective_user.username, update.effective_user.first_name)

    count = await get_episode_count(user_id)
    if count < 3:
        await update.message.reply_text(
            "I need a few more conversations before I can create a meaningful summary. "
            "Keep sharing! 💭"
        )
        return

    # Show typing while generating
    try:
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=ChatAction.TYPING
        )
    except TelegramError:
        pass

    # Gather data for summary
    trends = await get_emotional_trends(user_id, days=30)
    profile = await get_profile(user_id)
    profile_str = profile_to_context_string(profile)

    recent_summaries_data = await db.get_recent_summaries(user_id, "daily", limit=5)
    recent_summaries = "\n".join(
        f"[{s.get('period_start', '')[:10]}] {s.get('content', '')}"
        for s in recent_summaries_data
    ) if recent_summaries_data else "(no summaries yet)"

    from memory_engine import get_recent_episodes
    recent_eps = await get_recent_episodes(user_id, limit=5)
    recent_eps_text = "\n".join(
        f"[{ep.get('timestamp', '')[:10]}] User: {ep.get('user_message', '')[:100]}"
        for ep in recent_eps
    ) if recent_eps else "(no recent conversations)"

    prompt = SUMMARY_COMMAND_PROMPT.format(
        emotional_trends=trends.get("trend_summary", "Not enough data"),
        profile=profile_str or "(no profile yet)",
        recent_summaries=recent_summaries,
        recent_episodes=recent_eps_text,
    )

    llm = get_llm()
    summary_text = await llm.chat(
        "You are a warm, insightful personal life summarizer. Write directly to the user.",
        prompt,
        max_tokens=600,
    )

    await update.message.reply_text(f"📊 Your Life Summary\n\n{summary_text}")


async def search_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/search <query> — Search across all memories."""
    user_id = update.effective_user.id
    db = get_db()
    await db.ensure_user(user_id, update.effective_user.username, update.effective_user.first_name)

    args = context.args
    if not args:
        await update.message.reply_text(
            "🔍 Usage: /search <query>\n\n"
            "Examples:\n"
            "/search exams\n"
            "/search burnout\n"
            "/search happiness"
        )
        return

    query = " ".join(args)

    # Show typing
    try:
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=ChatAction.TYPING
        )
    except TelegramError:
        pass

    results = await search_episodes(user_id, query, limit=10)

    if not results:
        await update.message.reply_text(
            f"🔍 No memories found for \"{query}\".\n"
            "Try different keywords or a broader search."
        )
        return

    # Format results for LLM to summarize
    results_text = "\n\n".join(
        f"[{r.get('timestamp', '')[:16].replace('T', ' ')}]\n"
        f"User: {r.get('user_message', '')[:200]}\n"
        f"Assistant: {r.get('bot_response', '')[:200]}"
        for r in results
    )

    prompt = SEARCH_RESULT_PROMPT.format(query=query, results=results_text)

    llm = get_llm()
    summary = await llm.chat(
        "You are a memory retrieval assistant. Present search results naturally and helpfully.",
        prompt,
        max_tokens=500,
    )

    await update.message.reply_text(
        f"🔍 Found {len(results)} memories about \"{query}\"\n\n{summary}"
    )


async def export_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/export — Export all diary data as JSON and Markdown files."""
    user_id = update.effective_user.id
    db = get_db()
    await db.ensure_user(user_id, update.effective_user.username, update.effective_user.first_name)

    count = await get_episode_count(user_id)
    if count == 0:
        await update.message.reply_text("No data to export yet!")
        return

    await update.message.reply_text("📦 Generating your export... This may take a moment.")

    try:
        # Gather all data
        episodes = await get_all_episodes(user_id)
        profile = await get_profile(user_id)
        summaries = await db.get_all_summaries(user_id)
        user_data = await db.get_user(user_id)
        schedule = await db.get_schedule(user_id)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ── JSON Export ──────────────────────────────────────────────────────
        diary_entries = await db.get_all_diary_entries(user_id)

        json_data = {
            "export_date": datetime.now().isoformat(),
            "user": user_data,
            "profile": profile,
            "schedule": schedule,
            "total_episodes": len(episodes),
            "total_diary_entries": len(diary_entries),
            "episodes": episodes,
            "diary_entries": diary_entries,
            "summaries": summaries,
        }

        json_path = EXPORT_DIR / f"diary_export_{user_id}_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)

        # ── Markdown Export ──────────────────────────────────────────────────
        md_lines = [
            f"# AI Diary Export — {datetime.now().strftime('%B %d, %Y')}",
            "",
            f"**Total entries:** {len(episodes)}",
            f"**Date range:** {episodes[0].get('timestamp', '')[:10]} to {episodes[-1].get('timestamp', '')[:10]}",
            "",
            "## Profile",
            "",
        ]

        if profile.get("name"):
            md_lines.append(f"**Name:** {profile['name']}")
        for key in ["goals", "stressors", "preferences", "habits", "aspirations"]:
            items = profile.get(key, [])
            if items:
                md_lines.append(f"**{key.title()}:** {', '.join(str(i) for i in items)}")
        md_lines.append("")

        md_lines.append("---")
        md_lines.append("")
        md_lines.append("## Diary Entries")
        md_lines.append("")

        current_date = ""
        for ep in episodes:
            ep_date = ep.get("timestamp", "")[:10]
            if ep_date != current_date:
                current_date = ep_date
                md_lines.append(f"### {current_date}")
                md_lines.append("")

            time = ep.get("timestamp", "")[11:16]
            emotion = ep.get("detected_emotion", "")
            emotion_tag = f" [{emotion}]" if emotion else ""
            md_lines.append(f"**{time}**{emotion_tag}")
            md_lines.append(f"> {ep.get('user_message', '')}")
            md_lines.append("")
            md_lines.append(f"{ep.get('bot_response', '')}")
            md_lines.append("")

        md_path = EXPORT_DIR / f"diary_export_{user_id}_{timestamp}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        # Send files
        with open(json_path, "rb") as f:
            await update.message.reply_document(
                document=f,
                filename=f"diary_export_{timestamp}.json",
                caption="📋 JSON export — complete raw data",
            )
        with open(md_path, "rb") as f:
            await update.message.reply_document(
                document=f,
                filename=f"diary_export_{timestamp}.md",
                caption="📖 Markdown export — readable diary format",
            )

        # Clean up export files after sending
        json_path.unlink(missing_ok=True)
        md_path.unlink(missing_ok=True)

    except Exception as exc:
        logger.error("Export failed for user %d: %s", user_id, exc)
        await update.message.reply_text(
            "❌ Export failed. Please try again later."
        )


# ── Diary Command Handlers ──────────────────────────────────────────────────────

async def diary_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/diary — Enter diary mode to write a structured diary entry."""
    user_id = update.effective_user.id
    db = get_db()
    await db.ensure_user(user_id, update.effective_user.username, update.effective_user.first_name)

    # Set diary mode flag
    context.user_data["diary_mode"] = True
    await update.message.reply_text(DIARY_ENTRY_INTRO)


async def diarylatest_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/diarylatest — Show the latest diary entry with analysis."""
    user_id = update.effective_user.id
    db = get_db()
    await db.ensure_user(user_id, update.effective_user.username, update.effective_user.first_name)

    latest = await db.get_latest_diary_entry(user_id)
    if not latest:
        await update.message.reply_text(
            "No diary entries yet! Use /diary to write your first one. 📝"
        )
        return

    display = await format_diary_entry_display(latest)
    await update.message.reply_text(display)


async def diarysearch_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/diarysearch <query> — Search across diary entries."""
    user_id = update.effective_user.id
    db = get_db()
    await db.ensure_user(user_id, update.effective_user.username, update.effective_user.first_name)

    args = context.args
    if not args:
        await update.message.reply_text(
            "🔍 Usage: /diarysearch <query>\n\n"
            "Examples:\n"
            "/diarysearch anxiety\n"
            "/diarysearch goals\n"
            "/diarysearch relationships"
        )
        return

    query = " ".join(args)

    try:
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=ChatAction.TYPING
        )
    except TelegramError:
        pass

    results = await db.search_diary_entries(user_id, query, limit=5)

    if not results:
        await update.message.reply_text(
            f"🔍 No diary entries found for \"{query}\".\n"
            "Try different keywords."
        )
        return

    # Format results
    lines = [f"🔍 Found {len(results)} diary entries about \"{query}\"\n"]
    for entry in results:
        date = entry.get("created_at", "")[:10]
        title = entry.get("title") or "Untitled"
        emotions = entry.get("detected_emotions") or ""
        importance = entry.get("importance_score") or 0.5
        stars = "⭐" * max(1, round(importance * 5))
        summary = entry.get("ai_summary") or entry.get("raw_text", "")[:100]
        lines.append(f"📓 {date} — {title} ({emotions}) {stars}")
        lines.append(f"   {summary}")
        lines.append("")

    await update.message.reply_text("\n".join(lines))


async def mood_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/mood — Show recent emotional trends from diary + conversations."""
    user_id = update.effective_user.id
    db = get_db()
    await db.ensure_user(user_id, update.effective_user.username, update.effective_user.first_name)

    diary_count = await db.get_diary_entry_count(user_id)
    episode_count = await get_episode_count(user_id)

    if diary_count + episode_count < 3:
        await update.message.reply_text(
            "I need a few more conversations or diary entries to analyze your mood. "
            "Keep sharing! 💭"
        )
        return

    try:
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=ChatAction.TYPING
        )
    except TelegramError:
        pass

    # Gather diary emotion timeline
    diary_timeline = await db.get_diary_emotion_timeline(user_id, limit=20)
    diary_timeline_text = "\n".join(
        f"[{e.get('created_at', '')[:10]}] {e.get('detected_emotions', 'unknown')} "
        f"(importance: {e.get('importance_score', 0.5):.1f}) — {(e.get('ai_summary') or '')[:80]}"
        for e in diary_timeline
    ) if diary_timeline else "(no diary entries yet)"

    # Gather conversation emotion data
    conversation_trends = await get_emotional_trends(user_id, days=30)
    conversation_text = conversation_trends.get("trend_summary", "Not enough data")

    # Get profile
    profile = await get_profile(user_id)
    profile_str = profile_to_context_string(profile)

    prompt = MOOD_SUMMARY_PROMPT.format(
        diary_timeline=diary_timeline_text,
        conversation_emotions=conversation_text,
        profile=profile_str or "(no profile yet)",
    )

    llm = get_llm()
    mood_report = await llm.chat(
        "You are an empathetic emotional wellness analyst. Write directly to the user.",
        prompt,
        max_tokens=500,
    )

    await update.message.reply_text(f"💭 Your Mood Report\n\n{mood_report}")


async def timeline_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/timeline — Show important life events from diary entries."""
    user_id = update.effective_user.id
    db = get_db()
    await db.ensure_user(user_id, update.effective_user.username, update.effective_user.first_name)

    events = await db.get_diary_timeline_events(user_id, min_importance=0.5, limit=20)

    if not events:
        await update.message.reply_text(
            "📌 No significant life events recorded yet.\n\n"
            "Write diary entries with /diary — I'll build your timeline "
            "from the most meaningful moments."
        )
        return

    try:
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=ChatAction.TYPING
        )
    except TelegramError:
        pass

    # Format events for LLM
    events_text = "\n".join(
        f"[{e.get('created_at', '')[:10]}] "
        f"(importance: {e.get('importance_score', 0.5):.1f}, "
        f"emotions: {e.get('detected_emotions', 'unknown')}) "
        f"{e.get('ai_summary') or e.get('title', 'Entry')}"
        for e in events
    )

    prompt = TIMELINE_PROMPT.format(events=events_text)

    llm = get_llm()
    timeline_text = await llm.chat(
        "You are a personal life historian. Present events meaningfully.",
        prompt,
        max_tokens=600,
    )

    await update.message.reply_text(f"📌 Your Life Timeline\n\n{timeline_text}")


# ── Onboarding & Welcome Helpers ─────────────────────────────────────────────────

async def _send_welcome_message(update: Update, name: str) -> None:
    """Send the full welcome message with all commands."""
    await update.message.reply_text(
        f"Hey {name}! 🌙\n\n"
        "I'm your personal AI Diary Companion — a journal that listens, "
        "remembers, and grows with you.\n\n"
        "Here's what I can do:\n"
        "📝 Daily check-ins to help you reflect\n"
        "📓 Structured diary entries with deep AI analysis\n"
        "🧠 Remember everything you share — forever\n"
        "💭 Spot emotional patterns and habits\n"
        "🔍 Search through your memories anytime\n"
        "📊 Generate life summaries\n\n"
        "📓 Diary Commands:\n"
        "/diary — write a diary entry\n"
        "/diarylatest — see your latest diary entry\n"
        "/diarysearch <query> — search diary entries\n\n"
        "📊 Insights Commands:\n"
        "/mood — see your emotional trends\n"
        "/timeline — view your life timeline\n"
        "/summary — get a life summary\n"
        "/memory — see your memory stats\n\n"
        "🔧 Utility Commands:\n"
        "/search <query> — search your memories\n"
        "/settime HH:MM — set daily reminder time\n"
        "/export — export your diary data\n"
        "/clear — erase all my memory of you\n\n"
        f"Your daily check-in is set for {DEFAULT_REMINDER_TIME}. "
        "Use /settime to change it.\n\n"
        "Start by telling me about your day!"
    )


async def _handle_name_setup(
    update: Update, context: ContextTypes.DEFAULT_TYPE,
    user_id: int, name_text: str
) -> None:
    """Handle the user's name response during first-time setup."""
    name = name_text.strip()

    # Basic validation — take only the first reasonable chunk
    if len(name) > 50:
        name = name[:50]

    # Save the name into the semantic profile
    db = get_db()
    profile = await get_profile(user_id)
    profile["name"] = name
    await db.save_semantic_profile(user_id, profile)

    logger.info("New user %d set their name to: %s", user_id, name)

    # Send the full welcome message
    await _send_welcome_message(update, name)


# ── Main Message Handler ────────────────────────────────────────────────────────

async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle all regular text messages — the main conversation flow."""
    user = update.effective_user
    user_id = user.id
    user_message = update.message.text

    # Cancel any pending /clear confirmation
    context.user_data["clear_pending"] = False

    db = get_db()
    await db.ensure_user(user_id, user.username, user.first_name)

    # ── Check if user is providing their name (first-time setup) ──────────────
    if context.user_data.get("awaiting_name"):
        context.user_data["awaiting_name"] = False
        await _handle_name_setup(update, context, user_id, user_message)
        return

    # ── Check if user is in diary mode ────────────────────────────────────────
    if context.user_data.get("diary_mode"):
        context.user_data["diary_mode"] = False
        await _handle_diary_entry(update, context, user_id, user_message)
        return

    # Show typing indicator
    try:
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=ChatAction.TYPING
        )
    except TelegramError as exc:
        logger.warning("Typing indicator failed: %s", exc)

    # ── Step 1: Build context from all memory layers ─────────────────────────
    memory_context = await build_context(user_id, user_message)

    # ── Step 2: Construct the full system prompt ─────────────────────────────
    if memory_context:
        full_system = SYSTEM_PROMPT + "\n\n" + memory_context
    else:
        full_system = SYSTEM_PROMPT

    # ── Step 3: Call LLM ─────────────────────────────────────────────────────
    llm = get_llm()
    try:
        bot_response = await llm.chat(full_system, user_message)
    except Exception as exc:
        logger.error("LLM call failed for user %d: %s", user_id, exc)
        await update.message.reply_text(
            "I'm having trouble thinking right now. Could you try again in a moment? 🤔"
        )
        return

    # ── Step 4: Send reply ───────────────────────────────────────────────────
    await update.message.reply_text(bot_response)

    # ── Step 5: Post-response pipeline (non-blocking) ────────────────────────
    asyncio.create_task(_post_response_pipeline(user_id, user_message, bot_response))


async def _handle_diary_entry(
    update: Update, context: ContextTypes.DEFAULT_TYPE,
    user_id: int, diary_text: str
) -> None:
    """Process a diary entry submitted in diary mode."""
    # Show typing — analysis takes a moment
    try:
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=ChatAction.TYPING
        )
    except TelegramError:
        pass

    try:
        result = await process_diary_entry(user_id, diary_text)

        # Send the AI follow-up response
        followup = result.get("followup", "Your diary entry has been saved. 🌙")
        analysis = result.get("analysis", {})

        # Build a rich response
        response_parts = []

        # Analysis header
        emotions = analysis.get("detected_emotions", "")
        importance = analysis.get("importance_score", 0.5)
        title = analysis.get("title", "")

        if title:
            response_parts.append(f"📓 {title}")
        if emotions:
            stars = "⭐" * max(1, round(importance * 5))
            response_parts.append(f"💭 Detected mood: {emotions} {stars}")

        # Goals/stressors extracted
        goals = analysis.get("extracted_goals", [])
        if goals:
            response_parts.append(f"🎯 Goals spotted: {', '.join(goals)}")
        stressors = analysis.get("extracted_stressors", [])
        if stressors:
            response_parts.append(f"⚡ Stressors noted: {', '.join(stressors)}")

        response_parts.append("")
        response_parts.append(followup)

        await update.message.reply_text("\n".join(response_parts))

        # Also save as a conversation episode for continuity
        asyncio.create_task(
            _post_response_pipeline(user_id, f"[DIARY ENTRY] {diary_text}", followup)
        )

    except Exception as exc:
        logger.error("Diary entry processing failed for user %d: %s", user_id, exc)
        await update.message.reply_text(
            "I saved your diary entry but had trouble with the analysis. "
            "Your words are safe — I'll analyze them properly next time. 🌙"
        )


async def _post_response_pipeline(
    user_id: int, user_message: str, bot_response: str
) -> None:
    """
    Background pipeline that runs after each response:
    1. Analyze emotion
    2. Save episode with emotion + topics
    3. Update semantic profile
    4. Check if summaries need generating
    """
    try:
        # 1. Emotion analysis
        emotion_data = await analyze_emotion(user_message, bot_response)

        # 2. Save the episode
        await save_episode(
            user_id=user_id,
            user_message=user_message,
            bot_response=bot_response,
            detected_emotion=emotion_data["emotion"],
            emotion_confidence=emotion_data["confidence"],
            secondary_emotion=emotion_data.get("secondary_emotion"),
            topics=emotion_data.get("topics"),
        )

        # 3. Update semantic profile
        await update_profile_from_conversation(user_id, user_message, bot_response)

        # 4. Check and generate summaries
        await check_and_generate_summaries(user_id)

    except Exception as exc:
        logger.error(
            "Post-response pipeline failed for user %d: %s", user_id, exc,
            exc_info=True,
        )


# ── Application Entry Point ─────────────────────────────────────────────────────

async def post_init(app) -> None:
    """Called after the application is initialized."""
    db = get_db()
    await db.initialize()
    logger.info("Database initialized")

    await setup_scheduler(app)
    logger.info("Scheduler initialized")


async def post_shutdown(app) -> None:
    """Called during application shutdown."""
    await shutdown_scheduler()
    db = get_db()
    await db.create_backup()
    await db.close()
    logger.info("Graceful shutdown complete")


def main() -> None:
    """Build and launch the Telegram application."""
    app = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    # Register command handlers
    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("memory", memory_handler))
    app.add_handler(CommandHandler("clear", clear_handler))
    app.add_handler(CommandHandler("settime", settime_handler))
    app.add_handler(CommandHandler("summary", summary_handler))
    app.add_handler(CommandHandler("search", search_handler))
    app.add_handler(CommandHandler("export", export_handler))

    # Diary command handlers
    app.add_handler(CommandHandler("diary", diary_handler))
    app.add_handler(CommandHandler("diarylatest", diarylatest_handler))
    app.add_handler(CommandHandler("diarysearch", diarysearch_handler))
    app.add_handler(CommandHandler("mood", mood_handler))
    app.add_handler(CommandHandler("timeline", timeline_handler))

    # Register the message handler — all text that isn't a command
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))

    logger.info("🌙 AI Diary Companion is starting...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
