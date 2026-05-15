"""
Telegram Bot core logic and handler registration.
Modified for production webhook usage.
"""

import logging
import asyncio
from telegram import Update
from telegram.ext import (
    Application, ApplicationBuilder, CommandHandler, 
    MessageHandler, filters, ContextTypes
)
from telegram.constants import ChatAction

from app.config import TELEGRAM_BOT_TOKEN, DEFAULT_TIMEZONE, DEFAULT_REMINDER_TIME
from app.database import get_db
from app.memory_engine import get_memory_summary, save_episode, analyze_emotion
from app.retrieval_engine import build_context
from app.semantic_engine import get_profile, update_profile_from_conversation, profile_to_context_string
from app.diary_engine import process_diary_entry, get_diary_summary, format_diary_entry_display
from app.scheduler import schedule_user_reminder
from app.prompts import SYSTEM_PROMPT, DIARY_ENTRY_INTRO
from app.utils import get_llm, check_and_generate_summaries

logger = logging.getLogger(__name__)

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    db = get_db()
    await db.ensure_user(user.id, user.username, user.first_name)
    await schedule_user_reminder(context.application, user.id, DEFAULT_REMINDER_TIME, DEFAULT_TIMEZONE)
    
    profile = await get_profile(user.id)
    name = profile.get("name") or user.first_name or "there"
    
    await update.message.reply_text(
        f"Hey {name}! 🌙 I'm your AI Diary Companion. I'll remember our chats and help you reflect.\n\n"
        "Commands:\n/diary - Write a entry\n/mood - Recent trends\n/memory - Stats\n/settime HH:MM - Daily reminder"
    )

async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text
    
    if context.user_data.get("diary_mode"):
        context.user_data["diary_mode"] = False
        await update.message.reply_chat_action(ChatAction.TYPING)
        res = await process_diary_entry(user_id, text)
        await update.message.reply_text(res["followup"])
        return

    await update.message.reply_chat_action(ChatAction.TYPING)
    mem_context = await build_context(user_id, text)
    full_prompt = f"{SYSTEM_PROMPT}\n\n{mem_context}"
    
    try:
        response = await get_llm().chat(full_prompt, text)
        await update.message.reply_text(response)
        
        # Post-process
        async def post_process():
            emo = await analyze_emotion(text, response)
            await save_episode(user_id, text, response, **emo)
            await update_profile_from_conversation(user_id, text, response)
            await check_and_generate_summaries(user_id)
        
        asyncio.create_task(post_process())
    except Exception as e:
        logger.error("Chat failed: %s", e)
        await update.message.reply_text("I'm a bit overwhelmed right now. Try again soon? 🤔")

async def diary_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["diary_mode"] = True
    await update.message.reply_text(DIARY_ENTRY_INTRO)

async def memory_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    summary = await get_memory_summary(update.effective_user.id)
    d_summary = await get_diary_summary(update.effective_user.id)
    await update.message.reply_text(f"{summary}\n\n{d_summary}")

async def settime_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /settime HH:MM")
        return
    try:
        await schedule_user_reminder(context.application, update.effective_user.id, context.args[0])
        await update.message.reply_text(f"✅ Reminder set to {context.args[0]}")
    except:
        await update.message.reply_text("❌ Invalid format. Use HH:MM")

def build_ptb_application() -> Application:
    """Factory to build the PTB application."""
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("diary", diary_handler))
    app.add_handler(CommandHandler("memory", memory_handler))
    app.add_handler(CommandHandler("settime", settime_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))
    
    return app
