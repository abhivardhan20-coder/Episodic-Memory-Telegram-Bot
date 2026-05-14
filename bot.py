import os
import logging
from dotenv import load_dotenv
from openai import OpenAI
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
from memory_engine import (
    build_context_prompt,
    save_episode,
    clear_memory,
    get_memory_summary,
)

# ── Environment & clients ──────────────────────────────────────────────────────
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

llm_client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ── OpenRouter LLM integration ─────────────────────────────────────────────────
def ask_llm(user_message: str, context_prompt: str) -> str:
    """Send a message to OpenRouter LLM with episodic memory context injected."""
    system_instruction = (
        "You are a helpful, friendly AI assistant on Telegram.\n"
        "You have a great memory and always refer back to what the user has told you before.\n"
        "When the user shares something personal - their name, job, interests, problems - you remember it.\n"
        "Keep responses concise for a messaging app. Use plain text, no markdown."
    )

    # If we have past context, append it to the system message
    if context_prompt:
        full_system = system_instruction + "\n\n" + context_prompt
    else:
        full_system = system_instruction

    response = llm_client.chat.completions.create(
        model="google/gemini-2.5-flash",
        messages=[
            {"role": "system", "content": full_system},
            {"role": "user", "content": user_message},
        ],
        max_completion_tokens=500,
        temperature=0.7,
    )

    return response.choices[0].message.content


# ── Command handlers ───────────────────────────────────────────────────────────
async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/start command — greet the user."""
    name = update.effective_user.first_name
    await update.message.reply_text(
        f"Hey {name}! I am your personal AI assistant.\n\n"
        "I remember everything you tell me - even between sessions.\n\n"
        "Commands:\n"
        "/memory - see how much I remember\n"
        "/clear - wipe my memory of you\n\n"
        "Just start chatting!"
    )


async def memory_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/memory command — show the user their memory stats."""
    user_id = update.effective_user.id
    summary = get_memory_summary(user_id)
    await update.message.reply_text(f"My memory of you:\n{summary}")


async def clear_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/clear command — wipe all memory for this user."""
    user_id = update.effective_user.id
    clear_memory(user_id)
    await update.message.reply_text(
        "Done. I have forgotten everything about you. Fresh start!"
    )


# ── Main message handler ───────────────────────────────────────────────────────
async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle regular text messages — the main conversation flow."""
    user_id = update.effective_user.id
    user_message = update.message.text

    # Show typing indicator while we process
    try:
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action=ChatAction.TYPING,
        )
    except TelegramError as exc:
        logger.warning("Typing indicator failed: %s", exc)

    # Step 1: Load this user's episodic memory as context
    context_prompt = build_context_prompt(user_id)

    # Step 2: Call LLM with the user's message + memory context
    bot_response = ask_llm(user_message, context_prompt)

    # Step 3: Send the reply
    await update.message.reply_text(bot_response)

    # Step 4: Save this exchange as a new episode
    save_episode(user_id, user_message, bot_response)


# ── Application entry point ────────────────────────────────────────────────────
def main() -> None:
    """Build and launch the Telegram application."""
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Register command handlers
    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("memory", memory_handler))
    app.add_handler(CommandHandler("clear", clear_handler))

    # Register the message handler — catches all text that is NOT a command
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))

    logger.info("Bot is running. Press Ctrl+C to stop.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
