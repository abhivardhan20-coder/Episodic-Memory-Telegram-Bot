# PRD & Complete Code — Telegram Bot with Episodic Memory

---

## PRODUCT REQUIREMENTS DOCUMENT (PRD)

### Overview
A Telegram bot that remembers users across sessions using episodic memory — not just within a single conversation, but every time they return. Each exchange is stored as a timestamped episode per user and injected as context into every new LLM call.

### Goals
- Build a personal AI assistant on Telegram that feels like it genuinely knows you
- Demonstrate how memory separates a toy chatbot from a real AI assistant

### Features
| Feature | Description |
|---|---|
| Persistent episodic memory | Stores each user message + bot response as a timestamped episode in a per-user JSON file |
| Context injection | Injects the last 8 episodes into the system prompt before every LLM call |
| Rolling window | Keeps only the most recent 50 episodes per user; trims older ones automatically |
| `/start` command | Greets the user by Telegram first name, explains capabilities |
| `/memory` command | Shows the user how many episodes are stored and when the oldest one was recorded |
| `/clear` command | Wipes all stored memory for that user |
| Groq LLM integration | Uses `llama-3.3-70b-versatile` via Groq's OpenAI-compatible API (free tier, no credit card) |
| Typing indicator | Shows Telegram's animated "typing…" while Groq processes the request |

### Tech Stack
| Layer | Tool |
|---|---|
| Bot framework | `python-telegram-bot` v21+ |
| LLM API | Groq (`llama-3.3-70b-versatile`) |
| Memory storage | Local JSON files (one per user, named by Telegram user ID) |
| Env management | `python-dotenv` |
| Package manager | `uv` |
| Language | Python 3.10+ |

### Non-Functional Requirements
- Free tier only — no credit card for Groq or Telegram
- No database required; plain JSON files
- Runs locally with long polling; can be switched to webhook for production
- Memory files never grow unbounded (MAX_EPISODES = 50 cap)

### Project Structure
```
telegram-memory-bot/
├── bot.py              ← Telegram bot logic + Groq integration
├── memory_engine.py    ← Episodic memory: save, load, build context
├── memory/             ← Auto-created; one JSON file per user
├── .env                ← TELEGRAM_BOT_TOKEN + GROQ_API_KEY
└── pyproject.toml      ← Managed by uv
```

### Prerequisites
- Python 3.10+
- `uv` package manager
- A Telegram account
- A free Groq API key from https://console.groq.com/keys

---

## SETUP COMMANDS

```bash
# 1. Create project
uv init telegram-memory-bot
cd telegram-memory-bot

# 2. Install dependencies
uv add python-telegram-bot groq python-dotenv

# 3. Create file structure
mkdir memory
touch bot.py memory_engine.py .env

# 4. Protect secrets
echo ".env" >> .gitignore
```

---

## ENVIRONMENT VARIABLES (.env)

```env
TELEGRAM_BOT_TOKEN=your_token_here
GROQ_API_KEY=your_groq_key_here
```

Get your `TELEGRAM_BOT_TOKEN` from @BotFather on Telegram (`/newbot`).
Get your `GROQ_API_KEY` from https://console.groq.com/keys (free, no credit card).

---

## CODE: memory_engine.py

```python
import json
import os
from datetime import datetime
from pathlib import Path

MEMORY_DIR = Path("memory")
MEMORY_DIR.mkdir(exist_ok=True)

MAX_EPISODES = 50       # Maximum episodes to keep per user
CONTEXT_EPISODES = 8    # How many recent episodes to inject into each prompt


def get_memory_path(user_id: int) -> Path:
    """Returns the file path for a given user's memory."""
    return MEMORY_DIR / f"{user_id}.json"


def load_episodes(user_id: int) -> list[dict]:
    """Load all stored episodes for a user. Returns empty list if none exist."""
    path = get_memory_path(user_id)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_episode(user_id: int, user_message: str, bot_response: str) -> None:
    """Save a new episode to the user's memory file."""
    episodes = load_episodes(user_id)

    new_episode = {
        "timestamp": datetime.now().isoformat(),
        "user_message": user_message,
        "bot_response": bot_response,
    }

    episodes.append(new_episode)

    # Trim oldest episodes if we exceed the limit
    if len(episodes) > MAX_EPISODES:
        episodes = episodes[-MAX_EPISODES:]

    path = get_memory_path(user_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(episodes, f, indent=2, ensure_ascii=False)


def clear_memory(user_id: int) -> None:
    """Delete all stored episodes for a user."""
    path = get_memory_path(user_id)
    if path.exists():
        path.unlink()


def build_context_prompt(user_id: int) -> str:
    """
    Build a context string from recent episodes to inject into the system prompt.
    Returns empty string if no history exists.
    """
    episodes = load_episodes(user_id)

    if not episodes:
        return ""

    # Take only the most recent N episodes
    recent = episodes[-CONTEXT_EPISODES:]

    lines = ["Here is your conversation history with this user (most recent last):"]
    lines.append("")

    for ep in recent:
        ts = ep["timestamp"][:16].replace("T", " ")  # Format: 2026-04-14 09:32
        lines.append(f"[{ts}]")
        lines.append(f"User: {ep['user_message']}")
        lines.append(f"You: {ep['bot_response']}")
        lines.append("")

    lines.append("Use this history to give a personalized, context-aware response.")
    return "\n".join(lines)


def get_memory_summary(user_id: int) -> str:
    """Returns a human-readable summary of how much memory a user has."""
    episodes = load_episodes(user_id)
    if not episodes:
        return "No memory stored yet."
    oldest = episodes[0]["timestamp"][:10]
    return f"{len(episodes)} episodes stored. Oldest: {oldest}."
```

---

## CODE: bot.py

```python
import os
import logging
from dotenv import load_dotenv
from groq import Groq
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

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

groq_client = Groq(api_key=GROQ_API_KEY)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ── Groq integration ───────────────────────────────────────────────────────────
def ask_groq(user_message: str, context_prompt: str) -> str:
    """Send a message to Groq with episodic memory context injected."""
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

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
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

    # Step 2: Call Groq with the user's message + memory context
    bot_response = ask_groq(user_message, context_prompt)

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
```

---

## RUNNING THE BOT

```bash
uv run python bot.py
```

You should see:
```
INFO - Bot is running. Press Ctrl+C to stop.
```

Open Telegram, search for your bot's username, and send `/start`.

---

## COMMON ERRORS & FIXES

| Error | Cause | Fix |
|---|---|---|
| `groq.AuthenticationError: 401` | Wrong API key | Check `.env` — copy key from https://console.groq.com/keys exactly, no extra spaces |
| `groq.RateLimitError: 429` | Exceeded free tier (30 req/min) | Switch to `llama-3.1-8b-instant` for higher rate limits |
| `telegram.error.Conflict` | Two bot instances running | Run `pkill -f "python bot.py"` then restart |
| `KeyError: TELEGRAM_BOT_TOKEN` | `.env` not loaded | Make sure `.env` is in the same folder as `bot.py` and `load_dotenv()` is called first |
| Memory file not created | `memory/` directory missing or no write permission | Run `mkdir memory` and `chmod 755 memory` |

---

## IDEAS TO EXTEND

| Improvement | How |
|---|---|
| Smart summarization | After each exchange, ask Groq to generate a one-line summary ("User prefers Python over JS") and store it alongside the full text |
| Semantic profile | Run a second Groq call after each conversation to extract structured facts (name, job, location, projects) into `profile.json` per user |
| SQLite storage | Replace JSON files with SQLite + `aiosqlite` for atomic writes and indexed queries (ideal for multi-user production bots) |
| Webhook for production | Switch `app.run_polling()` to `app.run_webhook()` with a public HTTPS URL (Railway or Render both give you one free) |
| `/summarize` command | Ask Groq to generate a paragraph summarizing everything it knows about the user from their episode history |
| Swap models | Change `model=` to `llama-3.1-8b-instant` (faster, higher rate limit) or `mixtral-8x7b-32768` (32K context window, more memory per call) |
