# AI Diary Companion 🌙📓

A deeply personalized, persistent AI diary and long-term advisor on Telegram. Unlike standard chatbots, this companion uses a multi-layered memory system to remember your emotions, goals, stressors, and life events, providing context-aware advice and reflective journaling support.

## ✨ Features

### 🧠 Advanced Multi-Layered Memory
- **Episodic Memory**: Every conversation is stored permanently.
- **Semantic Profile**: Automatically builds and updates a structured profile of who you are (goals, habits, stressors, relationships).
- **Diary Entry System**: Dedicated long-form reflective journaling with deep AI analysis.
- **Rolling Summaries**: Generates daily, weekly, and monthly summaries of your life.
- **Emotional Intelligence**: Detects 18+ emotional states and tracks patterns over time.

### 📝 Reflective Journaling
- **Structured Entries**: Use `/diary` to write deep reflections that are analyzed for meaning and importance.
- **Importance Scoring**: AI identifies major life milestones and breakthroughs.
- **AI Follow-ups**: Receive warm, empathetic, and reflective questions after every entry.
- **Life Timeline**: View a chronological history of your most important moments with `/timeline`.

### 📊 Insights & Reflection
- **Mood Tracking**: View your emotional trajectory and triggers with `/mood`.
- **Context-Aware Advisor**: The bot uses all historical context to give deeply personalized advice.
- **Daily Check-ins**: Scheduled reminders to encourage consistent journaling.

### 🔒 Privacy & Resilience
- **SQLite WAL Storage**: Transactional database ensures zero data loss even during crashes.
- **Automated Backups**: Periodic snapshots of your entire memory database.
- **Data Export**: Export your entire history in readable JSON or Markdown formats with `/export`.
- **Self-Destruct**: Total data wipe available via `/clear`.

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| **Bot Framework** | `python-telegram-bot` v21+ |
| **LLM Gateway** | [OpenRouter](https://openrouter.ai/) (Gemini 2.5 Flash / Pro) |
| **Database** | SQLite (WAL Mode) with `aiosqlite` |
| **Scheduling** | `APScheduler` |
| **Package Manager** | `uv` |
| **Language** | Python 3.12+ |

## 🚀 Quick Start

### 1. Prerequisites
- [uv](https://github.com/astral-sh/uv) installed.
- A Telegram bot token from [@BotFather](https://t.me/botfather).
- An API key from [OpenRouter](https://openrouter.ai/).

### 2. Setup
Clone the repository and install dependencies:
```bash
uv sync
```

### 3. Configuration
Create a `.env` file in the root directory:
```env
TELEGRAM_BOT_TOKEN=your_telegram_token
OPENROUTER_API_KEY=your_openrouter_key
# Optional: Set preferred model
# LLM_MODEL=google/gemini-2.0-flash-001
```

### 4. Run the Bot
```bash
uv run python bot.py
```

## 🎮 Commands

| Command | Description |
|---|---|
| `/start` | Welcome, name setup, and capability overview |
| `/diary` | Enter diary mode for a structured entry |
| `/diarylatest` | View your most recent diary analysis |
| `/diarysearch <q>` | Search through your diary entries |
| `/mood` | View your emotional trends and report |
| `/timeline` | See your chronological life timeline |
| `/summary` | Generate a life summary (daily/weekly/monthly) |
| `/memory` | View your memory statistics |
| `/search <query>` | Search your conversation history |
| `/settime HH:MM` | Set your daily check-in reminder time |
| `/export` | Export your data (JSON + Markdown) |
| `/clear` | Permanently erase all your data |

## 📂 Project Structure

- `bot.py`: Main Telegram handlers and message routing.
- `database.py`: Core SQLite logic and schema management.
- `diary_engine.py`: Deep analysis pipeline for diary entries.
- `retrieval_engine.py`: 7-layer hybrid memory retrieval.
- `semantic_profile.py`: User profile extraction and storage.
- `emotion_engine.py`: Pattern detection and emotion analysis.
- `scheduler.py`: Daily reminders and periodic backups.
- `summarizer.py`: Rolling life summary generation.
- `llm_client.py`: Resilient async LLM interface.

## ⚙️ Customization
Edit `config.py` to adjust:
- Memory context window size.
- Emotional analysis sensitivity.
- Default check-in times and timezones.
- Backup intervals.

## 📜 License
MIT
