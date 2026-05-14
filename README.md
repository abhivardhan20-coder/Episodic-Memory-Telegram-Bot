# Telegram Memory Bot 🧠

A Telegram AI assistant that remembers you across sessions using **episodic memory**. Every exchange is stored as a timestamped episode and injected as context into future conversations.

## ✨ Features

- **Persistent Episodic Memory**: Stores message pairs in per-user JSON files.
- **Context Injection**: Automatically injects the last 8 episodes into the system prompt for every new request.
- **Rolling Memory Window**: Keeps only the most recent 50 episodes per user to prevent context bloat.
- **Smart Commands**:
  - `/start`: Greeting and capability overview.
  - `/memory`: Show statistics about your stored history.
  - `/clear`: Wipe all stored memory for a fresh start.
- **OpenRouter Integration**: Powered by any LLM via OpenRouter (defaults to Gemini 2.5 Flash).
- **Typing Indicators**: Real-time feedback while the AI "thinks".

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Bot Framework | `python-telegram-bot` v21+ |
| LLM Gateway | [OpenRouter](https://openrouter.ai/) (OpenAI-compatible) |
| Memory Storage | Local JSON files |
| Package Manager | `uv` |
| Language | Python 3.10+ |

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
Create a `.env` file in the root directory (one has been created for you during setup):
```env
TELEGRAM_BOT_TOKEN=your_telegram_token
OPENROUTER_API_KEY=your_openrouter_key
```

### 4. Run the Bot
```bash
uv run python bot.py
```

## 📂 Project Structure

```
telegram-memory-bot/
├── bot.py              ← Telegram handlers & LLM logic
├── memory_engine.py    ← Episodic memory management
├── memory/             ← Storage directory (JSON files)
├── .env                ← Secrets (not committed)
└── pyproject.toml      ← Dependencies
```

## ⚙️ Customization

- **Change Model**: Edit the `model=` parameter in `bot.py`.
- **Memory Length**: Adjust `MAX_EPISODES` or `CONTEXT_EPISODES` in `memory_engine.py`.
- **System Prompt**: Customize the AI's personality in `ask_llm` inside `bot.py`.

## 📜 License
MIT
