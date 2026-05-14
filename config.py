"""
Centralized configuration for the AI Diary Companion.
All environment variables, paths, and constants live here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Secrets ─────────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")

# ── LLM Settings ────────────────────────────────────────────────────────────────
LLM_BASE_URL: str = "https://openrouter.ai/api/v1"
LLM_MODEL: str = os.getenv("LLM_MODEL", "google/gemini-2.5-flash")
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "800"))
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_RETRY_ATTEMPTS: int = 3
LLM_RETRY_BASE_DELAY: float = 1.0  # seconds, doubles each retry

# ── Paths ────────────────────────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).parent
DATA_DIR: Path = BASE_DIR / "data"
BACKUP_DIR: Path = BASE_DIR / "backups"
EXPORT_DIR: Path = BASE_DIR / "exports"
LOG_DIR: Path = BASE_DIR / "logs"

# Create directories on import
for d in (DATA_DIR, BACKUP_DIR, EXPORT_DIR, LOG_DIR):
    d.mkdir(exist_ok=True)

DB_PATH: Path = DATA_DIR / "diary.db"

# ── Timezone & Scheduling ───────────────────────────────────────────────────────
DEFAULT_TIMEZONE: str = os.getenv("DEFAULT_TIMEZONE", "Asia/Kolkata")
DEFAULT_REMINDER_TIME: str = "22:00"  # 10:00 PM
BACKUP_INTERVAL_HOURS: int = int(os.getenv("BACKUP_INTERVAL_HOURS", "6"))

# ── Memory & Retrieval Settings ─────────────────────────────────────────────────
RECENT_EPISODES_COUNT: int = 5       # recent episodes injected into context
TOPIC_MATCH_COUNT: int = 3           # topic-matched episodes from history
EMOTION_MATCH_COUNT: int = 3         # emotionally similar episodes
SUMMARY_COUNT: int = 2               # recent summaries to include
MAX_CONTEXT_CHARS: int = 12000       # rough char limit for context window

# ── Logging ──────────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s"
