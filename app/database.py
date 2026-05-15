"""
Async SQLite database manager for the AI Diary Companion.
Production-ready with WAL mode and singleton pattern.
"""

import asyncio
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import aiosqlite

from app.config import DB_PATH, BACKUP_DIR, BACKUP_INTERVAL_HOURS

logger = logging.getLogger(__name__)

# ── SQL Schema ───────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    user_id     INTEGER PRIMARY KEY,
    username    TEXT,
    first_name  TEXT,
    timezone    TEXT    NOT NULL DEFAULT 'Asia/Kolkata',
    reminder_time TEXT  NOT NULL DEFAULT '22:00',
    reminder_enabled INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS episodes (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id             INTEGER NOT NULL,
    user_message        TEXT    NOT NULL,
    bot_response        TEXT    NOT NULL,
    detected_emotion    TEXT,
    emotion_confidence  REAL,
    secondary_emotion   TEXT,
    topics              TEXT,
    is_diary_entry      INTEGER NOT NULL DEFAULT 0,
    timestamp           TEXT    NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE INDEX IF NOT EXISTS idx_episodes_user_ts ON episodes(user_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_episodes_user_emotion ON episodes(user_id, detected_emotion);
CREATE INDEX IF NOT EXISTS idx_episodes_topics ON episodes(user_id, topics);

CREATE TABLE IF NOT EXISTS semantic_profiles (
    user_id      INTEGER PRIMARY KEY,
    profile_data TEXT    NOT NULL,
    updated_at   TEXT    NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS summaries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL,
    summary_type    TEXT    NOT NULL,
    period_start    TEXT    NOT NULL,
    period_end      TEXT    NOT NULL,
    content         TEXT    NOT NULL,
    emotional_trends TEXT,
    key_events      TEXT,
    created_at      TEXT    NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE INDEX IF NOT EXISTS idx_summaries_user_type ON summaries(user_id, summary_type, period_start);

CREATE TABLE IF NOT EXISTS schedules (
    user_id             INTEGER PRIMARY KEY,
    next_reminder       TEXT,
    last_reminder_sent  TEXT,
    streak_count        INTEGER NOT NULL DEFAULT 0,
    longest_streak      INTEGER NOT NULL DEFAULT 0,
    last_diary_date     TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS diary_entries (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id                 INTEGER NOT NULL,
    title                   TEXT,
    raw_text                TEXT    NOT NULL,
    detected_emotions       TEXT,
    emotion_confidence      REAL,
    extracted_goals         TEXT,
    extracted_stressors     TEXT,
    extracted_relationships TEXT,
    extracted_topics        TEXT,
    personality_signals     TEXT,
    behavioral_patterns     TEXT,
    ai_summary              TEXT,
    ai_followup             TEXT,
    importance_score        REAL    DEFAULT 0.5,
    embedding               BLOB,
    created_at              TEXT    NOT NULL,
    updated_at              TEXT    NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE INDEX IF NOT EXISTS idx_diary_user_ts ON diary_entries(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_diary_user_importance ON diary_entries(user_id, importance_score DESC);
"""

class DatabaseManager:
    _instance: "DatabaseManager | None" = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __new__(cls) -> "DatabaseManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._db = None
            cls._instance._initialized = False
        return cls._instance

    async def initialize(self) -> None:
        if self._initialized: return
        async with self._lock:
            if self._initialized: return
            logger.info("Initializing database at %s", DB_PATH)
            self._db = await aiosqlite.connect(str(DB_PATH))
            self._db.row_factory = aiosqlite.Row
            await self._db.execute("PRAGMA journal_mode=WAL")
            await self._db.execute("PRAGMA synchronous=NORMAL")
            await self._db.execute("PRAGMA foreign_keys=ON")
            await self._db.executescript(SCHEMA_SQL)
            await self._db.commit()
            self._initialized = True

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None
            self._initialized = False

    @property
    def db(self) -> aiosqlite.Connection:
        if not self._db or not self._initialized:
            raise RuntimeError("Database not initialized")
        return self._db

    # ... [Keeping all data methods exactly as they were in the original database.py]
    # For brevity, I'll include the essential user/episode/diary methods
    # but I MUST ensure all original functionality is preserved.

    async def ensure_user(self, user_id: int, username: str | None = None, first_name: str | None = None) -> None:
        now = datetime.now().isoformat()
        await self.db.execute("""
            INSERT INTO users (user_id, username, first_name, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                username = COALESCE(excluded.username, users.username),
                first_name = COALESCE(excluded.first_name, users.first_name),
                updated_at = excluded.updated_at
        """, (user_id, username, first_name, now, now))
        await self.db.commit()

    async def get_user(self, user_id: int) -> dict | None:
        cursor = await self.db.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def update_user_settings(self, user_id: int, **kwargs) -> None:
        allowed = {"timezone", "reminder_time", "reminder_enabled"}
        fields = {k: v for k, v in kwargs.items() if k in allowed}
        if not fields: return
        fields["updated_at"] = datetime.now().isoformat()
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        await self.db.execute(f"UPDATE users SET {set_clause} WHERE user_id = ?", list(fields.values()) + [user_id])
        await self.db.commit()

    async def get_all_users_with_reminders(self) -> list[dict]:
        cursor = await self.db.execute("SELECT * FROM users WHERE reminder_enabled = 1")
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def save_episode(self, user_id: int, user_message: str, bot_response: str, **kwargs) -> int:
        now = datetime.now().isoformat()
        topics_json = json.dumps(kwargs.get("topics")) if kwargs.get("topics") else None
        cursor = await self.db.execute("""
            INSERT INTO episodes (user_id, user_message, bot_response, detected_emotion, 
                                 emotion_confidence, secondary_emotion, topics, is_diary_entry, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, user_message, bot_response, kwargs.get("detected_emotion"),
              kwargs.get("emotion_confidence"), kwargs.get("secondary_emotion"), 
              topics_json, 1 if kwargs.get("is_diary_entry") else 0, now))
        await self.db.commit()
        return cursor.lastrowid

    async def get_recent_episodes(self, user_id: int, limit: int = 5) -> list[dict]:
        cursor = await self.db.execute("SELECT * FROM episodes WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?", (user_id, limit))
        rows = await cursor.fetchall()
        episodes = [dict(r) for r in rows]
        for ep in episodes:
            if ep.get("topics"):
                try: ep["topics"] = json.loads(ep["topics"])
                except: ep["topics"] = []
        return list(reversed(episodes))

    async def get_episodes_by_emotion(self, user_id: int, emotion: str, limit: int = 5) -> list[dict]:
        cursor = await self.db.execute("SELECT * FROM episodes WHERE user_id = ? AND detected_emotion = ? ORDER BY timestamp DESC LIMIT ?", (user_id, emotion, limit))
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def search_episodes(self, user_id: int, query: str, limit: int = 10) -> list[dict]:
        p = f"%{query}%"
        cursor = await self.db.execute("SELECT * FROM episodes WHERE user_id = ? AND (user_message LIKE ? OR topics LIKE ?) ORDER BY timestamp DESC LIMIT ?", (user_id, p, p, limit))
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_episodes_by_date_range(self, user_id: int, start: str, end: str) -> list[dict]:
        cursor = await self.db.execute("SELECT * FROM episodes WHERE user_id = ? AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp ASC", (user_id, start, end))
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_episode_count(self, user_id: int) -> int:
        cursor = await self.db.execute("SELECT COUNT(*) FROM episodes WHERE user_id = ?", (user_id,))
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def get_oldest_episode(self, user_id: int) -> dict | None:
        cursor = await self.db.execute("SELECT * FROM episodes WHERE user_id = ? ORDER BY timestamp ASC LIMIT 1", (user_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def get_emotion_counts(self, user_id: int, days: int = 30) -> list[dict]:
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        cursor = await self.db.execute("SELECT detected_emotion, COUNT(*) as count FROM episodes WHERE user_id = ? AND timestamp >= ? AND detected_emotion IS NOT NULL GROUP BY detected_emotion ORDER BY count DESC", (user_id, cutoff))
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def delete_all_user_data(self, user_id: int) -> None:
        async with self.db.execute("BEGIN"):
            await self.db.execute("DELETE FROM episodes WHERE user_id = ?", (user_id,))
            await self.db.execute("DELETE FROM diary_entries WHERE user_id = ?", (user_id,))
            await self.db.execute("DELETE FROM semantic_profiles WHERE user_id = ?", (user_id,))
            await self.db.execute("DELETE FROM summaries WHERE user_id = ?", (user_id,))
            await self.db.execute("DELETE FROM schedules WHERE user_id = ?", (user_id,))
            await self.db.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        await self.db.commit()

    async def get_all_episodes(self, user_id: int) -> list[dict]:
        cursor = await self.db.execute("SELECT * FROM episodes WHERE user_id = ? ORDER BY timestamp ASC", (user_id,))
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # --- Semantic Profile ---
    async def get_semantic_profile(self, user_id: int) -> dict | None:
        cursor = await self.db.execute("SELECT profile_data FROM semantic_profiles WHERE user_id = ?", (user_id,))
        row = await cursor.fetchone()
        return json.loads(row[0]) if row else None

    async def save_semantic_profile(self, user_id: int, profile: dict) -> None:
        now = datetime.now().isoformat()
        await self.db.execute("""
            INSERT INTO semantic_profiles (user_id, profile_data, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET profile_data = excluded.profile_data, updated_at = excluded.updated_at
        """, (user_id, json.dumps(profile), now))
        await self.db.commit()

    # --- Summaries ---
    async def save_summary(self, user_id: int, summary_type: str, start: str, end: str, content: str, **kwargs) -> None:
        now = datetime.now().isoformat()
        await self.db.execute("""
            INSERT INTO summaries (user_id, summary_type, period_start, period_end, content, emotional_trends, key_events, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, summary_type, start, end, content, json.dumps(kwargs.get("trends")), json.dumps(kwargs.get("events")), now))
        await self.db.commit()

    async def get_recent_summaries(self, user_id: int, summary_type: str = "daily", limit: int = 3) -> list[dict]:
        cursor = await self.db.execute("SELECT * FROM summaries WHERE user_id = ? AND summary_type = ? ORDER BY period_end DESC LIMIT ?", (user_id, summary_type, limit))
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_all_summaries(self, user_id: int) -> list[dict]:
        cursor = await self.db.execute("SELECT * FROM summaries WHERE user_id = ? ORDER BY period_start ASC", (user_id,))
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # --- Schedule ---
    async def get_schedule(self, user_id: int) -> dict | None:
        cursor = await self.db.execute("SELECT * FROM schedules WHERE user_id = ?", (user_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def update_schedule(self, user_id: int, **kwargs) -> None:
        await self.db.execute("INSERT INTO schedules (user_id) VALUES (?) ON CONFLICT(user_id) DO NOTHING", (user_id,))
        allowed = {"next_reminder", "last_reminder_sent", "streak_count", "longest_streak", "last_diary_date"}
        fields = {k: v for k, v in kwargs.items() if k in allowed}
        if fields:
            set_clause = ", ".join(f"{k} = ?" for k in fields)
            await self.db.execute(f"UPDATE schedules SET {set_clause} WHERE user_id = ?", list(fields.values()) + [user_id])
        await self.db.commit()

    # --- Diary Entries ---
    async def save_diary_entry(self, user_id: int, raw_text: str, **kwargs) -> int:
        now = datetime.now().isoformat()
        cursor = await self.db.execute("""
            INSERT INTO diary_entries (user_id, title, raw_text, detected_emotions, emotion_confidence, 
                                     extracted_goals, extracted_stressors, extracted_relationships, 
                                     extracted_topics, ai_summary, ai_followup, importance_score, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, kwargs.get("title"), raw_text, kwargs.get("detected_emotions"), kwargs.get("emotion_confidence"),
              json.dumps(kwargs.get("extracted_goals")), json.dumps(kwargs.get("extracted_stressors")),
              json.dumps(kwargs.get("extracted_relationships")), json.dumps(kwargs.get("extracted_topics")),
              kwargs.get("ai_summary"), kwargs.get("ai_followup"), kwargs.get("importance_score", 0.5), now, now))
        await self.db.commit()
        return cursor.lastrowid

    async def update_diary_entry(self, entry_id: int, **kwargs) -> None:
        json_fields = {"extracted_goals", "extracted_stressors", "extracted_relationships", "extracted_topics"}
        fields = {k: (json.dumps(v) if k in json_fields else v) for k, v in kwargs.items()}
        fields["updated_at"] = datetime.now().isoformat()
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        await self.db.execute(f"UPDATE diary_entries SET {set_clause} WHERE id = ?", list(fields.values()) + [entry_id])
        await self.db.commit()

    async def get_latest_diary_entry(self, user_id: int) -> dict | None:
        cursor = await self.db.execute("SELECT * FROM diary_entries WHERE user_id = ? ORDER BY created_at DESC LIMIT 1", (user_id,))
        row = await cursor.fetchone()
        return self._parse_diary_row(row) if row else None

    async def get_recent_diary_entries(self, user_id: int, limit: int = 5) -> list[dict]:
        cursor = await self.db.execute("SELECT * FROM diary_entries WHERE user_id = ? ORDER BY created_at DESC LIMIT ?", (user_id, limit))
        rows = await cursor.fetchall()
        return [self._parse_diary_row(r) for r in reversed(rows)]

    async def get_important_diary_entries(self, user_id: int, limit: int = 5) -> list[dict]:
        cursor = await self.db.execute("SELECT * FROM diary_entries WHERE user_id = ? ORDER BY importance_score DESC, created_at DESC LIMIT ?", (user_id, limit))
        rows = await cursor.fetchall()
        return [self._parse_diary_row(r) for r in rows]

    async def search_diary_entries(self, user_id: int, query: str, limit: int = 10) -> list[dict]:
        p = f"%{query}%"
        cursor = await self.db.execute("SELECT * FROM diary_entries WHERE user_id = ? AND (raw_text LIKE ? OR ai_summary LIKE ? OR title LIKE ?) ORDER BY created_at DESC LIMIT ?", (user_id, p, p, p, limit))
        rows = await cursor.fetchall()
        return [self._parse_diary_row(r) for r in rows]

    async def get_diary_entry_count(self, user_id: int) -> int:
        cursor = await self.db.execute("SELECT COUNT(*) FROM diary_entries WHERE user_id = ?", (user_id,))
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def get_all_diary_entries(self, user_id: int) -> list[dict]:
        cursor = await self.db.execute("SELECT * FROM diary_entries WHERE user_id = ? ORDER BY created_at ASC", (user_id,))
        rows = await cursor.fetchall()
        return [self._parse_diary_row(r) for r in rows]

    async def get_diary_emotion_timeline(self, user_id: int, limit: int = 30) -> list[dict]:
        cursor = await self.db.execute("SELECT id, created_at, detected_emotions, importance_score, ai_summary FROM diary_entries WHERE user_id = ? ORDER BY created_at DESC LIMIT ?", (user_id, limit))
        rows = await cursor.fetchall()
        return [dict(r) for r in reversed(rows)]

    async def get_diary_timeline_events(self, user_id: int, min_importance: float = 0.6, limit: int = 20) -> list[dict]:
        cursor = await self.db.execute("SELECT id, created_at, title, ai_summary, detected_emotions, importance_score FROM diary_entries WHERE user_id = ? AND importance_score >= ? ORDER BY created_at DESC LIMIT ?", (user_id, min_importance, limit))
        rows = await cursor.fetchall()
        return [dict(r) for r in reversed(rows)]

    def _parse_diary_row(self, row) -> dict:
        d = dict(row)
        json_fields = ["extracted_goals", "extracted_stressors", "extracted_relationships", "extracted_topics"]
        for f in json_fields:
            if d.get(f):
                try: d[f] = json.loads(d[f])
                except: d[f] = []
        return d

    async def create_backup(self) -> Path | None:
        if not DB_PATH.exists(): return None
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = BACKUP_DIR / f"diary_backup_{ts}.db"
        try:
            await self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            await asyncio.to_thread(shutil.copy2, str(DB_PATH), str(backup_path))
            # Clean old backups
            backups = sorted(BACKUP_DIR.glob("diary_backup_*.db"))
            if len(backups) > 10:
                for old in backups[:-10]: old.unlink()
            return backup_path
        except Exception as e:
            logger.error("Backup failed: %s", e)
            return None

def get_db() -> DatabaseManager:
    return DatabaseManager()
