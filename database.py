"""
Async SQLite database manager for the AI Diary Companion.

Uses aiosqlite with WAL journaling mode for crash-safe, concurrent-read writes.
All mutations are wrapped in transactions for atomicity.
"""

import asyncio
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import aiosqlite

from config import DB_PATH, BACKUP_DIR, BACKUP_INTERVAL_HOURS

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

CREATE INDEX IF NOT EXISTS idx_episodes_user_ts
    ON episodes(user_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_episodes_user_emotion
    ON episodes(user_id, detected_emotion);
CREATE INDEX IF NOT EXISTS idx_episodes_topics
    ON episodes(user_id, topics);

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

CREATE INDEX IF NOT EXISTS idx_summaries_user_type
    ON summaries(user_id, summary_type, period_start);

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

CREATE INDEX IF NOT EXISTS idx_diary_user_ts
    ON diary_entries(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_diary_user_importance
    ON diary_entries(user_id, importance_score DESC);
CREATE INDEX IF NOT EXISTS idx_diary_user_emotions
    ON diary_entries(user_id, detected_emotions);
"""


class DatabaseManager:
    """
    Singleton async SQLite manager.

    Usage:
        db = DatabaseManager()
        await db.initialize()
        ...
        await db.close()
    """

    _instance: "DatabaseManager | None" = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __new__(cls) -> "DatabaseManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._db = None
            cls._instance._initialized = False
        return cls._instance

    # ── Lifecycle ────────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Open the database connection, enable WAL, and create tables."""
        if self._initialized:
            return
        async with self._lock:
            if self._initialized:
                return
            logger.info("Opening database at %s", DB_PATH)
            self._db = await aiosqlite.connect(str(DB_PATH))
            self._db.row_factory = aiosqlite.Row
            await self._db.execute("PRAGMA journal_mode=WAL")
            await self._db.execute("PRAGMA synchronous=NORMAL")
            await self._db.execute("PRAGMA foreign_keys=ON")
            await self._db.executescript(SCHEMA_SQL)
            await self._db.commit()
            self._initialized = True
            logger.info("Database initialized successfully")

    async def close(self) -> None:
        """Safely close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None
            self._initialized = False
            logger.info("Database closed")

    @property
    def db(self) -> aiosqlite.Connection:
        if not self._db or not self._initialized:
            raise RuntimeError("Database not initialized. Call await db.initialize() first.")
        return self._db

    # ── User Management ──────────────────────────────────────────────────────────

    async def ensure_user(
        self, user_id: int, username: str | None = None, first_name: str | None = None
    ) -> None:
        """Create user record if it doesn't exist, otherwise update name fields."""
        now = datetime.now().isoformat()
        await self.db.execute(
            """
            INSERT INTO users (user_id, username, first_name, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                username = COALESCE(excluded.username, users.username),
                first_name = COALESCE(excluded.first_name, users.first_name),
                updated_at = excluded.updated_at
            """,
            (user_id, username, first_name, now, now),
        )
        await self.db.commit()

    async def get_user(self, user_id: int) -> dict | None:
        """Fetch a user record."""
        cursor = await self.db.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def update_user_settings(self, user_id: int, **kwargs) -> None:
        """Update specific user settings (timezone, reminder_time, etc.)."""
        allowed = {"timezone", "reminder_time", "reminder_enabled"}
        fields = {k: v for k, v in kwargs.items() if k in allowed}
        if not fields:
            return
        fields["updated_at"] = datetime.now().isoformat()
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [user_id]
        await self.db.execute(f"UPDATE users SET {set_clause} WHERE user_id = ?", values)
        await self.db.commit()

    async def get_all_users_with_reminders(self) -> list[dict]:
        """Fetch all users who have reminders enabled."""
        cursor = await self.db.execute(
            "SELECT * FROM users WHERE reminder_enabled = 1"
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # ── Episode Storage ──────────────────────────────────────────────────────────

    async def save_episode(
        self,
        user_id: int,
        user_message: str,
        bot_response: str,
        detected_emotion: str | None = None,
        emotion_confidence: float | None = None,
        secondary_emotion: str | None = None,
        topics: list[str] | None = None,
        is_diary_entry: bool = False,
    ) -> int:
        """Save a conversation episode. Returns the episode ID."""
        now = datetime.now().isoformat()
        topics_json = json.dumps(topics) if topics else None
        cursor = await self.db.execute(
            """
            INSERT INTO episodes
                (user_id, user_message, bot_response, detected_emotion,
                 emotion_confidence, secondary_emotion, topics, is_diary_entry, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id, user_message, bot_response, detected_emotion,
                emotion_confidence, secondary_emotion, topics_json,
                1 if is_diary_entry else 0, now,
            ),
        )
        await self.db.commit()
        return cursor.lastrowid

    async def get_recent_episodes(self, user_id: int, limit: int = 5) -> list[dict]:
        """Fetch the N most recent episodes for a user."""
        cursor = await self.db.execute(
            """
            SELECT * FROM episodes
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
        rows = await cursor.fetchall()
        episodes = [dict(r) for r in rows]
        # Parse topics JSON
        for ep in episodes:
            if ep.get("topics"):
                try:
                    ep["topics"] = json.loads(ep["topics"])
                except (json.JSONDecodeError, TypeError):
                    ep["topics"] = []
        return list(reversed(episodes))  # chronological order

    async def get_episodes_by_emotion(
        self, user_id: int, emotion: str, limit: int = 5
    ) -> list[dict]:
        """Fetch episodes matching a specific emotion."""
        cursor = await self.db.execute(
            """
            SELECT * FROM episodes
            WHERE user_id = ? AND detected_emotion = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (user_id, emotion, limit),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def search_episodes(self, user_id: int, query: str, limit: int = 10) -> list[dict]:
        """Full-text search across user messages and bot responses."""
        pattern = f"%{query}%"
        cursor = await self.db.execute(
            """
            SELECT * FROM episodes
            WHERE user_id = ?
              AND (user_message LIKE ? OR bot_response LIKE ? OR topics LIKE ?)
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (user_id, pattern, pattern, pattern, limit),
        )
        rows = await cursor.fetchall()
        episodes = [dict(r) for r in rows]
        for ep in episodes:
            if ep.get("topics"):
                try:
                    ep["topics"] = json.loads(ep["topics"])
                except (json.JSONDecodeError, TypeError):
                    ep["topics"] = []
        return episodes

    async def get_episodes_by_date_range(
        self, user_id: int, start: str, end: str
    ) -> list[dict]:
        """Fetch episodes within a date range (ISO format strings)."""
        cursor = await self.db.execute(
            """
            SELECT * FROM episodes
            WHERE user_id = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
            """,
            (user_id, start, end),
        )
        rows = await cursor.fetchall()
        episodes = [dict(r) for r in rows]
        for ep in episodes:
            if ep.get("topics"):
                try:
                    ep["topics"] = json.loads(ep["topics"])
                except (json.JSONDecodeError, TypeError):
                    ep["topics"] = []
        return episodes

    async def get_episode_count(self, user_id: int) -> int:
        """Total number of stored episodes for a user."""
        cursor = await self.db.execute(
            "SELECT COUNT(*) FROM episodes WHERE user_id = ?", (user_id,)
        )
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def get_oldest_episode(self, user_id: int) -> dict | None:
        """Fetch the very first episode for a user."""
        cursor = await self.db.execute(
            """
            SELECT * FROM episodes
            WHERE user_id = ?
            ORDER BY timestamp ASC
            LIMIT 1
            """,
            (user_id,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def get_emotion_counts(self, user_id: int, days: int = 30) -> list[dict]:
        """Get emotion frequency distribution over the last N days."""
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        cursor = await self.db.execute(
            """
            SELECT detected_emotion, COUNT(*) as count
            FROM episodes
            WHERE user_id = ? AND timestamp >= ? AND detected_emotion IS NOT NULL
            GROUP BY detected_emotion
            ORDER BY count DESC
            """,
            (user_id, cutoff),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def delete_all_user_data(self, user_id: int) -> None:
        """Delete ALL data for a user across all tables."""
        async with self.db.execute("BEGIN"):
            await self.db.execute("DELETE FROM episodes WHERE user_id = ?", (user_id,))
            await self.db.execute("DELETE FROM diary_entries WHERE user_id = ?", (user_id,))
            await self.db.execute("DELETE FROM semantic_profiles WHERE user_id = ?", (user_id,))
            await self.db.execute("DELETE FROM summaries WHERE user_id = ?", (user_id,))
            await self.db.execute("DELETE FROM schedules WHERE user_id = ?", (user_id,))
            await self.db.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        await self.db.commit()
        logger.info("Deleted all data for user %d", user_id)

    async def get_all_episodes(self, user_id: int) -> list[dict]:
        """Fetch ALL episodes for a user (for export)."""
        cursor = await self.db.execute(
            """
            SELECT * FROM episodes
            WHERE user_id = ?
            ORDER BY timestamp ASC
            """,
            (user_id,),
        )
        rows = await cursor.fetchall()
        episodes = [dict(r) for r in rows]
        for ep in episodes:
            if ep.get("topics"):
                try:
                    ep["topics"] = json.loads(ep["topics"])
                except (json.JSONDecodeError, TypeError):
                    ep["topics"] = []
        return episodes

    # ── Semantic Profile ─────────────────────────────────────────────────────────

    async def get_semantic_profile(self, user_id: int) -> dict | None:
        """Fetch the semantic profile for a user."""
        cursor = await self.db.execute(
            "SELECT profile_data FROM semantic_profiles WHERE user_id = ?",
            (user_id,),
        )
        row = await cursor.fetchone()
        if row:
            try:
                return json.loads(row[0])
            except (json.JSONDecodeError, TypeError):
                return None
        return None

    async def save_semantic_profile(self, user_id: int, profile: dict) -> None:
        """Save or update the semantic profile for a user."""
        now = datetime.now().isoformat()
        profile_json = json.dumps(profile, ensure_ascii=False)
        await self.db.execute(
            """
            INSERT INTO semantic_profiles (user_id, profile_data, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                profile_data = excluded.profile_data,
                updated_at = excluded.updated_at
            """,
            (user_id, profile_json, now),
        )
        await self.db.commit()

    # ── Summaries ────────────────────────────────────────────────────────────────

    async def save_summary(
        self,
        user_id: int,
        summary_type: str,
        period_start: str,
        period_end: str,
        content: str,
        emotional_trends: dict | None = None,
        key_events: list | None = None,
    ) -> None:
        """Save a summary (daily, weekly, or monthly)."""
        now = datetime.now().isoformat()
        await self.db.execute(
            """
            INSERT INTO summaries
                (user_id, summary_type, period_start, period_end,
                 content, emotional_trends, key_events, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id, summary_type, period_start, period_end, content,
                json.dumps(emotional_trends) if emotional_trends else None,
                json.dumps(key_events) if key_events else None,
                now,
            ),
        )
        await self.db.commit()

    async def get_recent_summaries(
        self, user_id: int, summary_type: str = "daily", limit: int = 3
    ) -> list[dict]:
        """Fetch recent summaries of a given type."""
        cursor = await self.db.execute(
            """
            SELECT * FROM summaries
            WHERE user_id = ? AND summary_type = ?
            ORDER BY period_end DESC
            LIMIT ?
            """,
            (user_id, summary_type, limit),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_all_summaries(self, user_id: int) -> list[dict]:
        """Fetch all summaries for a user (for export)."""
        cursor = await self.db.execute(
            "SELECT * FROM summaries WHERE user_id = ? ORDER BY period_start ASC",
            (user_id,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # ── Schedule Tracking ────────────────────────────────────────────────────────

    async def get_schedule(self, user_id: int) -> dict | None:
        """Fetch schedule data for a user."""
        cursor = await self.db.execute(
            "SELECT * FROM schedules WHERE user_id = ?", (user_id,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def update_schedule(self, user_id: int, **kwargs) -> None:
        """Update schedule fields (streak_count, last_reminder_sent, etc.)."""
        now = datetime.now().isoformat()
        # Ensure schedule record exists
        await self.db.execute(
            """
            INSERT INTO schedules (user_id) VALUES (?)
            ON CONFLICT(user_id) DO NOTHING
            """,
            (user_id,),
        )
        if kwargs:
            allowed = {
                "next_reminder", "last_reminder_sent", "streak_count",
                "longest_streak", "last_diary_date",
            }
            fields = {k: v for k, v in kwargs.items() if k in allowed}
            if fields:
                set_clause = ", ".join(f"{k} = ?" for k in fields)
                values = list(fields.values()) + [user_id]
                await self.db.execute(
                    f"UPDATE schedules SET {set_clause} WHERE user_id = ?", values
                )
        await self.db.commit()

    # ── Diary Entries ─────────────────────────────────────────────────────────────

    async def save_diary_entry(
        self,
        user_id: int,
        raw_text: str,
        title: str | None = None,
        detected_emotions: str | None = None,
        emotion_confidence: float | None = None,
        extracted_goals: list | None = None,
        extracted_stressors: list | None = None,
        extracted_relationships: list | None = None,
        extracted_topics: list | None = None,
        personality_signals: list | None = None,
        behavioral_patterns: list | None = None,
        ai_summary: str | None = None,
        ai_followup: str | None = None,
        importance_score: float = 0.5,
    ) -> int:
        """Save a diary entry. Returns the entry ID."""
        now = datetime.now().isoformat()
        cursor = await self.db.execute(
            """
            INSERT INTO diary_entries
                (user_id, title, raw_text, detected_emotions, emotion_confidence,
                 extracted_goals, extracted_stressors, extracted_relationships,
                 extracted_topics, personality_signals, behavioral_patterns,
                 ai_summary, ai_followup, importance_score, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id, title, raw_text, detected_emotions, emotion_confidence,
                json.dumps(extracted_goals) if extracted_goals else None,
                json.dumps(extracted_stressors) if extracted_stressors else None,
                json.dumps(extracted_relationships) if extracted_relationships else None,
                json.dumps(extracted_topics) if extracted_topics else None,
                json.dumps(personality_signals) if personality_signals else None,
                json.dumps(behavioral_patterns) if behavioral_patterns else None,
                ai_summary, ai_followup, importance_score, now, now,
            ),
        )
        await self.db.commit()
        return cursor.lastrowid

    async def update_diary_entry(self, entry_id: int, **kwargs) -> None:
        """Update specific fields of a diary entry after analysis."""
        json_fields = {
            "extracted_goals", "extracted_stressors", "extracted_relationships",
            "extracted_topics", "personality_signals", "behavioral_patterns",
        }
        fields = {}
        for k, v in kwargs.items():
            if k in json_fields and isinstance(v, (list, dict)):
                fields[k] = json.dumps(v)
            else:
                fields[k] = v
        if not fields:
            return
        fields["updated_at"] = datetime.now().isoformat()
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [entry_id]
        await self.db.execute(
            f"UPDATE diary_entries SET {set_clause} WHERE id = ?", values
        )
        await self.db.commit()

    async def get_latest_diary_entry(self, user_id: int) -> dict | None:
        """Fetch the most recent diary entry for a user."""
        cursor = await self.db.execute(
            "SELECT * FROM diary_entries WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
            (user_id,),
        )
        row = await cursor.fetchone()
        return self._parse_diary_row(row) if row else None

    async def get_recent_diary_entries(
        self, user_id: int, limit: int = 5
    ) -> list[dict]:
        """Fetch the N most recent diary entries."""
        cursor = await self.db.execute(
            "SELECT * FROM diary_entries WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        )
        rows = await cursor.fetchall()
        return [self._parse_diary_row(r) for r in reversed(rows)]

    async def get_important_diary_entries(
        self, user_id: int, limit: int = 5
    ) -> list[dict]:
        """Fetch the highest-importance diary entries."""
        cursor = await self.db.execute(
            """
            SELECT * FROM diary_entries
            WHERE user_id = ?
            ORDER BY importance_score DESC, created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
        rows = await cursor.fetchall()
        return [self._parse_diary_row(r) for r in rows]

    async def search_diary_entries(
        self, user_id: int, query: str, limit: int = 10
    ) -> list[dict]:
        """Full-text search across diary entries."""
        pattern = f"%{query}%"
        cursor = await self.db.execute(
            """
            SELECT * FROM diary_entries
            WHERE user_id = ?
              AND (raw_text LIKE ? OR ai_summary LIKE ? OR extracted_topics LIKE ?
                   OR extracted_goals LIKE ? OR extracted_stressors LIKE ? OR title LIKE ?)
            ORDER BY importance_score DESC, created_at DESC
            LIMIT ?
            """,
            (user_id, pattern, pattern, pattern, pattern, pattern, pattern, limit),
        )
        rows = await cursor.fetchall()
        return [self._parse_diary_row(r) for r in rows]

    async def get_diary_entries_by_emotion(
        self, user_id: int, emotion: str, limit: int = 5
    ) -> list[dict]:
        """Fetch diary entries matching a specific emotion."""
        pattern = f"%{emotion}%"
        cursor = await self.db.execute(
            """
            SELECT * FROM diary_entries
            WHERE user_id = ? AND detected_emotions LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, pattern, limit),
        )
        rows = await cursor.fetchall()
        return [self._parse_diary_row(r) for r in rows]

    async def get_diary_entry_count(self, user_id: int) -> int:
        """Total number of diary entries for a user."""
        cursor = await self.db.execute(
            "SELECT COUNT(*) FROM diary_entries WHERE user_id = ?", (user_id,)
        )
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def get_all_diary_entries(self, user_id: int) -> list[dict]:
        """Fetch ALL diary entries for export."""
        cursor = await self.db.execute(
            "SELECT * FROM diary_entries WHERE user_id = ? ORDER BY created_at ASC",
            (user_id,),
        )
        rows = await cursor.fetchall()
        return [self._parse_diary_row(r) for r in rows]

    async def get_diary_emotion_timeline(
        self, user_id: int, limit: int = 30
    ) -> list[dict]:
        """Get emotion timeline from diary entries (date + emotions + importance)."""
        cursor = await self.db.execute(
            """
            SELECT id, created_at, detected_emotions, emotion_confidence,
                   importance_score, ai_summary
            FROM diary_entries
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in reversed(rows)]

    async def get_diary_timeline_events(
        self, user_id: int, min_importance: float = 0.6, limit: int = 20
    ) -> list[dict]:
        """Fetch high-importance diary entries for life timeline."""
        cursor = await self.db.execute(
            """
            SELECT id, created_at, title, ai_summary, detected_emotions,
                   extracted_topics, importance_score
            FROM diary_entries
            WHERE user_id = ? AND importance_score >= ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, min_importance, limit),
        )
        rows = await cursor.fetchall()
        entries = []
        for r in reversed(rows):
            d = dict(r)
            for field in ("extracted_topics",):
                if d.get(field):
                    try:
                        d[field] = json.loads(d[field])
                    except (json.JSONDecodeError, TypeError):
                        d[field] = []
            entries.append(d)
        return entries

    def _parse_diary_row(self, row) -> dict:
        """Parse a diary_entries row, deserializing JSON fields."""
        d = dict(row)
        json_fields = [
            "extracted_goals", "extracted_stressors", "extracted_relationships",
            "extracted_topics", "personality_signals", "behavioral_patterns",
        ]
        for field in json_fields:
            if d.get(field):
                try:
                    d[field] = json.loads(d[field])
                except (json.JSONDecodeError, TypeError):
                    d[field] = []
        return d

    # ── Backup ───────────────────────────────────────────────────────────────────

    async def create_backup(self) -> Path | None:
        """Create a timestamped backup of the database file."""
        if not DB_PATH.exists():
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = BACKUP_DIR / f"diary_backup_{timestamp}.db"

        try:
            # Checkpoint WAL before backup
            await self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            # Copy the database file
            await asyncio.to_thread(shutil.copy2, str(DB_PATH), str(backup_path))
            logger.info("Database backup created: %s", backup_path)

            # Clean up old backups (keep last 10)
            backups = sorted(BACKUP_DIR.glob("diary_backup_*.db"))
            if len(backups) > 10:
                for old in backups[:-10]:
                    old.unlink()
                    logger.debug("Removed old backup: %s", old)

            return backup_path
        except Exception as exc:
            logger.error("Backup failed: %s", exc)
            return None


# ── Module-level convenience ─────────────────────────────────────────────────────

def get_db() -> DatabaseManager:
    """Get the singleton DatabaseManager instance."""
    return DatabaseManager()
