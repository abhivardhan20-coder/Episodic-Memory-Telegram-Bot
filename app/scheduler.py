"""
Daily diary check-in scheduler for the AI Diary Companion.
"""

import logging
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.config import DEFAULT_TIMEZONE, DEFAULT_REMINDER_TIME, BACKUP_INTERVAL_HOURS
from app.database import get_db
from app.prompts import get_contextual_diary_prompt
from app.semantic_engine import get_profile
from app.utils import check_and_generate_summaries

logger = logging.getLogger(__name__)

_scheduler: AsyncIOScheduler | None = None

def get_scheduler() -> AsyncIOScheduler:
    global _scheduler
    if _scheduler is None: _scheduler = AsyncIOScheduler()
    return _scheduler

async def setup_scheduler(app) -> None:
    scheduler = get_scheduler()
    
    # Backup job
    scheduler.add_job(_periodic_backup, CronTrigger(hour=f"*/{BACKUP_INTERVAL_HOURS}"), id="backup")
    
    # Restore user jobs
    db = get_db()
    users = await db.get_all_users_with_reminders()
    for u in users:
        _add_reminder_job(scheduler, app, u["user_id"], u.get("reminder_time", DEFAULT_REMINDER_TIME), u.get("timezone", DEFAULT_TIMEZONE))
    
    scheduler.start()
    logger.info("Scheduler started with %d users", len(users))

async def shutdown_scheduler():
    s = get_scheduler()
    if s.running: s.shutdown()

async def schedule_user_reminder(app, user_id: int, time_str: str, tz: str = DEFAULT_TIMEZONE):
    s = get_scheduler()
    jid = f"reminder_{user_id}"
    if s.get_job(jid): s.remove_job(jid)
    _add_reminder_job(s, app, user_id, time_str, tz)
    await get_db().update_user_settings(user_id, reminder_time=time_str, timezone=tz, reminder_enabled=1)

def _add_reminder_job(s, app, user_id, time_str, tz):
    h, m = time_str.split(":")
    s.add_job(_send_prompt, CronTrigger(hour=int(h), minute=int(m), timezone=tz), id=f"reminder_{user_id}", args=[app, user_id])

async def _send_prompt(app, user_id):
    try:
        p = await get_profile(user_id)
        prompt = get_contextual_diary_prompt(goals=p.get("goals"), stressors=p.get("stressors"))
        await app.bot.send_message(chat_id=user_id, text=prompt)
        await get_db().update_schedule(user_id, last_reminder_sent=datetime.now().isoformat())
        await check_and_generate_summaries(user_id)
    except Exception as e:
        logger.error("Failed to send prompt to %d: %s", user_id, e)

async def _periodic_backup():
    await get_db().create_backup()
