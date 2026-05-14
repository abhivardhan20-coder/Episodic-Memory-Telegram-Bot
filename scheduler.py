"""
Daily diary check-in scheduler for the AI Diary Companion.

Uses APScheduler to manage per-user daily reminder jobs.
Jobs survive restarts by reloading from the database on startup.
"""

import logging
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from config import DEFAULT_TIMEZONE, DEFAULT_REMINDER_TIME, BACKUP_INTERVAL_HOURS
from database import get_db
from prompts import get_contextual_diary_prompt
from semantic_profile import get_profile
from summarizer import check_and_generate_summaries

logger = logging.getLogger(__name__)

# Module-level scheduler instance
_scheduler: AsyncIOScheduler | None = None


def get_scheduler() -> AsyncIOScheduler:
    """Get the singleton scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = AsyncIOScheduler()
    return _scheduler


async def setup_scheduler(app) -> None:
    """
    Initialize the scheduler and restore all user reminder jobs.

    Args:
        app: The python-telegram-bot Application instance.
    """
    scheduler = get_scheduler()

    # Add periodic backup job
    scheduler.add_job(
        _periodic_backup,
        CronTrigger(hour=f"*/{BACKUP_INTERVAL_HOURS}"),
        id="periodic_backup",
        replace_existing=True,
    )

    # Restore user reminder jobs from database
    db = get_db()
    users = await db.get_all_users_with_reminders()

    for user in users:
        user_id = user["user_id"]
        reminder_time = user.get("reminder_time", DEFAULT_REMINDER_TIME)
        timezone = user.get("timezone", DEFAULT_TIMEZONE)

        _add_reminder_job(scheduler, app, user_id, reminder_time, timezone)

    scheduler.start()
    logger.info(
        "Scheduler started with %d user reminder(s) and periodic backup",
        len(users),
    )


async def shutdown_scheduler() -> None:
    """Gracefully shut down the scheduler."""
    scheduler = get_scheduler()
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler shut down")


async def schedule_user_reminder(
    app, user_id: int, time_str: str, timezone: str = DEFAULT_TIMEZONE
) -> None:
    """
    Add or update a user's daily reminder job.

    Args:
        app: The python-telegram-bot Application instance.
        user_id: Telegram user ID.
        time_str: Time in HH:MM format.
        timezone: Timezone string (e.g., 'Asia/Kolkata').
    """
    scheduler = get_scheduler()

    # Remove existing job if any
    job_id = f"diary_reminder_{user_id}"
    if scheduler.get_job(job_id):
        scheduler.remove_job(job_id)

    _add_reminder_job(scheduler, app, user_id, time_str, timezone)

    # Update database
    db = get_db()
    await db.update_user_settings(
        user_id, reminder_time=time_str, timezone=timezone, reminder_enabled=1
    )

    logger.info("Scheduled reminder for user %d at %s (%s)", user_id, time_str, timezone)


async def remove_user_reminder(user_id: int) -> None:
    """Remove a user's daily reminder job."""
    scheduler = get_scheduler()
    job_id = f"diary_reminder_{user_id}"
    if scheduler.get_job(job_id):
        scheduler.remove_job(job_id)

    db = get_db()
    await db.update_user_settings(user_id, reminder_enabled=0)
    logger.info("Removed reminder for user %d", user_id)


def _add_reminder_job(
    scheduler: AsyncIOScheduler,
    app,
    user_id: int,
    time_str: str,
    timezone: str,
) -> None:
    """Internal: add a cron job for the daily diary reminder."""
    try:
        hour, minute = time_str.split(":")
        job_id = f"diary_reminder_{user_id}"

        scheduler.add_job(
            _send_diary_prompt,
            CronTrigger(hour=int(hour), minute=int(minute), timezone=timezone),
            id=job_id,
            replace_existing=True,
            args=[app, user_id],
        )
        logger.debug("Added reminder job %s at %s (%s)", job_id, time_str, timezone)
    except Exception as exc:
        logger.error("Failed to add reminder job for user %d: %s", user_id, exc)


async def _send_diary_prompt(app, user_id: int) -> None:
    """
    Send the daily diary prompt to a user.
    Uses context from their profile to personalise the prompt.
    """
    try:
        # Get user's profile for contextual prompts
        profile = await get_profile(user_id)

        # Build a contextual prompt
        prompt = get_contextual_diary_prompt(
            topics=None,  # Could extract from recent episodes
            emotions=profile.get("recurring_emotions"),
            goals=profile.get("goals"),
            stressors=profile.get("stressors"),
            events=profile.get("important_events"),
        )

        # Send the message
        await app.bot.send_message(chat_id=user_id, text=prompt)

        # Update schedule tracking
        db = get_db()
        now = datetime.now()

        schedule = await db.get_schedule(user_id)
        last_diary = schedule.get("last_diary_date") if schedule else None

        # Update streak
        if last_diary:
            last_date = datetime.strptime(last_diary, "%Y-%m-%d").date()
            today = now.date()
            if (today - last_date).days == 1:
                new_streak = (schedule.get("streak_count", 0) if schedule else 0) + 1
            elif (today - last_date).days == 0:
                new_streak = schedule.get("streak_count", 0) if schedule else 0
            else:
                new_streak = 1  # streak broken
        else:
            new_streak = 1

        longest = max(
            new_streak,
            schedule.get("longest_streak", 0) if schedule else 0,
        )

        await db.update_schedule(
            user_id,
            last_reminder_sent=now.isoformat(),
            streak_count=new_streak,
            longest_streak=longest,
            last_diary_date=now.strftime("%Y-%m-%d"),
        )

        # Trigger summary generation check
        try:
            await check_and_generate_summaries(user_id)
        except Exception as exc:
            logger.warning("Summary generation check failed for user %d: %s", user_id, exc)

        logger.info("Sent diary prompt to user %d (streak: %d)", user_id, new_streak)

    except Exception as exc:
        logger.error("Failed to send diary prompt to user %d: %s", user_id, exc)


async def _periodic_backup() -> None:
    """Run periodic database backup."""
    try:
        db = get_db()
        result = await db.create_backup()
        if result:
            logger.info("Periodic backup completed: %s", result)
    except Exception as exc:
        logger.error("Periodic backup failed: %s", exc)
