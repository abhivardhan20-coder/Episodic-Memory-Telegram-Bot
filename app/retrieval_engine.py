"""
Hybrid context retrieval engine for the AI Diary Companion.
"""

import logging
from app.config import RECENT_EPISODES_COUNT, MAX_CONTEXT_CHARS
from app.database import get_db
from app.memory_engine import get_recent_episodes, get_emotional_trends
from app.semantic_engine import get_profile, profile_to_context_string

logger = logging.getLogger(__name__)

async def build_context(user_id: int, current_message: str) -> str:
    sections = []
    
    # 1. Profile
    profile = await get_profile(user_id)
    p_str = profile_to_context_string(profile)
    if p_str: sections.append(p_str)
    
    # 2. Recent Episodes
    recent = await get_recent_episodes(user_id, limit=RECENT_EPISODES_COUNT)
    if recent:
        lines = ["=== RECENT CONVERSATIONS ==="]
        for ep in recent:
            lines.append(f"User: {ep['user_message']}\nYou: {ep['bot_response'][:200]}")
        sections.append("\n".join(lines) + "\n=== END RECENT ===")
        
    # 3. Trends
    trends = await get_emotional_trends(user_id, days=14)
    if trends.get("total_entries", 0) > 0:
        sections.append(f"=== EMOTIONAL TREND ===\n{trends['trend_summary']}")

    full_context = "\n\n".join(sections)
    return full_context[:MAX_CONTEXT_CHARS]
