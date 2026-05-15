"""
Semantic profile manager for the AI Diary Companion.
Maintains a structured long-term profile of each user.
"""

import json
import logging
from copy import deepcopy

from app.database import get_db
from app.utils import get_llm
from app.prompts import PROFILE_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)

DEFAULT_PROFILE: dict = {
    "name": None, "goals": [], "stressors": [], "preferences": [],
    "relationships": [], "recurring_emotions": [], "important_events": [],
    "habits": [], "routines": [], "personality_traits": [], "fears": [],
    "aspirations": [], "strengths": [],
}

async def get_profile(user_id: int) -> dict:
    db = get_db()
    profile = await db.get_semantic_profile(user_id)
    if profile is None: return deepcopy(DEFAULT_PROFILE)
    merged = deepcopy(DEFAULT_PROFILE)
    merged.update(profile)
    return merged

async def update_profile_from_conversation(user_id: int, user_message: str, bot_response: str):
    current = await get_profile(user_id)
    prompt = PROFILE_EXTRACTION_PROMPT.format(
        user_message=user_message, bot_response=bot_response,
        current_profile=json.dumps(current, indent=2)
    )
    updates = await get_llm().extract_profile(prompt)
    if not updates or updates.get("no_update"): return
    
    updated = _merge_profile(current, updates)
    await get_db().save_semantic_profile(user_id, updated)

def _merge_profile(current: dict, updates: dict) -> dict:
    merged = deepcopy(current)
    for k, v in updates.items():
        if k == "no_update" or k not in merged or v is None: continue
        if isinstance(merged[k], list) and isinstance(v, list):
            existing = {str(i).lower() for i in merged[k]}
            for item in v:
                if str(item).lower() not in existing: merged[k].append(item)
        else:
            merged[k] = v
    return merged

def profile_to_context_string(profile: dict) -> str:
    lines = ["=== LONG-TERM PROFILE ==="]
    if profile.get("name"): lines.append(f"Name: {profile['name']}")
    for k in ["goals", "stressors", "preferences", "relationships"]:
        if profile.get(k): lines.append(f"{k.title()}: {', '.join(profile[k])}")
    return "\n".join(lines) + "\n=== END PROFILE ===" if len(lines) > 1 else ""
