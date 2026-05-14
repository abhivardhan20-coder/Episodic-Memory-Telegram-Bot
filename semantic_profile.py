"""
Semantic profile manager for the AI Diary Companion.

Maintains a structured long-term profile of each user that grows
over time — goals, stressors, relationships, habits, personality, etc.
The profile is updated after each conversation via LLM extraction.
"""

import json
import logging
from copy import deepcopy

from database import get_db
from llm_client import get_llm
from prompts import PROFILE_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)

# ── Default Profile Structure ────────────────────────────────────────────────────

DEFAULT_PROFILE: dict = {
    "name": None,
    "goals": [],
    "stressors": [],
    "preferences": [],
    "relationships": [],
    "recurring_emotions": [],
    "important_events": [],
    "habits": [],
    "routines": [],
    "personality_traits": [],
    "fears": [],
    "aspirations": [],
    "strengths": [],
}


async def get_profile(user_id: int) -> dict:
    """
    Load the semantic profile for a user.
    Returns the default profile structure if none exists.
    """
    db = get_db()
    profile = await db.get_semantic_profile(user_id)
    if profile is None:
        return deepcopy(DEFAULT_PROFILE)
    # Ensure all keys exist (forward-compat if we add new fields)
    merged = deepcopy(DEFAULT_PROFILE)
    merged.update(profile)
    return merged


async def save_profile(user_id: int, profile: dict) -> None:
    """Save the semantic profile for a user."""
    db = get_db()
    await db.save_semantic_profile(user_id, profile)
    logger.debug("Saved semantic profile for user %d", user_id)


async def update_profile_from_conversation(
    user_id: int, user_message: str, bot_response: str
) -> dict | None:
    """
    After a conversation, ask the LLM to extract any new personal
    information and merge it into the existing profile.

    Returns the updated profile, or None if no updates were found.
    """
    current_profile = await get_profile(user_id)

    # Format the extraction prompt
    prompt = PROFILE_EXTRACTION_PROMPT.format(
        user_message=user_message,
        bot_response=bot_response,
        current_profile=json.dumps(current_profile, indent=2),
    )

    llm = get_llm()
    updates = await llm.extract_profile(prompt)

    if updates is None:
        logger.debug("Profile extraction returned None for user %d", user_id)
        return None

    if updates.get("no_update"):
        logger.debug("No profile updates for user %d", user_id)
        return None

    # Merge updates into the current profile
    updated = _merge_profile(current_profile, updates)
    await save_profile(user_id, updated)
    logger.info("Updated semantic profile for user %d", user_id)
    return updated


def _merge_profile(current: dict, updates: dict) -> dict:
    """
    Merge new profile updates into the existing profile.
    - Scalar fields (like 'name') are overwritten if non-null
    - List fields are extended with new unique items
    """
    merged = deepcopy(current)

    for key, value in updates.items():
        if key == "no_update":
            continue
        if key not in merged:
            continue
        if value is None:
            continue

        if isinstance(merged[key], list) and isinstance(value, list):
            # Add only new unique items (case-insensitive dedup)
            existing_lower = {
                item.lower() if isinstance(item, str) else str(item)
                for item in merged[key]
            }
            for item in value:
                item_lower = item.lower() if isinstance(item, str) else str(item)
                if item_lower not in existing_lower:
                    merged[key].append(item)
                    existing_lower.add(item_lower)
        elif isinstance(merged[key], list) and isinstance(value, str):
            # Single string being added to a list
            existing_lower = {
                item.lower() if isinstance(item, str) else str(item)
                for item in merged[key]
            }
            if value.lower() not in existing_lower:
                merged[key].append(value)
        else:
            # Scalar overwrite (e.g., name)
            merged[key] = value

    return merged


def profile_to_context_string(profile: dict) -> str:
    """
    Convert a semantic profile dict into a human-readable string
    suitable for injection into the LLM system prompt.
    """
    lines = ["=== LONG-TERM PROFILE OF THIS USER ==="]

    if profile.get("name"):
        lines.append(f"Name: {profile['name']}")

    list_fields = [
        ("goals", "Goals"),
        ("stressors", "Stressors"),
        ("preferences", "Preferences"),
        ("relationships", "Relationships"),
        ("recurring_emotions", "Recurring emotions"),
        ("important_events", "Important life events"),
        ("habits", "Habits"),
        ("routines", "Routines"),
        ("personality_traits", "Personality traits"),
        ("fears", "Fears"),
        ("aspirations", "Aspirations"),
        ("strengths", "Strengths"),
    ]

    for key, label in list_fields:
        items = profile.get(key, [])
        if items:
            items_str = ", ".join(str(i) for i in items)
            lines.append(f"{label}: {items_str}")

    if len(lines) == 1:
        return ""  # Empty profile, nothing to inject

    lines.append("=== END PROFILE ===")
    return "\n".join(lines)
