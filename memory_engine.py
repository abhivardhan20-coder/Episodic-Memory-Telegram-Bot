import json
import os
from datetime import datetime
from pathlib import Path

MEMORY_DIR = Path("memory")
MEMORY_DIR.mkdir(exist_ok=True)

MAX_EPISODES = 50       # Maximum episodes to keep per user
CONTEXT_EPISODES = 8    # How many recent episodes to inject into each prompt


def get_memory_path(user_id: int) -> Path:
    """Returns the file path for a given user's memory."""
    return MEMORY_DIR / f"{user_id}.json"


def load_episodes(user_id: int) -> list[dict]:
    """Load all stored episodes for a user. Returns empty list if none exist."""
    path = get_memory_path(user_id)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_episode(user_id: int, user_message: str, bot_response: str) -> None:
    """Save a new episode to the user's memory file."""
    episodes = load_episodes(user_id)

    new_episode = {
        "timestamp": datetime.now().isoformat(),
        "user_message": user_message,
        "bot_response": bot_response,
    }

    episodes.append(new_episode)

    # Trim oldest episodes if we exceed the limit
    if len(episodes) > MAX_EPISODES:
        episodes = episodes[-MAX_EPISODES:]

    path = get_memory_path(user_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(episodes, f, indent=2, ensure_ascii=False)


def clear_memory(user_id: int) -> None:
    """Delete all stored episodes for a user."""
    path = get_memory_path(user_id)
    if path.exists():
        path.unlink()


def build_context_prompt(user_id: int) -> str:
    """
    Build a context string from recent episodes to inject into the system prompt.
    Returns empty string if no history exists.
    """
    episodes = load_episodes(user_id)

    if not episodes:
        return ""

    # Take only the most recent N episodes
    recent = episodes[-CONTEXT_EPISODES:]

    lines = ["Here is your conversation history with this user (most recent last):"]
    lines.append("")

    for ep in recent:
        ts = ep["timestamp"][:16].replace("T", " ")  # Format: 2026-04-14 09:32
        lines.append(f"[{ts}]")
        lines.append(f"User: {ep['user_message']}")
        lines.append(f"You: {ep['bot_response']}")
        lines.append("")

    lines.append("Use this history to give a personalized, context-aware response.")
    return "\n".join(lines)


def get_memory_summary(user_id: int) -> str:
    """Returns a human-readable summary of how much memory a user has."""
    episodes = load_episodes(user_id)
    if not episodes:
        return "No memory stored yet."
    oldest = episodes[0]["timestamp"][:10]
    return f"{len(episodes)} episodes stored. Oldest: {oldest}."
