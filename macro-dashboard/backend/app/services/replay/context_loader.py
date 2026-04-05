"""
Replay context loader — serves curated historical context notes for a given date.

Notes are stored in app/data/replay_context.json as date-ranged entries.
This is NOT a generic news feed — each note is hand-written to explain the
macro narrative of a period in speaker-faithful language.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

from app.schemas.replay import ContextNote

logger = logging.getLogger(__name__)

_DATA_FILE = Path(__file__).parent.parent.parent / "data" / "replay_context.json"


def _load_raw() -> list[dict]:
    """Load and parse the raw notes list from disk."""
    try:
        with open(_DATA_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("notes", [])
    except FileNotFoundError:
        logger.warning("replay_context.json not found at %s", _DATA_FILE)
        return []
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse replay_context.json: %s", exc)
        return []


def load_context_for_date(as_of_date: date) -> list[ContextNote]:
    """
    Return all curated context notes whose date range covers as_of_date.

    A note is included if as_of_date falls within [date_start, date_end] inclusive.
    """
    raw_notes = _load_raw()
    matching: list[ContextNote] = []

    for entry in raw_notes:
        try:
            date_start = date.fromisoformat(entry["date_start"])
            date_end = date.fromisoformat(entry["date_end"])
        except (KeyError, ValueError) as exc:
            logger.warning("Skipping malformed context note entry: %s", exc)
            continue

        if date_start <= as_of_date <= date_end:
            matching.append(ContextNote(
                title=entry.get("title", ""),
                body=entry.get("body", ""),
                tags=entry.get("tags", []),
            ))

    return matching
