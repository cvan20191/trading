"""
In-memory cache with TTL and optional JSON snapshot persistence.

Avoids hammering provider APIs on every request.
Stale cached values are served with explicit status flags when
provider calls fail and live_allow_stale_cache is True.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CACHE_FILE = Path(__file__).parent.parent.parent.parent / ".cache" / "live_snapshot.json"


class TTLCache:
    """Simple single-slot TTL cache for the live snapshot."""

    def __init__(self, ttl_seconds: int = 900) -> None:
        self._ttl = ttl_seconds
        self._data: Any = None
        self._stored_at: float = 0.0

    def get(self) -> Any | None:
        if self._data is None:
            return None
        if time.monotonic() - self._stored_at > self._ttl:
            return None    # expired
        return self._data

    def set(self, value: Any) -> None:
        self._data = value
        self._stored_at = time.monotonic()

    def clear(self) -> None:
        self._data = None
        self._stored_at = 0.0

    def is_populated(self) -> bool:
        return self._data is not None

    def save_to_disk(self, payload: dict) -> None:
        """Optional: persist the last good snapshot as JSON for crash recovery."""
        try:
            _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            _CACHE_FILE.write_text(json.dumps(payload, default=str), encoding="utf-8")
        except Exception as exc:
            logger.warning("Could not persist cache to disk: %s", exc)

    def load_from_disk(self) -> dict | None:
        """Return persisted snapshot or None if missing/unreadable."""
        try:
            if _CACHE_FILE.exists():
                return json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not read disk cache: %s", exc)
        return None


# Module-level singleton
_snapshot_cache = TTLCache()


def get_snapshot_cache() -> TTLCache:
    return _snapshot_cache
