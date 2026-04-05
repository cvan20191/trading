"""
Catalyst config loader.

Load order:
  1. backend/app/data/catalysts.json    (local working copy, may be gitignored)
  2. backend/app/data/catalysts.example.json  (tracked default template)

Returns a raw dict. The engine validates and maps it to typed models.
Falls back silently if neither file exists.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent.parent / "data"
_LIVE_FILE = _DATA_DIR / "catalysts.json"
_EXAMPLE_FILE = _DATA_DIR / "catalysts.example.json"

# Module-level cache — cleared on reload
_cached_config: dict[str, Any] | None = None

# Config source metadata — updated on every load
_config_meta: dict[str, Any] = {
    "loaded_from": "not yet loaded",
    "last_loaded_at": "",
    "using_example_fallback": False,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_file(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning("Catalyst config at %s is not a JSON object — skipping", path)
            return None
        return data
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        logger.warning("Catalyst config parse error at %s: %s", path, exc)
        return None
    except Exception as exc:
        logger.warning("Unexpected error loading catalyst config at %s: %s", path, exc)
        return None


def load_catalyst_config(force_reload: bool = False) -> dict[str, Any]:
    """
    Return the catalyst config dict.

    Uses module-level cache unless `force_reload=True`.
    Side-effects: updates `_config_meta` with provenance metadata.
    """
    global _cached_config, _config_meta

    if _cached_config is not None and not force_reload:
        return _cached_config

    config = _load_file(_LIVE_FILE)
    if config is not None:
        logger.info("Loaded catalyst config from %s", _LIVE_FILE)
        _config_meta = {
            "loaded_from": str(_LIVE_FILE),
            "last_loaded_at": _now_iso(),
            "using_example_fallback": False,
        }
    else:
        config = _load_file(_EXAMPLE_FILE)
        if config is not None:
            logger.info("Loaded catalyst config from example file %s", _EXAMPLE_FILE)
            _config_meta = {
                "loaded_from": str(_EXAMPLE_FILE),
                "last_loaded_at": _now_iso(),
                "using_example_fallback": True,
            }
        else:
            logger.warning("No catalyst config file found — using empty defaults")
            config = {}
            _config_meta = {
                "loaded_from": "empty defaults",
                "last_loaded_at": _now_iso(),
                "using_example_fallback": True,
            }

    _cached_config = config
    return config


def get_config_meta() -> dict[str, Any]:
    """Return metadata about the currently loaded config (source, timestamp, fallback status)."""
    return dict(_config_meta)


def reload_catalyst_config() -> dict[str, Any]:
    """Force-reload config from disk and return new config."""
    return load_catalyst_config(force_reload=True)
