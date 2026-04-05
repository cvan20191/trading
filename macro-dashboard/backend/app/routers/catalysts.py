"""
Catalyst endpoints — inspect catalyst config and trigger reload.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from app.schemas.catalysts import CatalystState  # noqa: F401
from app.services.catalysts.config_loader import (
    get_config_meta,
    load_catalyst_config,
    reload_catalyst_config,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/catalysts", tags=["catalysts"])


@router.get("", response_model=dict)
async def get_catalyst_config() -> dict[str, Any]:
    """
    Return the current validated catalyst config as loaded from disk.
    Useful for inspecting what catalyst values the engine is using.
    """
    try:
        config = load_catalyst_config()
        meta = get_config_meta()
        return {
            "status": "ok",
            "config": config,
            **meta,
        }
    except Exception as exc:
        logger.error("get_catalyst_config failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to load catalyst config: {exc}") from exc


@router.post("/reload", response_model=dict)
async def reload_catalysts() -> dict[str, Any]:
    """
    Force-reload catalyst config from disk without restarting the server.
    Returns confirmation, item counts, and updated config metadata.
    """
    try:
        config = reload_catalyst_config()
        meta = get_config_meta()
        return {
            "status": "reloaded",
            "mega_ipos_count": len(config.get("mega_ipos", [])),
            "fed_chair_count": len(config.get("fed_chair", [])),
            **meta,
        }
    except Exception as exc:
        logger.error("reload_catalysts failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Reload failed: {exc}") from exc
