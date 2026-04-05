"""
Live data endpoints — fetch real macro data and serve computed playbook.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from app.schemas.dashboard_state import DashboardState
from app.schemas.live_snapshot_response import LivePlaybookResponse, LiveSnapshotResponse
from app.services.ingestion.cache import get_snapshot_cache
from app.services.ingestion.live_snapshot_service import get_live_playbook, get_live_snapshot
from app.services.rules.dashboard_state_builder import build_dashboard_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/live", tags=["live"])


@router.get("/snapshot", response_model=LiveSnapshotResponse)
async def live_snapshot(
    force_refresh: bool = Query(default=False, description="Bypass cache and re-fetch all providers"),
) -> LiveSnapshotResponse:
    """
    Fetch the current live IndicatorSnapshot from FRED, Yahoo, and CPI providers.
    Returns normalized snapshot with full source provenance and freshness status.
    """
    try:
        return await get_live_snapshot(force_refresh=force_refresh)
    except Exception as exc:
        logger.error("live_snapshot failed: %s", exc)
        raise HTTPException(status_code=503, detail=f"Live data unavailable: {exc}") from exc


@router.get("/dashboard-state", response_model=DashboardState)
async def live_dashboard_state(
    force_refresh: bool = Query(default=False),
) -> DashboardState:
    """
    Return the deterministic DashboardState computed from the live IndicatorSnapshot.
    """
    try:
        snap_resp = await get_live_snapshot(force_refresh=force_refresh)
        return build_dashboard_state(snap_resp.snapshot)
    except Exception as exc:
        logger.error("live_dashboard_state failed: %s", exc)
        raise HTTPException(status_code=503, detail=f"Live data unavailable: {exc}") from exc


@router.get("/playbook", response_model=LivePlaybookResponse)
async def live_playbook(
    force_refresh: bool = Query(default=False),
    pmi_manufacturing: float | None = Query(default=None, description="Optional manual PMI manufacturing override"),
    pmi_services: float | None = Query(default=None, description="Optional manual PMI services override"),
) -> LivePlaybookResponse:
    """
    Full live pipeline: providers → IndicatorSnapshot → rule engine → LLM summary.
    Returns LivePlaybookResponse with snapshot, state, summary, and source metadata.
    """
    try:
        return await get_live_playbook(
            force_refresh=force_refresh,
            pmi_manufacturing_override=pmi_manufacturing,
            pmi_services_override=pmi_services,
        )
    except Exception as exc:
        logger.error("live_playbook failed: %s", exc)
        raise HTTPException(status_code=503, detail=f"Live playbook unavailable: {exc}") from exc


@router.post("/refresh", tags=["live"])
async def manual_refresh() -> dict[str, str]:
    """
    Bypass cache and trigger a full provider re-fetch.
    Returns a status summary of the refresh.
    """
    try:
        cache = get_snapshot_cache()
        cache.clear()
        resp = await get_live_snapshot(force_refresh=True)
        return {
            "status": "refreshed",
            "overall_status": resp.overall_status,
            "generated_at": resp.generated_at,
            "stale_count": str(len(resp.stale_series)),
        }
    except Exception as exc:
        logger.error("manual_refresh failed: %s", exc)
        raise HTTPException(status_code=503, detail=f"Refresh failed: {exc}") from exc
