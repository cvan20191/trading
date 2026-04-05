"""
Replay Lab endpoints — historical playbook and outcome review.

GET /api/replay/playbook?as_of=YYYY-MM-DD
  Returns ReplayPlaybookResponse: historical snapshot + rule engine state +
  LLM summary + curated context notes. No catalysts (not date-aware).

GET /api/replay/outcomes?as_of=YYYY-MM-DD
  Returns OutcomeReview: SPY/QQQ/WTI forward returns at 1W, 1M, 3M.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from app.config import settings
from app.schemas.replay import OutcomeReview, ReplayPlaybookResponse
from app.services.replay.context_loader import load_context_for_date
from app.services.replay.outcomes import compute_outcomes
from app.services.replay.snapshot_builder import build_historical_snapshot
from app.services.rules.dashboard_state_builder import build_dashboard_state_with_conclusion
from app.services.summary_engine import generate_summary

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/replay", tags=["replay"])

_MIN_DATE = date(2015, 1, 1)


def _parse_and_validate_date(as_of: str) -> date:
    """Parse the as_of query param and enforce MVP date constraints."""
    try:
        parsed = date.fromisoformat(as_of)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date format '{as_of}'. Use YYYY-MM-DD.",
        )

    today = date.today()
    yesterday = today.replace(day=today.day - 1) if today.day > 1 else today  # safe approx
    # Use timedelta for correctness
    from datetime import timedelta
    yesterday = today - timedelta(days=1)

    if parsed >= today:
        raise HTTPException(
            status_code=400,
            detail=f"Replay date must be in the past. Received: {as_of}",
        )
    if parsed < _MIN_DATE:
        raise HTTPException(
            status_code=400,
            detail=f"Replay dates before {_MIN_DATE} are not supported. Received: {as_of}",
        )
    return parsed


@router.get("/playbook", response_model=ReplayPlaybookResponse)
async def replay_playbook(
    as_of: str = Query(..., description="Historical date to replay (YYYY-MM-DD)"),
) -> ReplayPlaybookResponse:
    """
    Build a complete macro playbook for a historical as_of date.

    - Historical FRED + Yahoo data is fetched up to as_of date.
    - The existing deterministic rule engine and LLM summary engine are reused.
    - No catalysts — build_catalyst_state() is not date-aware.
    - Curated context notes replace the catalyst forward-watch layer.
    - data_notes surface all data quality caveats (vintage, PMI proxy, no forward P/E).
    """
    as_of_date = _parse_and_validate_date(as_of)

    try:
        snapshot, sources, data_notes = await build_historical_snapshot(
            as_of_date=as_of_date,
            fred_api_key=settings.fred_api_key,
            http_timeout=settings.http_timeout_seconds,
        )
    except Exception as exc:
        logger.error("build_historical_snapshot failed for %s: %s", as_of, exc)
        raise HTTPException(
            status_code=503,
            detail=f"Failed to build historical snapshot for {as_of}: {exc}",
        ) from exc

    state, playbook_conclusion = build_dashboard_state_with_conclusion(snapshot)
    summary = await generate_summary(state, conclusion=playbook_conclusion)
    context = load_context_for_date(as_of_date)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return ReplayPlaybookResponse(
        as_of=as_of_date.isoformat(),
        snapshot=snapshot,
        state=state,
        playbook_conclusion=playbook_conclusion,
        summary=summary,
        sources=sources,
        context=context,
        data_notes=data_notes,
        generated_at=generated_at,
    )


@router.get("/outcomes", response_model=OutcomeReview)
async def replay_outcomes(
    as_of: str = Query(..., description="Historical date (YYYY-MM-DD) to compute outcomes from"),
) -> OutcomeReview:
    """
    Compute forward price outcomes (SPY, QQQ, WTI) at 1W, 1M, and 3M
    after the given as_of date.

    Fields are None when the horizon is in the future or data is unavailable.
    """
    as_of_date = _parse_and_validate_date(as_of)

    try:
        return await compute_outcomes(as_of_date)
    except Exception as exc:
        logger.error("compute_outcomes failed for %s: %s", as_of, exc)
        raise HTTPException(
            status_code=503,
            detail=f"Failed to compute outcomes for {as_of}: {exc}",
        ) from exc
