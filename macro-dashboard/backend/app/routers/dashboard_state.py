from fastapi import APIRouter

from app.schemas.dashboard_state import DashboardState
from app.schemas.indicator_snapshot import IndicatorSnapshot
from app.services.rules.dashboard_state_builder import build_dashboard_state

router = APIRouter(prefix="/api", tags=["rule-engine"])


@router.post("/dashboard-state", response_model=DashboardState)
async def get_dashboard_state(snapshot: IndicatorSnapshot) -> DashboardState:
    """
    Accept a raw IndicatorSnapshot and return the computed DashboardState.

    This endpoint is for debugging, inspection, and direct consumption by
    any client that wants the structured state without the LLM summary.
    The rule engine is entirely deterministic — no LLM involved.
    """
    return build_dashboard_state(snapshot)
