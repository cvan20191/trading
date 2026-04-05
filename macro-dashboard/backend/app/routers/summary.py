from fastapi import APIRouter

from app.schemas.dashboard_state import DashboardState
from app.schemas.summary import PlaybookSummary
from app.services.summary_engine import generate_summary

router = APIRouter(prefix="/api", tags=["summary"])


@router.post("/summary", response_model=PlaybookSummary)
async def get_summary(state: DashboardState) -> PlaybookSummary:
    """
    Accept a structured DashboardState and return a PlaybookSummary.

    Always returns a valid response — the LLM fallback path is transparent
    to the caller (visible only through the `meta.used_fallback` field).
    """
    return await generate_summary(state)
