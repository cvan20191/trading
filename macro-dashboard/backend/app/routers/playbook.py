from fastapi import APIRouter

from app.schemas.indicator_snapshot import IndicatorSnapshot
from app.schemas.playbook_response import PlaybookResponse
from app.services.catalysts.config_loader import load_catalyst_config
from app.services.catalysts.engine import build_catalyst_state
from app.services.rules.dashboard_state_builder import build_dashboard_state_with_conclusion
from app.services.summary_engine import generate_summary

router = APIRouter(prefix="/api", tags=["playbook"])


@router.post("/playbook", response_model=PlaybookResponse)
async def get_playbook(snapshot: IndicatorSnapshot) -> PlaybookResponse:
    """
    Accept a raw IndicatorSnapshot and return the computed DashboardState,
    LLM-generated (or deterministic fallback) PlaybookSummary, and CatalystState.

    Flow:
      IndicatorSnapshot -> rule engine -> DashboardState -> generate_summary()
                       -> build_catalyst_state() -> PlaybookResponse
    """
    state, playbook_conclusion = build_dashboard_state_with_conclusion(snapshot)
    summary = await generate_summary(state, conclusion=playbook_conclusion)
    catalyst_config = load_catalyst_config()
    catalysts = build_catalyst_state(catalyst_config, snapshot, state)
    return PlaybookResponse(
        state=state,
        playbook_conclusion=playbook_conclusion,
        summary=summary,
        catalysts=catalysts,
    )
