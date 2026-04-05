from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.catalysts import CatalystState
from app.schemas.dashboard_state import DashboardState
from app.schemas.playbook_conclusion import PlaybookConclusion
from app.schemas.summary import PlaybookSummary


class PlaybookResponse(BaseModel):
    """Combined response for POST /api/playbook."""

    state: DashboardState
    playbook_conclusion: PlaybookConclusion | None = None
    summary: PlaybookSummary
    catalysts: CatalystState = Field(default_factory=CatalystState)
