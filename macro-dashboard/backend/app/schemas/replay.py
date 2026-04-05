"""
Replay Lab schemas — historical playbook replay, outcome review, and context notes.

No catalysts field: build_catalyst_state() is not date-aware and is intentionally
omitted from replay responses. Curated context notes serve as the historical
forward-watch layer.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.dashboard_state import DashboardState
from app.schemas.indicator_snapshot import IndicatorSnapshot
from app.schemas.playbook_conclusion import PlaybookConclusion
from app.schemas.source_meta import SourceMeta
from app.schemas.summary import PlaybookSummary


class ContextNote(BaseModel):
    """A curated historical context note covering a date range."""
    title: str
    body: str
    tags: list[str] = Field(default_factory=list)


class OutcomePoint(BaseModel):
    """Price outcomes for a single forward time horizon."""
    date_end: str           # ISO date of the horizon endpoint
    # SPY (SPDR S&P 500 ETF) — field names match the ticker used
    spy_return_pct: float | None = None
    spy_price_start: float | None = None
    spy_price_end: float | None = None
    # QQQ (Invesco Nasdaq-100 ETF)
    qqq_return_pct: float | None = None
    qqq_price_start: float | None = None
    qqq_price_end: float | None = None
    # WTI crude (CL=F) — closes the inflation/oil loop for the user
    wti_change_pct: float | None = None
    wti_price_start: float | None = None
    wti_price_end: float | None = None


class OutcomeReview(BaseModel):
    """Forward price outcomes at 1W, 1M, and 3M from the replay date."""
    as_of: str
    outcomes_1w: OutcomePoint
    outcomes_1m: OutcomePoint
    outcomes_3m: OutcomePoint
    data_note: str = ""


class ReplayPlaybookResponse(BaseModel):
    """
    Full replay response for a historical as_of date.

    Intentionally omits `catalysts` — build_catalyst_state() relies on live
    config and is not date-aware. The `context` list (curated notes) serves
    as the historical forward-watch layer instead.
    """
    as_of: str
    snapshot: IndicatorSnapshot
    state: DashboardState
    playbook_conclusion: PlaybookConclusion | None = None
    summary: PlaybookSummary
    sources: dict[str, SourceMeta] = Field(default_factory=dict)
    context: list[ContextNote] = Field(default_factory=list)
    data_notes: list[str] = Field(default_factory=list)
    generated_at: str
