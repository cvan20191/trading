"""
Live response schemas — wraps IndicatorSnapshot with source provenance.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.catalysts import CatalystState
from app.schemas.dashboard_state import DashboardState
from app.schemas.indicator_snapshot import IndicatorSnapshot
from app.schemas.playbook_conclusion import PlaybookConclusion
from app.schemas.playbook_response import PlaybookResponse  # noqa: F401  (re-exported)
from app.schemas.source_meta import SourceMeta
from app.schemas.summary import PlaybookSummary


class LiveSnapshotResponse(BaseModel):
    snapshot: IndicatorSnapshot
    sources: dict[str, SourceMeta]
    overall_status: str          # fresh | mixed | stale
    stale_series: list[str]
    generated_at: str


class LivePlaybookResponse(BaseModel):
    snapshot: IndicatorSnapshot
    state: DashboardState
    playbook_conclusion: PlaybookConclusion | None = None
    summary: PlaybookSummary
    catalysts: CatalystState = Field(default_factory=CatalystState)
    sources: dict[str, SourceMeta]
    overall_status: str
    stale_series: list[str]
    generated_at: str
