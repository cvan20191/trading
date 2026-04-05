"""
Catalyst schemas — structured speaker-faithful forward-watch logic.

All sections are deliberately optional so the app degrades gracefully when
the local config file is missing or partially filled.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Mega-IPO Watch
# ---------------------------------------------------------------------------

class MegaIPOItem(BaseModel):
    name: str
    status: str = "unknown"
    # Allowed: unknown | rumored | on_track | delayed | weak_demand |
    #          strong_demand | missed_target | completed
    target_valuation: str | None = None
    notes: str | None = None
    impact_label: str | None = None


class MegaIPOWatch(BaseModel):
    items: list[MegaIPOItem] = Field(default_factory=list)
    overall_signal: str = "Neutral"   # Weakening | Supportive | Neutral
    why_it_matters: str = ""
    playbook_impact: str = ""
    provenance: str = "config-backed"   # config-backed | live-linked | mixed


# ---------------------------------------------------------------------------
# Fed Chair Watch
# ---------------------------------------------------------------------------

class FedChairCandidate(BaseModel):
    name: str
    status: str = "unknown"
    # Allowed: front_runner | contender | unlikely | unknown
    tone: str = "unknown"
    # Allowed: bullish_fast_cuts | hawkish_inflation_wary | neutral | unknown
    notes: str | None = None


class FedChairWatch(BaseModel):
    candidates: list[FedChairCandidate] = Field(default_factory=list)
    current_bias: str = "Unknown / Mixed"
    why_it_matters: str = ""
    playbook_impact: str = ""
    provenance: str = "config-backed"


# ---------------------------------------------------------------------------
# Tariff / Twin Deficit / Dollar Watch
# ---------------------------------------------------------------------------

class TariffTwinDeficitWatch(BaseModel):
    dxy: float | None = None
    dxy_pressure: bool = False
    tariff_status: str = "unknown"
    # Allowed: inactive | watch | active | escalating | paused | unknown
    tariff_pressure_active: bool = False
    notes: str | None = None
    why_it_matters: str = ""
    playbook_impact: str = ""
    provenance: str = "mixed"   # DXY is live-linked; tariff status is config-backed


# ---------------------------------------------------------------------------
# Clean-Cut Watch
# ---------------------------------------------------------------------------

class CleanCutWatch(BaseModel):
    clean_cut_window_open: bool = False
    unemployment_condition_met: bool = False
    inflation_condition_met: bool = False
    why_not_open: str | None = None
    why_it_matters: str = ""
    playbook_impact: str = ""
    provenance: str = "live-linked"   # Derived from live unemployment + CPI data


# ---------------------------------------------------------------------------
# Plumbing Watch (repo / reverse repo / reserves)
# ---------------------------------------------------------------------------

class PlumbingWatch(BaseModel):
    repo_status: str = "unknown"
    # Allowed: normal | watch | stress | unknown
    reverse_repo_status: str = "unknown"
    reserve_status: str = "unknown"
    stress_label: str = "Unknown"   # Normal | Watch | Stress | Unknown
    notes: str | None = None
    why_it_matters: str = ""
    playbook_impact: str = ""
    provenance: str = "config-backed"


# ---------------------------------------------------------------------------
# Top-level CatalystState
# ---------------------------------------------------------------------------

class CatalystState(BaseModel):
    mega_ipos: MegaIPOWatch = Field(default_factory=MegaIPOWatch)
    fed_chair: FedChairWatch = Field(default_factory=FedChairWatch)
    tariff_twin_deficit: TariffTwinDeficitWatch = Field(default_factory=TariffTwinDeficitWatch)
    clean_cut_watch: CleanCutWatch = Field(default_factory=CleanCutWatch)
    plumbing_watch: PlumbingWatch = Field(default_factory=PlumbingWatch)
    catalyst_overlays: list[str] = Field(default_factory=list)
    next_lookouts: list[str] = Field(default_factory=list)
    updated_at: str = ""
