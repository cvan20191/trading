from __future__ import annotations

from pydantic import BaseModel, Field


class DataFreshness(BaseModel):
    overall_status: str = "unknown"
    stale_series: list[str] = Field(default_factory=list)


class FedChessboard(BaseModel):
    quadrant: str | None = None
    label: str | None = None
    rate_trend_1m: str | None = None
    rate_trend_3m: str | None = None
    balance_sheet_trend_1m: str | None = None
    balance_sheet_trend_3m: str | None = None
    direction_vs_1m_ago: str | None = None
    # Two-layer chessboard fields — added by policy stance refactor
    policy_stance: str | None = None           # "easy" | "middle" | "restrictive"
    rate_impulse: str | None = None            # "easing" | "stable" | "tightening"
    balance_sheet_direction: str | None = None # "expanding" | "flat_or_mixed" | "contracting"
    transition_tag: str | None = None          # "Improving" | "Stable" | "Deteriorating"


class StagflationTrap(BaseModel):
    active: bool = False
    growth_weakening: bool = False
    sticky_inflation: bool = False
    pmi_manufacturing: float | None = None
    pmi_services: float | None = None
    unemployment_rate: float | None = None
    core_cpi_yoy: float | None = None
    shelter_status: str | None = None
    services_ex_energy_status: str | None = None
    wti_oil: float | None = None
    oil_risk_active: bool = False


class ValuationConstituent(BaseModel):
    ticker: str
    price: float | None = None
    forward_eps: float | None = None
    forward_pe: float | None = None


class Valuation(BaseModel):
    forward_pe: float | None = None
    zone: str | None = None
    zone_label: str | None = None
    buy_zone_low: float | None = None
    buy_zone_high: float | None = None
    pause_threshold: float | None = None
    # ---- valuation data quality fields ----
    # basis: forward | trailing | ttm_derived | unavailable
    basis: str = "unavailable"
    # human-readable basis label shown in the UI
    basis_label: str = "Unavailable"
    # provider note explaining the exact source
    source_note: str | None = None
    # True whenever basis != "forward" and a value is present
    is_fallback: bool = False
    # ---- metric identity fields ----
    metric_name: str | None = None          # e.g. "Mag 7 Forward P/E"
    object_label: str | None = None         # e.g. "Mag 7 Basket"
    provider: str | None = None             # e.g. "fmp" or "yahoo"
    coverage_count: int | None = None       # number of valid basket constituents
    coverage_ratio: float | None = None     # fraction of Mag 7 market cap included
    # per-ticker breakdown for the FMP Mag 7 basket
    constituents: list[ValuationConstituent] = Field(default_factory=list)


class SystemicStress(BaseModel):
    yield_curve_inverted: bool = False
    yield_curve_value: float | None = None
    npl_ratio: float | None = None
    npl_zone: str | None = None
    market_cap_m2_ratio: float | None = None
    market_cap_m2_zone: str | None = None


class DollarContext(BaseModel):
    dxy: float | None = None
    dxy_pressure: bool = False


class RallyConditions(BaseModel):
    rally_fuel_score: int | None = None
    fed_put: bool = False
    treasury_put: bool = False
    political_put: bool = False
    market_ignoring_bad_news: bool = False


class DashboardState(BaseModel):
    as_of: str | None = None
    data_freshness: DataFreshness = Field(default_factory=DataFreshness)
    primary_regime: str
    secondary_overlays: list[str] = Field(default_factory=list)
    confidence: str = "Medium"
    current_posture: str
    fed_chessboard: FedChessboard | None = None
    stagflation_trap: StagflationTrap | None = None
    valuation: Valuation | None = None
    systemic_stress: SystemicStress | None = None
    dollar_context: DollarContext | None = None
    rally_conditions: RallyConditions | None = None
    top_watchpoints: list[str] = Field(default_factory=list)
    what_changed: list[str] = Field(default_factory=list)
    what_changes_call: list[str] = Field(default_factory=list)
