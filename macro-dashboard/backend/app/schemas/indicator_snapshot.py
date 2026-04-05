"""
IndicatorSnapshot — raw macro indicator input for the deterministic rule engine.

All fields are optional to allow partial data ingestion without hard failures.
The rule engine degrades gracefully when fields are absent.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DataFreshnessInput(BaseModel):
    overall_status: str | None = None
    stale_series: list[str] = Field(default_factory=list)


class LiquidityInput(BaseModel):
    fed_funds_rate: float | None = None
    rate_trend_1m: str | None = None       # "up" | "down" | "flat"
    rate_trend_3m: str | None = None
    balance_sheet_assets: float | None = None
    balance_sheet_trend_1m: str | None = None   # "up" | "down" | "flat"
    balance_sheet_trend_3m: str | None = None
    # Normalized position of current rate within trailing cycle range.
    # 0.0 = at cycle low, 1.0 = at cycle high.
    # Computed by normalizer from trailing 36-month rate history.
    # None when insufficient history is available.
    rate_cycle_position: float | None = None


class GrowthInput(BaseModel):
    pmi_manufacturing: float | None = None
    pmi_services: float | None = None
    unemployment_rate: float | None = None
    unemployment_trend: str | None = None  # "up" | "down" | "flat"
    initial_claims_trend: str | None = None
    payrolls_trend: str | None = None


class InflationInput(BaseModel):
    core_cpi_yoy: float | None = None
    core_cpi_mom: float | None = None
    shelter_status: str | None = None          # "sticky" | "easing" | "neutral"
    services_ex_energy_status: str | None = None
    wti_oil: float | None = None
    oil_risk_active: bool | None = None


class ValuationInput(BaseModel):
    forward_pe: float | None = None
    # forward | trailing | ttm_derived | unavailable — carried from the provider
    pe_basis: str = "unavailable"
    # raw provider note (e.g. "Mag 7 market-cap-weighted forward P/E — 7/7 constituents")
    pe_source_note: str | None = None
    # human-readable metric descriptor (e.g. "Mag 7 Forward P/E" or "QQQ P/E Proxy")
    metric_name: str | None = None
    # short object label (e.g. "Mag 7 Basket" or "QQQ (QQQ)")
    object_label: str | None = None
    # provider slug (e.g. "fmp" or "yahoo")
    pe_provider: str | None = None
    # basket coverage (None for non-basket sources)
    coverage_count: int | None = None
    coverage_ratio: float | None = None
    # per-ticker breakdown — only populated for the FMP Mag 7 basket
    # each entry: {ticker, price, forward_eps, forward_pe} (forward_pe may be None)
    constituents: list[dict] = Field(default_factory=list)


class SystemicStressInput(BaseModel):
    yield_curve_10y_2y: float | None = None
    npl_ratio: float | None = None
    market_cap_m2_ratio: float | None = None


class DollarContextInput(BaseModel):
    dxy: float | None = None


class PolicySupportInput(BaseModel):
    fed_put: bool | None = None
    treasury_put: bool | None = None
    political_put: bool | None = None


class IndicatorSnapshot(BaseModel):
    as_of: str | None = None
    data_freshness: DataFreshnessInput = Field(default_factory=DataFreshnessInput)
    liquidity: LiquidityInput = Field(default_factory=LiquidityInput)
    growth: GrowthInput = Field(default_factory=GrowthInput)
    inflation: InflationInput = Field(default_factory=InflationInput)
    valuation: ValuationInput = Field(default_factory=ValuationInput)
    systemic_stress: SystemicStressInput = Field(default_factory=SystemicStressInput)
    dollar_context: DollarContextInput = Field(default_factory=DollarContextInput)
    policy_support: PolicySupportInput = Field(default_factory=PolicySupportInput)
