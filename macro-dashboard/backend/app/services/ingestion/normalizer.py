"""
Normalization layer — transforms raw FetchResult dict into IndicatorSnapshot.

Responsible for:
 - trend derivation from recent series
 - CPI YoY / status computation
 - oil risk inference
 - policy support heuristics
 - assembling the final IndicatorSnapshot
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from app.schemas.indicator_snapshot import (
    DataFreshnessInput,
    DollarContextInput,
    GrowthInput,
    IndicatorSnapshot,
    InflationInput,
    LiquidityInput,
    PolicySupportInput,
    SystemicStressInput,
    ValuationInput,
)
from app.services.providers.base import FetchResult
from app.services.providers.cpi_provider import compute_yoy_and_status

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trend helpers
# ---------------------------------------------------------------------------

def _simple_trend(series: list[tuple[str, float]], lookback: int = 4) -> str:
    """
    Compute direction over the last `lookback` observations.
    Returns 'up' | 'down' | 'flat' | 'unknown'.
    """
    if len(series) < 2:
        return "unknown"
    window = series[-lookback:] if len(series) >= lookback else series
    first_val = window[0][1]
    last_val = window[-1][1]
    if first_val == 0:
        return "unknown"
    pct = (last_val - first_val) / abs(first_val)
    if pct > 0.005:
        return "up"
    if pct < -0.005:
        return "down"
    return "flat"


def _trend_1m(series: list[tuple[str, float]]) -> str:
    """Trend over ~4 weekly / daily observations (≈ 1 month)."""
    return _simple_trend(series, lookback=4)


def _trend_3m(series: list[tuple[str, float]]) -> str:
    """Trend over ~12 observations."""
    return _simple_trend(series, lookback=12)


def _compute_cycle_position(series: list[tuple[str, float]]) -> float | None:
    """
    Return the current rate's normalized position within the trailing series range.

    cycle_position = (current - cycle_low) / (cycle_high - cycle_low)

    Returns None when:
    - fewer than 12 observations are available (insufficient history)
    - the range is effectively flat (< 0.05 percentage points)

    This is a heuristic implementation helper, not speaker doctrine.
    It is used by chessboard.py as a secondary input to _policy_stance().
    """
    if len(series) < 12:
        return None
    values = [v for _, v in series]
    lo, hi = min(values), max(values)
    if hi - lo < 0.05:
        # Rate has been essentially flat throughout the window — not useful
        return None
    return round((values[-1] - lo) / (hi - lo), 3)


# ---------------------------------------------------------------------------
# Market-cap / M2 helper
# ---------------------------------------------------------------------------

def _compute_market_cap_m2(sp500_price: float | None, m2: float | None) -> float | None:
    """
    Rough Buffett-indicator-style proxy: total US equity market cap / M2.

    Method:
      - SPY has ~900 million shares outstanding → AUM = price * 0.9 (billions)
      - SPY AUM represents ~1% of total US equity market cap
      - Total US equity ≈ SPY_AUM * 100
      - M2 (WM2NS) is already in billions of dollars

    Example: SPY=$540 → AUM=$486B → total_US=$48,600B; M2=$21,500B → ratio≈2.26
    This is a rough directional proxy, not a precise replication.
    """
    if sp500_price is None or m2 is None or m2 == 0:
        return None
    spy_aum_billions = sp500_price * 0.9        # 900M shares, result in $B
    total_us_equity_billions = spy_aum_billions * 100   # SPY ≈ 1% of US market
    return round(total_us_equity_billions / m2, 3)


# ---------------------------------------------------------------------------
# Main normalizer
# ---------------------------------------------------------------------------

def build_indicator_snapshot(
    raw: dict[str, FetchResult],
    freshness_statuses: dict[str, str],
    stale_series: list[str],
    overall_status: str,
    fed_put: bool = False,
    treasury_put: bool = False,
    political_put: bool = False,
) -> IndicatorSnapshot:
    """
    Transform a dict of FetchResult values into a complete IndicatorSnapshot.

    `raw` keys correspond to series_map.FRED_SERIES and YAHOO_TICKERS keys.
    Missing keys degrade gracefully to None fields.
    """
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def get_val(key: str) -> float | None:
        r = raw.get(key)
        return r.value if r else None

    def get_series(key: str) -> list[tuple[str, float]]:
        r = raw.get(key)
        return r.series if r else []

    # ------------------------------------------------------------------
    # Liquidity
    # ------------------------------------------------------------------
    rate_series = get_series("fed_funds_rate")
    bs_series = get_series("balance_sheet")

    # Infer policy put from rate trend if not explicitly provided
    rate_t1m = _trend_1m(rate_series)
    if not fed_put and rate_t1m == "down":
        fed_put = True
        logger.info("Fed put inferred from falling rate trend")

    liquidity = LiquidityInput(
        fed_funds_rate=get_val("fed_funds_rate"),
        rate_trend_1m=rate_t1m,
        rate_trend_3m=_trend_3m(rate_series),
        balance_sheet_assets=get_val("balance_sheet"),
        balance_sheet_trend_1m=_trend_1m(bs_series),
        balance_sheet_trend_3m=_trend_3m(bs_series),
        rate_cycle_position=_compute_cycle_position(rate_series),
    )

    # ------------------------------------------------------------------
    # Growth
    # ------------------------------------------------------------------
    unemployment_series = get_series("unemployment_rate")
    claims_series = get_series("initial_claims")
    payrolls_series = get_series("nonfarm_payrolls")

    growth = GrowthInput(
        pmi_manufacturing=get_val("pmi_manufacturing"),
        pmi_services=get_val("pmi_services"),
        unemployment_rate=get_val("unemployment_rate"),
        unemployment_trend=_trend_1m(unemployment_series),
        initial_claims_trend=_trend_1m(claims_series),
        payrolls_trend=_trend_1m(payrolls_series),
    )

    # ------------------------------------------------------------------
    # Inflation
    # ------------------------------------------------------------------
    core_result = raw.get("core_cpi")
    shelter_result = raw.get("shelter_cpi")
    services_result = raw.get("services_ex_energy")

    core_yoy, core_mom, _ = compute_yoy_and_status(core_result) if core_result else (None, None, "unknown")
    _, _, shelter_status = compute_yoy_and_status(shelter_result) if shelter_result else (None, None, "unknown")
    _, _, services_status = compute_yoy_and_status(services_result) if services_result else (None, None, "unknown")

    wti_val = get_val("wti_oil")
    oil_risk = (wti_val is not None and wti_val >= 100.0)

    inflation = InflationInput(
        core_cpi_yoy=core_yoy,
        core_cpi_mom=core_mom,
        shelter_status=shelter_status,
        services_ex_energy_status=services_status,
        wti_oil=wti_val,
        oil_risk_active=oil_risk,
    )

    # ------------------------------------------------------------------
    # Valuation — FMP Mag 7 basket (primary) or Yahoo QQQ proxy (fallback).
    # The FetchResult carries all metadata in its extra dict.
    # ------------------------------------------------------------------
    pe_result = raw.get("forward_pe")
    _pe_extra = pe_result.extra if pe_result else {}
    pe_basis = _pe_extra.get("pe_basis", "unavailable")
    pe_source_note = pe_result.note if pe_result else None
    valuation = ValuationInput(
        forward_pe=get_val("forward_pe"),
        pe_basis=pe_basis,
        pe_source_note=pe_source_note,
        metric_name=_pe_extra.get("metric_name"),
        object_label=_pe_extra.get("object_label"),
        pe_provider=_pe_extra.get("provider"),
        coverage_count=_pe_extra.get("coverage_count"),
        coverage_ratio=_pe_extra.get("coverage_ratio"),
        constituents=_pe_extra.get("constituents", []),
    )

    # ------------------------------------------------------------------
    # Systemic Stress
    # ------------------------------------------------------------------
    m2_val = get_val("m2")
    sp500_val = get_val("sp500_etf")
    market_cap_m2 = _compute_market_cap_m2(sp500_val, m2_val)

    stress = SystemicStressInput(
        yield_curve_10y_2y=get_val("yield_curve"),
        npl_ratio=get_val("npl_ratio"),
        market_cap_m2_ratio=market_cap_m2,
    )

    # ------------------------------------------------------------------
    # Dollar context
    # ------------------------------------------------------------------
    dollar = DollarContextInput(dxy=get_val("dxy"))

    # ------------------------------------------------------------------
    # Policy support
    # ------------------------------------------------------------------
    policy = PolicySupportInput(
        fed_put=fed_put,
        treasury_put=treasury_put,
        political_put=political_put,
    )

    # ------------------------------------------------------------------
    # Freshness
    # ------------------------------------------------------------------
    data_freshness = DataFreshnessInput(
        overall_status=overall_status,
        stale_series=stale_series,
    )

    return IndicatorSnapshot(
        as_of=now_iso,
        data_freshness=data_freshness,
        liquidity=liquidity,
        growth=growth,
        inflation=inflation,
        valuation=valuation,
        systemic_stress=stress,
        dollar_context=dollar,
        policy_support=policy,
    )
