"""
Historical snapshot builder — orchestrates all historical fetches and assembles
a complete IndicatorSnapshot for a given as_of date.

Returns the snapshot, source metadata dict, and a list of data_notes that
explain any data quality limitations (vintage, proxy series, missing fields).
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone

from app.schemas.indicator_snapshot import IndicatorSnapshot
from app.schemas.source_meta import SourceMeta
from app.services.ingestion.normalizer import build_indicator_snapshot
from app.services.ingestion.series_map import FRED_SERIES, YAHOO_TICKERS
from app.services.providers.base import FetchResult, ProviderError
from app.services.replay.historical_fetch import (
    FRED_VINTAGE_NOTE,
    FWD_PE_UNAVAILABLE_NOTE,
    PMI_PROXY_NOTE,
    fetch_fred_series_as_of,
    fetch_historical_cpi_components_as_of,
    fetch_historical_pmi_as_of,
    fetch_yahoo_ticker_as_of,
    make_historical_pe_unavailable,
)

logger = logging.getLogger(__name__)

# FRED series + metadata for historical fetch (mirrors live_snapshot_service _FRED_FETCH_MAP)
_FRED_HIST_MAP: dict[str, tuple[str, str, str, int]] = {
    # key: (series_id, display_name, frequency, window_days)
    # 1100d window so normalizer has 36 months of history for cycle position computation
    "fed_funds_rate":    (FRED_SERIES["fed_funds_rate"],    "Fed Funds Upper Bound",        "daily",     1100),
    "balance_sheet":     (FRED_SERIES["balance_sheet"],     "Fed Balance Sheet",            "weekly",    120),
    "unemployment_rate": (FRED_SERIES["unemployment_rate"], "Unemployment Rate",            "monthly",   180),
    "initial_claims":    (FRED_SERIES["initial_claims"],    "Initial Jobless Claims",       "weekly",    120),
    "nonfarm_payrolls":  (FRED_SERIES["nonfarm_payrolls"],  "Nonfarm Payrolls",             "monthly",   180),
    "yield_curve":       (FRED_SERIES["yield_curve"],       "10Y-2Y Yield Spread",          "daily",     120),
    "npl_ratio":         (FRED_SERIES["npl_ratio"],         "Delinquency Rate (NPL Proxy)", "quarterly", 180),
    "m2":                (FRED_SERIES["m2"],                "M2 Money Stock",               "weekly",    120),
}

_YAHOO_HIST_MAP: dict[str, tuple[str, str]] = {
    "wti_oil":   (YAHOO_TICKERS["wti_oil"],   "WTI Crude Oil"),
    "dxy":       (YAHOO_TICKERS["dxy"],       "US Dollar Index"),
    "sp500_etf": (YAHOO_TICKERS["sp500_etf"], "S&P 500 ETF (SPY)"),
}


def _make_error_result(key: str, series_id: str, name: str, frequency: str, note: str) -> FetchResult:
    return FetchResult(
        value=None, observed_at=None, series=[],
        provider="FRED", series_id=series_id, series_name=name,
        frequency=frequency, status="error", note=note,
    )


def _build_historical_source_meta(
    raw: dict[str, FetchResult],
    as_of_date: date,
) -> dict[str, SourceMeta]:
    """Build SourceMeta for each fetched result, tagged as historical."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta: dict[str, SourceMeta] = {}
    for key, result in raw.items():
        is_valuation = key == "forward_pe"
        provider_label = result.provider
        if is_valuation and result.extra.get("provider"):
            provider_label = result.extra["provider"]
        meta[key] = SourceMeta(
            provider=provider_label,
            series_name=result.series_name,
            series_id=result.series_id,
            fetched_at=now,
            observed_at=result.observed_at,
            frequency=result.frequency,
            status=result.status,
            note=result.note,
            basis=result.extra.get("pe_basis") if is_valuation else None,
        )
    return meta


async def build_historical_snapshot(
    as_of_date: date,
    fred_api_key: str,
    http_timeout: int = 20,
) -> tuple[IndicatorSnapshot, dict[str, SourceMeta], list[str]]:
    """
    Build a complete IndicatorSnapshot for the given historical as_of_date.

    Returns:
        snapshot: IndicatorSnapshot frozen to as_of_date
        sources:  dict[str, SourceMeta] for the debug panel
        data_notes: list of transparency strings about data quality
    """
    import asyncio
    data_notes: list[str] = [FRED_VINTAGE_NOTE]
    raw: dict[str, FetchResult] = {}

    # ── FRED series (run all in thread pool) ─────────────────────────────────
    def _fetch_all_fred() -> dict[str, FetchResult]:
        results: dict[str, FetchResult] = {}
        for key, (series_id, name, freq, window) in _FRED_HIST_MAP.items():
            try:
                results[key] = fetch_fred_series_as_of(
                    series_id=series_id,
                    series_name=name,
                    frequency=freq,
                    as_of_date=as_of_date,
                    observation_window_days=window,
                    api_key=fred_api_key,
                    timeout=http_timeout,
                )
            except ProviderError as exc:
                logger.warning("Historical FRED fetch failed for %s: %s", key, exc)
                results[key] = _make_error_result(key, series_id, name, freq, str(exc))
        return results

    # ── Yahoo price series ────────────────────────────────────────────────────
    def _fetch_all_yahoo() -> dict[str, FetchResult]:
        results: dict[str, FetchResult] = {}
        for key, (ticker, name) in _YAHOO_HIST_MAP.items():
            try:
                results[key] = fetch_yahoo_ticker_as_of(
                    ticker=ticker,
                    series_name=name,
                    as_of_date=as_of_date,
                    window_days=180,
                    timeout=http_timeout,
                )
            except ProviderError as exc:
                logger.warning("Historical Yahoo fetch failed for %s: %s", key, exc)
                from app.services.ingestion.series_map import PROVIDER_YAHOO
                results[key] = FetchResult(
                    value=None, observed_at=None, series=[],
                    provider=PROVIDER_YAHOO, series_id=ticker,
                    series_name=name, frequency="daily",
                    status="error", note=str(exc),
                )
        return results

    # ── CPI ───────────────────────────────────────────────────────────────────
    def _fetch_cpi() -> dict[str, FetchResult]:
        return fetch_historical_cpi_components_as_of(
            as_of_date=as_of_date,
            api_key=fred_api_key,
            timeout=http_timeout,
        )

    # ── PMI ───────────────────────────────────────────────────────────────────
    def _fetch_pmi() -> tuple[dict[str, FetchResult], bool]:
        return fetch_historical_pmi_as_of(
            as_of_date=as_of_date,
            api_key=fred_api_key,
            timeout=http_timeout,
        )

    # Run all fetches concurrently in thread pool
    fred_raw, yahoo_raw, cpi_raw, (pmi_raw, pmi_proxy_used) = await asyncio.gather(
        asyncio.to_thread(_fetch_all_fred),
        asyncio.to_thread(_fetch_all_yahoo),
        asyncio.to_thread(_fetch_cpi),
        asyncio.to_thread(_fetch_pmi),
    )

    raw.update(fred_raw)
    raw.update(yahoo_raw)
    raw.update(cpi_raw)
    raw.update(pmi_raw)

    # Forward P/E — always unavailable for historical dates
    raw["forward_pe"] = make_historical_pe_unavailable()
    data_notes.append(FWD_PE_UNAVAILABLE_NOTE)

    # PMI data note
    if pmi_proxy_used:
        data_notes.append(PMI_PROXY_NOTE)
    else:
        # Check if PMI came back at all
        mfg = pmi_raw.get("pmi_manufacturing")
        if not mfg or mfg.value is None:
            data_notes.append(f"PMI unavailable for historical date {as_of_date}.")

    # Normalise all freshness as "historical" — staleness rules don't apply
    freshness_statuses = {k: "historical" for k in raw}
    stale_series: list[str] = []
    overall_status = "historical"

    # Assemble IndicatorSnapshot using the unchanged normalizer
    snapshot = build_indicator_snapshot(
        raw=raw,
        freshness_statuses=freshness_statuses,
        stale_series=stale_series,
        overall_status=overall_status,
        fed_put=False,
        treasury_put=False,
        political_put=False,
    )
    # Override as_of to reflect the requested replay date
    snapshot = snapshot.model_copy(update={"as_of": as_of_date.isoformat()})

    source_meta = _build_historical_source_meta(raw, as_of_date)

    return snapshot, source_meta, data_notes
