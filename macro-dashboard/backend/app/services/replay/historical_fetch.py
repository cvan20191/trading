"""
Historical fetch wrappers — thin adapters over existing providers that accept
an as_of_date argument to freeze data to a specific historical point.

IMPORTANT DATA NOTES (always surfaced in data_notes):
- FRED observation_end filters by observation date, NOT release date. Data may
  include post-release revisions that were not available at the time.
- Forward P/E is always None for historical dates — analyst estimate archives
  are not supported in the MVP.
- PMI uses S&P Global proxy series (MFGPMNBUS / SVPMNBUS), not ISM.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from app.services.ingestion.series_map import FRED_SERIES, PROVIDER_FRED, PROVIDER_YAHOO
from app.services.providers.base import FetchResult, ProviderError
from app.services.providers.fred_client import fetch_series

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FRED series IDs for historical PMI (S&P Global proxies available on FRED)
# ---------------------------------------------------------------------------
_PMI_MFG_FRED = "MFGPMNBUS"       # S&P Global Manufacturing PMI, US, 2012+
_PMI_SVC_FRED = "SVPMNBUS"        # S&P Global Services PMI, US, 2012+

# Note added when S&P Global proxy series are used
PMI_PROXY_NOTE = (
    "PMI sourced from S&P Global proxy series (MFGPMNBUS / SVPMNBUS) — "
    "not ISM Manufacturing / Services; methodology and thresholds differ slightly"
)

# Note always added for FRED vintage caveat
FRED_VINTAGE_NOTE = (
    "FRED data reflects currently-known historical values, not real-time vintage "
    "as published on that date. Post-release revisions are included."
)

# Note added for forward P/E
FWD_PE_UNAVAILABLE_NOTE = (
    "Forward P/E unavailable for historical dates — "
    "analyst estimate archives not supported in MVP."
)


# ---------------------------------------------------------------------------
# FRED series wrappers
# ---------------------------------------------------------------------------

def fetch_fred_series_as_of(
    series_id: str,
    series_name: str,
    frequency: str,
    as_of_date: date,
    observation_window_days: int = 180,
    api_key: str = "",
    timeout: int = 20,
) -> FetchResult:
    """Fetch a FRED series up to and including as_of_date."""
    return fetch_series(
        series_id=series_id,
        series_name=series_name,
        frequency=frequency,
        observation_window_days=observation_window_days,
        api_key=api_key,
        timeout=timeout,
        observation_end_date=as_of_date.isoformat(),
    )


# ---------------------------------------------------------------------------
# Yahoo price history wrapper
# ---------------------------------------------------------------------------

def fetch_yahoo_ticker_as_of(
    ticker: str,
    series_name: str,
    as_of_date: date,
    window_days: int = 180,
    timeout: int = 20,
) -> FetchResult:
    """Fetch Yahoo Finance price history up to as_of_date."""
    try:
        import yfinance as yf  # type: ignore[import]
    except ImportError as exc:
        raise ProviderError("yfinance is not installed") from exc

    end = as_of_date
    start = end - timedelta(days=window_days)

    try:
        df = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),  # yfinance end is exclusive
            progress=False,
            auto_adjust=True,
            timeout=timeout,
        )
    except Exception as exc:
        raise ProviderError(f"yfinance historical download failed for {ticker}: {exc}") from exc

    if df.empty:
        return FetchResult(
            value=None,
            observed_at=None,
            series=[],
            provider=PROVIDER_YAHOO,
            series_id=ticker,
            series_name=series_name,
            frequency="daily",
            status="missing",
            note=f"No historical data returned by Yahoo Finance for {ticker} up to {as_of_date}",
        )

    close = df["Close"].squeeze()
    series = [
        (str(idx.date()), float(val))
        for idx, val in zip(close.index, close.values)
        if val is not None
    ]
    if not series:
        return FetchResult(
            value=None,
            observed_at=None,
            series=[],
            provider=PROVIDER_YAHOO,
            series_id=ticker,
            series_name=series_name,
            frequency="daily",
            status="missing",
            note=f"No usable close prices for {ticker} up to {as_of_date}",
        )

    latest_date, latest_value = series[-1]
    return FetchResult(
        value=latest_value,
        observed_at=latest_date,
        series=series,
        provider=PROVIDER_YAHOO,
        series_id=ticker,
        series_name=series_name,
        frequency="daily",
        status="fresh",
    )


# ---------------------------------------------------------------------------
# CPI historical wrapper
# ---------------------------------------------------------------------------

def fetch_historical_cpi_components_as_of(
    as_of_date: date,
    api_key: str = "",
    timeout: int = 20,
) -> dict[str, FetchResult]:
    """Fetch core CPI, shelter CPI, and services-less-energy from FRED up to as_of_date."""
    results: dict[str, FetchResult] = {}
    for key in ("core_cpi", "shelter_cpi", "services_ex_energy"):
        series_id = FRED_SERIES[key]
        try:
            results[key] = fetch_fred_series_as_of(
                series_id=series_id,
                series_name=key,
                frequency="monthly",
                as_of_date=as_of_date,
                observation_window_days=430,   # ~14 months for YoY calc
                api_key=api_key,
                timeout=timeout,
            )
        except ProviderError as exc:
            logger.warning("Historical CPI fetch failed for %s: %s", key, exc)
            results[key] = FetchResult(
                value=None,
                observed_at=None,
                series=[],
                provider=PROVIDER_FRED,
                series_id=series_id,
                series_name=key,
                frequency="monthly",
                status="error",
                note=str(exc),
            )
    return results


# ---------------------------------------------------------------------------
# PMI historical wrapper (S&P Global proxy via FRED)
# ---------------------------------------------------------------------------

def fetch_historical_pmi_as_of(
    as_of_date: date,
    api_key: str = "",
    timeout: int = 20,
) -> tuple[dict[str, FetchResult], bool]:
    """
    Fetch historical PMI from FRED S&P Global proxy series.

    Returns (results_dict, pmi_proxy_used).
    pmi_proxy_used=True triggers the S&P Global methodology note in data_notes.
    """
    pmi_proxy_used = False
    results: dict[str, FetchResult] = {}

    for key, series_id, name in [
        ("pmi_manufacturing", _PMI_MFG_FRED, "S&P Global Manufacturing PMI (US)"),
        ("pmi_services",      _PMI_SVC_FRED, "S&P Global Services PMI (US)"),
    ]:
        try:
            result = fetch_fred_series_as_of(
                series_id=series_id,
                series_name=name,
                frequency="monthly",
                as_of_date=as_of_date,
                observation_window_days=120,
                api_key=api_key,
                timeout=timeout,
            )
            if result.value is not None:
                pmi_proxy_used = True
            results[key] = result
        except ProviderError as exc:
            logger.warning("Historical PMI fetch failed for %s: %s", key, exc)
            results[key] = FetchResult(
                value=None,
                observed_at=None,
                series=[],
                provider=PROVIDER_FRED,
                series_id=series_id,
                series_name=name,
                frequency="monthly",
                status="missing",
                note=f"PMI unavailable for historical date {as_of_date}",
            )

    return results, pmi_proxy_used


# ---------------------------------------------------------------------------
# Forward P/E — always unavailable for historical dates
# ---------------------------------------------------------------------------

def make_historical_pe_unavailable() -> FetchResult:
    """
    Returns a FetchResult indicating forward P/E is unavailable for historical dates.
    ValuationTriggerCard handles value=None with its existing 'data unavailable' state.
    is_fallback is intentionally False — this is not a proxy; it's genuinely absent.
    """
    return FetchResult(
        value=None,
        observed_at=None,
        series=[],
        provider="replay",
        series_id="forward_pe",
        series_name="Forward P/E (historical)",
        frequency="daily",
        status="missing",
        note=FWD_PE_UNAVAILABLE_NOTE,
        extra={
            "pe_basis": "unavailable",
            "metric_name": "Forward P/E",
            "object_label": "Historical — unavailable",
            "provider": "replay",
            "coverage_count": None,
            "coverage_ratio": None,
        },
    )
