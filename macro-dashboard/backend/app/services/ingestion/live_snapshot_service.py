"""
Live snapshot service — orchestrates provider fetching, normalization,
freshness evaluation, and caching into a single async interface.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from functools import partial

from app.config import settings
from app.schemas.indicator_snapshot import IndicatorSnapshot
from app.schemas.live_snapshot_response import LivePlaybookResponse, LiveSnapshotResponse
from app.schemas.source_meta import SourceMeta
from app.services.ingestion.cache import get_snapshot_cache
from app.services.ingestion.freshness import classify_result_freshness, compute_overall_freshness
from app.services.ingestion.normalizer import build_indicator_snapshot
from app.services.ingestion.series_map import (
    FRED_SERIES,
    PROVIDER_FRED,
    PROVIDER_STUB,
    PROVIDER_YAHOO,
    YAHOO_TICKERS,
)
from app.services.providers.base import FetchResult, ProviderError
from app.services.providers.cpi_provider import fetch_cpi_components
from app.services.providers.fmp_client import compute_mag7_basket
from app.services.providers.fred_client import fetch_series
from app.services.providers.pmi_provider import fetch_pmi_manufacturing, fetch_pmi_services
from app.services.providers.yahoo_client import fetch_pe_ratio_proxy, fetch_ticker
from app.services.catalysts.config_loader import load_catalyst_config
from app.services.catalysts.engine import build_catalyst_state
from app.services.rules.dashboard_state_builder import build_dashboard_state_with_conclusion
from app.services.summary_engine import generate_summary

logger = logging.getLogger(__name__)

# Bump when snapshot shape / provider mix changes so TTL cache + disk recovery are not stale.
_LIVE_SNAPSHOT_CACHE_VERSION = 4  # bumped: added per-ticker constituents to valuation
# Same-day in-memory cache for successful FMP Mag 7 valuation results only.
_FMP_VALUATION_DAY_CACHE: dict[str, FetchResult] = {}


def _strip_internal_cache_keys(payload: dict) -> dict:
    """Remove keys we add only for cache invalidation (not part of API schema)."""
    return {k: v for k, v in payload.items() if not str(k).startswith("_")}


def _missing_cache_critical_fields(snapshot: IndicatorSnapshot) -> list[str]:
    """
    Doctrine-critical fields that must exist before persisting live cache.
    """
    missing: list[str] = []
    if snapshot.liquidity.fed_funds_rate is None:
        missing.append("liquidity.fed_funds_rate")
    if snapshot.liquidity.balance_sheet_assets is None:
        missing.append("liquidity.balance_sheet_assets")
    if snapshot.valuation.forward_pe is None:
        missing.append("valuation.forward_pe")
    return missing


# ---------------------------------------------------------------------------
# Individual FRED series to fetch (outside of CPI block)
# ---------------------------------------------------------------------------
_FRED_FETCH_MAP: dict[str, tuple[str, str, int, int]] = {
    # key: (series_id, display_name, frequency_days, window_days)
    # window_days controls how much history to fetch; most series use 120d.
    # fed_funds_rate uses 1100d (~36 months) so the normalizer can compute
    # a cycle position for the chessboard's policy stance inference.
    "fed_funds_rate":    (FRED_SERIES["fed_funds_rate"],   "Fed Funds Upper Bound",        5,   1100),
    "balance_sheet":     (FRED_SERIES["balance_sheet"],    "Fed Balance Sheet",            10,  120),
    "unemployment_rate": (FRED_SERIES["unemployment_rate"], "Unemployment Rate",           35,  120),
    "initial_claims":    (FRED_SERIES["initial_claims"],   "Initial Jobless Claims",       10,  120),
    "nonfarm_payrolls":  (FRED_SERIES["nonfarm_payrolls"], "Nonfarm Payrolls",             40,  120),
    "yield_curve":       (FRED_SERIES["yield_curve"],      "10Y-2Y Yield Spread",          5,   120),
    "npl_ratio":         (FRED_SERIES["npl_ratio"],        "Delinquency Rate (NPL Proxy)", 120, 120),
    "m2":                (FRED_SERIES["m2"],               "M2 Money Stock",               10,  120),
}

_YAHOO_FETCH_MAP: dict[str, tuple[str, str]] = {
    "wti_oil":   (YAHOO_TICKERS["wti_oil"],    "WTI Crude Oil"),
    "dxy":       (YAHOO_TICKERS["dxy"],        "US Dollar Index"),
    "sp500_etf": (YAHOO_TICKERS["sp500_etf"],  "S&P 500 ETF (SPY)"),
}

_FREQUENCY_LABELS: dict[int, str] = {
    5:   "daily",
    10:  "weekly",
    35:  "monthly",
    40:  "monthly",
    100: "quarterly",
}


def _freq_label(days: int) -> str:
    return _FREQUENCY_LABELS.get(days, "periodic")


# ---------------------------------------------------------------------------
# Blocking fetch helpers (run in thread pool via asyncio.to_thread)
# ---------------------------------------------------------------------------

def _fetch_fred_all(api_key: str, timeout: int) -> dict[str, FetchResult]:
    results: dict[str, FetchResult] = {}
    for key, (series_id, name, freq_days, window_days) in _FRED_FETCH_MAP.items():
        try:
            results[key] = fetch_series(
                series_id=series_id,
                series_name=name,
                frequency=_freq_label(freq_days),
                observation_window_days=window_days,
                api_key=api_key,
                timeout=timeout,
            )
        except ProviderError as exc:
            logger.warning("FRED fetch failed for %s: %s", key, exc)
            results[key] = FetchResult(
                value=None,
                observed_at=None,
                series=[],
                provider=PROVIDER_FRED,
                series_id=series_id,
                series_name=name,
                frequency=_freq_label(freq_days),
                status="error",
                note=str(exc),
            )
    return results


def _fetch_yahoo_all(timeout: int) -> dict[str, FetchResult]:
    results: dict[str, FetchResult] = {}
    for key, (ticker, name) in _YAHOO_FETCH_MAP.items():
        try:
            results[key] = fetch_ticker(
                ticker=ticker,
                series_name=name,
                frequency="daily",
                window_days=90,
                timeout=timeout,
            )
        except ProviderError as exc:
            logger.warning("Yahoo fetch failed for %s: %s", key, exc)
            results[key] = FetchResult(
                value=None,
                observed_at=None,
                series=[],
                provider=PROVIDER_YAHOO,
                series_id=ticker,
                series_name=name,
                frequency="daily",
                status="error",
                note=str(exc),
            )
    return results


def _fetch_yahoo_valuation_fallback(timeout: int) -> FetchResult:
    """Yahoo QQQ P/E — used as fallback when FMP Mag 7 basket is unavailable."""
    qqq_ticker = YAHOO_TICKERS["nasdaq_etf"]
    try:
        return fetch_pe_ratio_proxy(
            ticker=qqq_ticker,
            series_name="Nasdaq 100 ETF (QQQ) P/E proxy",
        )
    except ProviderError as exc:
        logger.warning("Yahoo P/E fallback fetch failed: %s", exc)
        return FetchResult(
            value=None,
            observed_at=None,
            series=[],
            provider=PROVIDER_YAHOO,
            series_id=qqq_ticker,
            series_name="Nasdaq 100 ETF (QQQ) P/E proxy",
            frequency="quote",
            status="error",
            note=str(exc),
            extra={
                "pe_basis": "unavailable",
                "metric_name": "QQQ P/E Proxy",
                "object_label": f"QQQ ({qqq_ticker})",
                "provider": "yahoo",
                "coverage_count": None,
                "coverage_ratio": None,
            },
        )


def _fetch_valuation(timeout: int, force_refresh: bool = False) -> FetchResult:
    """
    Valuation fetch hierarchy:
      1. FMP Mag 7 Forward P/E basket  (primary — basis="forward")
      2. Yahoo QQQ P/E proxy           (fallback — basis="trailing"/"ttm_derived")
      3. Unavailable                   (both fail)

    Uses VALUATION_PROVIDER setting: "fmp" tries FMP first; "yahoo" skips straight
    to Yahoo (useful for testing the fallback path).
    """
    fmp_key = settings.fmp_api_key
    provider_pref = settings.valuation_provider.lower()

    # Allow bypass to Yahoo for testing or if FMP key is not configured
    if provider_pref == "yahoo" or not fmp_key:
        reason = "VALUATION_PROVIDER=yahoo" if provider_pref == "yahoo" else "FMP_API_KEY not configured"
        logger.info("Valuation: using Yahoo fallback (%s)", reason)
        return _fetch_yahoo_valuation_fallback(timeout)

    cache_day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if not force_refresh:
        cached = _FMP_VALUATION_DAY_CACHE.get(cache_day)
        if cached is not None and cached.value is not None:
            logger.info(
                "Valuation: reusing same-day cached FMP Mag 7 basket (%s, P/E=%.2f)",
                cache_day,
                cached.value,
            )
            return cached

    # Try FMP Mag 7 basket
    try:
        result = compute_mag7_basket(api_key=fmp_key, timeout=timeout)
        if result.value is not None:
            _FMP_VALUATION_DAY_CACHE[cache_day] = result
            logger.info("Valuation: FMP Mag 7 basket succeeded (P/E=%.2f)", result.value)
            return result
        # basket returned but coverage was insufficient
        logger.warning("Valuation: FMP basket insufficient coverage — falling back to Yahoo")
    except ProviderError as exc:
        logger.warning("Valuation: FMP basket failed (%s) — falling back to Yahoo", exc)

    # Fallback: Yahoo QQQ proxy
    return _fetch_yahoo_valuation_fallback(timeout)


def _fetch_cpi_all(api_key: str, timeout: int) -> dict[str, FetchResult]:
    return fetch_cpi_components(api_key=api_key, timeout=timeout)


def _manual_pmi_result(series_id: str, series_name: str, value: float) -> FetchResult:
    return FetchResult(
        value=value,
        observed_at=None,
        series=[],
        provider=PROVIDER_STUB,
        series_id=series_id,
        series_name=series_name,
        frequency="monthly",
        status="fallback",
        note=f"Value injected from live request query param ({series_id})",
    )


def _fetch_pmi_all(
    pmi_manufacturing_override: float | None = None,
    pmi_services_override: float | None = None,
) -> dict[str, FetchResult]:
    return {
        "pmi_manufacturing": (
            _manual_pmi_result("PMI_MANUFACTURING", "pmi_manufacturing", pmi_manufacturing_override)
            if pmi_manufacturing_override is not None
            else fetch_pmi_manufacturing()
        ),
        "pmi_services": (
            _manual_pmi_result("PMI_SERVICES", "pmi_services", pmi_services_override)
            if pmi_services_override is not None
            else fetch_pmi_services()
        ),
    }


# ---------------------------------------------------------------------------
# Source metadata builder
# ---------------------------------------------------------------------------

def _build_source_meta(raw: dict[str, FetchResult], statuses: dict[str, str]) -> dict[str, SourceMeta]:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta: dict[str, SourceMeta] = {}
    for key, result in raw.items():
        is_valuation = key == "forward_pe"
        # For the valuation metric, the actual provider may differ from result.provider
        # when extra["provider"] carries more specific context (e.g. "fmp" vs "FMP" label).
        provider_label = result.provider
        if is_valuation and result.extra.get("provider"):
            provider_label = result.extra["provider"].upper() if result.extra["provider"] in ("fmp", "yahoo") else result.extra["provider"]
            # Normalize to full labels used elsewhere
            if provider_label == "FMP":
                provider_label = "FMP"
            elif provider_label == "YAHOO":
                provider_label = "Yahoo Finance"
        meta[key] = SourceMeta(
            provider=provider_label,
            series_name=result.series_name,
            series_id=result.series_id,
            fetched_at=now,
            observed_at=result.observed_at,
            frequency=result.frequency,
            status=statuses.get(key, result.status),
            note=result.note,
            basis=result.extra.get("pe_basis") if is_valuation else None,
        )
    return meta


# ---------------------------------------------------------------------------
# Core fetch-and-assemble logic (synchronous, runs in thread)
# ---------------------------------------------------------------------------

def _run_full_fetch(
    pmi_manufacturing_override: float | None = None,
    pmi_services_override: float | None = None,
    force_refresh: bool = False,
) -> dict[str, FetchResult]:
    api_key = settings.fred_api_key
    timeout = settings.http_timeout_seconds

    fred = _fetch_fred_all(api_key, timeout)
    yahoo = _fetch_yahoo_all(timeout)
    cpi = _fetch_cpi_all(api_key, timeout)
    pmi = _fetch_pmi_all(
        pmi_manufacturing_override=pmi_manufacturing_override,
        pmi_services_override=pmi_services_override,
    )
    valuation = {"forward_pe": _fetch_valuation(timeout, force_refresh=force_refresh)}

    return {**fred, **yahoo, **cpi, **pmi, **valuation}


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------

async def get_live_snapshot(
    force_refresh: bool = False,
    pmi_manufacturing_override: float | None = None,
    pmi_services_override: float | None = None,
) -> LiveSnapshotResponse:
    """
    Fetch, normalize, and cache the live IndicatorSnapshot.
    Returns a LiveSnapshotResponse with full source provenance.
    """
    has_pmi_overrides = pmi_manufacturing_override is not None or pmi_services_override is not None
    cache = get_snapshot_cache()

    if not force_refresh and not has_pmi_overrides:
        cached = cache.get()
        if cached is not None:
            if cached.get("_cache_version") != _LIVE_SNAPSHOT_CACHE_VERSION:
                logger.info(
                    "Invalidating live snapshot cache (version %s → %s)",
                    cached.get("_cache_version"),
                    _LIVE_SNAPSHOT_CACHE_VERSION,
                )
                cache.clear()
            else:
                logger.debug("Serving live snapshot from cache")
                return LiveSnapshotResponse(**_strip_internal_cache_keys(cached))

    # --- run provider fetches in thread pool ---
    try:
        raw = await asyncio.to_thread(
            _run_full_fetch,
            pmi_manufacturing_override,
            pmi_services_override,
            force_refresh,
        )
    except Exception as exc:
        logger.error("Full fetch failed: %s", exc)
        # Try stale cache
        if settings.live_allow_stale_cache and not has_pmi_overrides:
            disk = cache.load_from_disk()
            if disk and disk.get("_cache_version") == _LIVE_SNAPSHOT_CACHE_VERSION:
                logger.warning("Serving stale disk-cached snapshot after fetch failure")
                return LiveSnapshotResponse(**_strip_internal_cache_keys(disk))
            if disk:
                logger.warning("Ignoring on-disk snapshot (cache version mismatch or missing)")
        raise

    # --- compute per-metric freshness ---
    statuses = {k: classify_result_freshness(k, v) for k, v in raw.items()}
    overall_status, stale_series = compute_overall_freshness(statuses)

    # --- normalize to IndicatorSnapshot ---
    snapshot = build_indicator_snapshot(
        raw=raw,
        freshness_statuses=statuses,
        stale_series=stale_series,
        overall_status=overall_status,
        fed_put=settings.default_fed_put,
        treasury_put=settings.default_treasury_put,
        political_put=settings.default_political_put,
    )

    source_meta = _build_source_meta(raw, statuses)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    response = LiveSnapshotResponse(
        snapshot=snapshot,
        sources=source_meta,
        overall_status=overall_status,
        stale_series=stale_series,
        generated_at=generated_at,
    )

    # --- persist to cache (only when doctrine-critical fields are present) ---
    missing_critical = _missing_cache_critical_fields(snapshot)
    if has_pmi_overrides:
        logger.info("Skipping live snapshot cache write: request used PMI overrides")
    elif missing_critical:
        logger.warning(
            "Skipping live snapshot cache write: missing critical fields (%s)",
            ", ".join(missing_critical),
        )
    else:
        payload = response.model_dump()
        payload["_cache_version"] = _LIVE_SNAPSHOT_CACHE_VERSION
        cache.set(payload)
        cache.save_to_disk(payload)

    return response


async def get_live_playbook(
    force_refresh: bool = False,
    pmi_manufacturing_override: float | None = None,
    pmi_services_override: float | None = None,
) -> LivePlaybookResponse:
    """
    Full live pipeline: providers → snapshot → rule engine → LLM summary → catalysts.
    """
    snapshot_resp = await get_live_snapshot(
        force_refresh=force_refresh,
        pmi_manufacturing_override=pmi_manufacturing_override,
        pmi_services_override=pmi_services_override,
    )
    snapshot = snapshot_resp.snapshot

    state, playbook_conclusion = build_dashboard_state_with_conclusion(snapshot)
    summary = await generate_summary(state, conclusion=playbook_conclusion)

    catalyst_config = load_catalyst_config()
    catalysts = build_catalyst_state(catalyst_config, snapshot, state)

    return LivePlaybookResponse(
        snapshot=snapshot,
        state=state,
        playbook_conclusion=playbook_conclusion,
        summary=summary,
        catalysts=catalysts,
        sources=snapshot_resp.sources,
        overall_status=snapshot_resp.overall_status,
        stale_series=snapshot_resp.stale_series,
        generated_at=snapshot_resp.generated_at,
    )
