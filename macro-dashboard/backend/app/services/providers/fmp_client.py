"""
Financial Modeling Prep (FMP) provider — used exclusively for the Mag 7 Forward P/E basket.

Endpoints used:
  GET /stable/profile?symbol=AAPL,MSFT,...  — market cap, price, shares outstanding
  GET /stable/analyst-estimates?symbol=X&period=annual&limit=4  — forward EPS consensus

Field-name adapter:
  FMP field names can vary across endpoint versions.  All lookups go through
  helper functions that try multiple candidate keys in priority order so the
  basket computation does not break on minor API changes.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

import httpx

from app.services.ingestion.series_map import PROVIDER_FMP
from app.services.providers.base import FetchResult, ProviderError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FMP_BASE = "https://financialmodelingprep.com/stable"

MAG7_TICKERS: list[str] = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]

MIN_CONSTITUENTS = 5          # require at least 5 valid members
MIN_COVERAGE_RATIO = 0.80     # require ≥80 % market-cap coverage of total Mag 7

# ---------------------------------------------------------------------------
# Internal helpers — field-name adapter
# ---------------------------------------------------------------------------

def _get_market_cap(profile: dict) -> float | None:
    """Try multiple FMP field names for market cap, fall back to price × shares."""
    for key in ("mktCap", "marketCap", "market_cap"):
        v = profile.get(key)
        if _positive_finite(v):
            return float(v)
    # derived fallback
    price = _get_price(profile)
    shares = _get_shares(profile)
    if price is not None and shares is not None:
        return price * shares
    return None


def _get_shares(profile: dict) -> float | None:
    for key in ("sharesOutstanding", "outstandingShares", "shares", "shareOutstanding"):
        v = profile.get(key)
        if _positive_finite(v):
            return float(v)
    # Derived fallback: shares = marketCap / price
    mktcap = None
    for key in ("mktCap", "marketCap", "market_cap"):
        v = profile.get(key)
        if _positive_finite(v):
            mktcap = float(v)
            break
    price = _get_price(profile)
    if mktcap is not None and price is not None and price > 0:
        return mktcap / price
    return None


def _get_price(profile: dict) -> float | None:
    for key in ("price", "stockPrice", "currentPrice", "lastAnnualDividend"):
        v = profile.get(key)
        if _positive_finite(v):
            return float(v)
    return None


def _get_forward_eps(estimate_row: dict) -> float | None:
    """Try multiple FMP field names for consensus forward EPS in an analyst-estimate row."""
    for key in ("epsAvg", "estimatedEpsAvg", "estimatedEpsMean", "epsMean",
                "epsEstimated", "estimatedEps", "consensusEps"):
        v = estimate_row.get(key)
        if _positive_finite(v):
            return float(v)
    return None


def _get_estimate_date(estimate_row: dict) -> datetime | None:
    for key in ("date", "period", "fiscalDateEnding", "fiscalYear"):
        v = estimate_row.get(key)
        if isinstance(v, str) and v:
            try:
                # Accept "YYYY-MM-DD" or "YYYY" formats
                if len(v) == 4:
                    return datetime(int(v), 12, 31, tzinfo=timezone.utc)
                return datetime.strptime(v[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    return None


def _positive_finite(x: object) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x)) and float(x) > 0


# ---------------------------------------------------------------------------
# FMP HTTP helpers
# ---------------------------------------------------------------------------

def _fmp_get(path: str, params: dict, timeout: int) -> list | dict:
    """Single GET call to the FMP stable API; raises ProviderError on failure."""
    try:
        resp = httpx.get(
            f"{FMP_BASE}/{path}",
            params=params,
            timeout=timeout,
        )
    except Exception as exc:
        raise ProviderError(f"FMP HTTP error on {path}: {exc}") from exc

    if resp.status_code == 401:
        raise ProviderError("FMP API key rejected (HTTP 401) — check FMP_API_KEY in .env")
    if resp.status_code == 403:
        raise ProviderError("FMP API key forbidden (HTTP 403) — endpoint may require a paid plan")
    if not resp.is_success:
        raise ProviderError(f"FMP {path} returned HTTP {resp.status_code}: {resp.text[:200]}")

    try:
        return resp.json()
    except Exception as exc:
        raise ProviderError(f"FMP {path} returned non-JSON: {exc}") from exc


# ---------------------------------------------------------------------------
# EPS estimate selector
# ---------------------------------------------------------------------------

def pick_forward_eps_estimate(estimates: list[dict]) -> float | None:
    """
    Select the best forward EPS from a list of annual analyst-estimate rows.

    Rules:
    1. Keep only rows whose date is strictly in the future.
    2. Among those, prefer the nearest future fiscal year annual consensus.
    3. The chosen EPS must be positive and finite.
    4. Returns None if no clearly forward-looking row qualifies.
    """
    now = datetime.now(timezone.utc)
    candidates: list[tuple[datetime, float]] = []

    for row in estimates:
        row_date = _get_estimate_date(row)
        if row_date is None:
            continue
        if row_date <= now:
            continue
        eps = _get_forward_eps(row)
        if eps is None:
            continue
        candidates.append((row_date, eps))

    if not candidates:
        return None

    # Pick the nearest future fiscal year
    candidates.sort(key=lambda t: t[0])
    return candidates[0][1]


# ---------------------------------------------------------------------------
# FMP data fetchers
# ---------------------------------------------------------------------------

def fetch_profiles_batch(tickers: list[str], api_key: str, timeout: int) -> dict[str, dict]:
    """
    Fetch company profiles for multiple tickers via individual FMP calls.
    The /stable/profile endpoint does not support comma-separated batch requests.
    Returns {ticker: profile_dict}.
    """
    result: dict[str, dict] = {}
    for ticker in tickers:
        try:
            data = _fmp_get("profile", {"symbol": ticker, "apikey": api_key}, timeout)
        except ProviderError as exc:
            logger.warning("FMP profile fetch failed for %s: %s", ticker, exc)
            continue

        if not isinstance(data, list) or not data:
            logger.debug("FMP profile returned empty/unexpected for %s", ticker)
            continue

        result[ticker.upper()] = data[0]

    return result


def fetch_analyst_estimates(ticker: str, api_key: str, timeout: int) -> list[dict]:
    """
    Fetch annual analyst EPS estimates for a single ticker.
    Returns list of estimate rows (most recent first from FMP).
    """
    try:
        data = _fmp_get(
            "analyst-estimates",
            {"symbol": ticker, "period": "annual", "limit": "4", "apikey": api_key},
            timeout,
        )
    except ProviderError:
        raise

    if not isinstance(data, list):
        return []
    return data


# ---------------------------------------------------------------------------
# Basket computation
# ---------------------------------------------------------------------------

def compute_mag7_basket(
    api_key: str,
    timeout: int = 20,
    tickers: list[str] | None = None,
) -> FetchResult:
    """
    Compute the Mag 7 market-cap-weighted forward P/E basket using FMP data.

    Returns a FetchResult with extra dict containing:
      pe_basis, metric_name, object_label, provider, coverage_count, coverage_ratio

    Raises ProviderError if FMP calls fail hard (caller handles fallback).
    Returns a FetchResult with status="missing" if coverage is insufficient.
    """
    mag7 = tickers or MAG7_TICKERS
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # --- 1. Batch profile fetch ---
    try:
        profiles = fetch_profiles_batch(mag7, api_key, timeout)
    except ProviderError as exc:
        raise ProviderError(f"FMP profile batch failed: {exc}") from exc

    # --- 2. Analyst estimates (sequential per ticker) ---
    total_market_cap = 0.0
    sum_market_cap_valid = 0.0
    sum_forward_earnings = 0.0
    valid_count = 0
    skipped: list[str] = []
    # Per-ticker detail for UI display
    constituents: list[dict] = []

    for ticker in mag7:
        profile = profiles.get(ticker, {})
        mktcap = _get_market_cap(profile)
        shares = _get_shares(profile)
        price = _get_price(profile)

        if mktcap is not None:
            total_market_cap += mktcap

        if mktcap is None or shares is None:
            logger.debug("FMP Mag7: skipping %s — no market cap or shares", ticker)
            skipped.append(f"{ticker}(no_mktcap_or_shares)")
            constituents.append({"ticker": ticker, "price": None, "forward_eps": None, "forward_pe": None})
            continue

        try:
            estimates = fetch_analyst_estimates(ticker, api_key, timeout)
        except ProviderError as exc:
            logger.warning("FMP analyst-estimates failed for %s: %s", ticker, exc)
            skipped.append(f"{ticker}(estimates_error)")
            constituents.append({"ticker": ticker, "price": price, "forward_eps": None, "forward_pe": None})
            continue

        fwd_eps = pick_forward_eps_estimate(estimates)
        if fwd_eps is None:
            logger.debug("FMP Mag7: skipping %s — no forward EPS estimate", ticker)
            skipped.append(f"{ticker}(no_forward_eps)")
            constituents.append({"ticker": ticker, "price": price, "forward_eps": None, "forward_pe": None})
            continue

        forward_earnings = fwd_eps * shares
        if not _positive_finite(forward_earnings):
            skipped.append(f"{ticker}(non_positive_earnings)")
            constituents.append({"ticker": ticker, "price": price, "forward_eps": fwd_eps, "forward_pe": None})
            continue

        fwd_pe = round(price / fwd_eps, 2) if price is not None and _positive_finite(price / fwd_eps) else None
        sum_market_cap_valid += mktcap
        sum_forward_earnings += forward_earnings
        valid_count += 1
        constituents.append({"ticker": ticker, "price": price, "forward_eps": round(fwd_eps, 2), "forward_pe": fwd_pe})
        logger.debug("FMP Mag7: %s — mktcap=%.2fB fwd_eps=%.2f pe=%.1f", ticker, mktcap / 1e9, fwd_eps, fwd_pe or 0)

    # --- 3. Coverage check ---
    coverage_ratio = (
        sum_market_cap_valid / total_market_cap
        if total_market_cap > 0
        else 0.0
    )

    if valid_count < MIN_CONSTITUENTS or coverage_ratio < MIN_COVERAGE_RATIO:
        note = (
            f"Mag 7 basket coverage insufficient: {valid_count}/{len(mag7)} constituents, "
            f"{coverage_ratio:.0%} market-cap coverage. Skipped: {', '.join(skipped) or 'none'}. "
            "Falling back to Yahoo proxy."
        )
        logger.warning("FMP Mag7 basket: %s", note)
        return FetchResult(
            value=None,
            observed_at=None,
            series=[],
            provider=PROVIDER_FMP,
            series_id="MAG7_BASKET",
            series_name="Mag 7 Forward P/E Basket",
            frequency="daily",
            status="missing",
            note=note,
            extra={
                "pe_basis": "unavailable",
                "metric_name": "Mag 7 Forward P/E",
                "object_label": "Mag 7 Basket",
                "provider": "fmp",
                "coverage_count": valid_count,
                "coverage_ratio": round(coverage_ratio, 4),
                "constituents": constituents,
            },
        )

    # --- 4. Compute basket P/E ---
    basket_pe = sum_market_cap_valid / sum_forward_earnings

    if not _positive_finite(basket_pe):
        raise ProviderError("Mag 7 basket P/E computation produced a non-finite result")

    note = (
        f"Mag 7 market-cap-weighted forward P/E — {valid_count}/{len(mag7)} constituents, "
        f"{coverage_ratio:.0%} market-cap coverage (FMP analyst estimates)"
    )
    if skipped:
        note += f". Excluded: {', '.join(skipped)}"

    logger.info("FMP Mag7 basket P/E = %.2f (%d constituents, %.0f%% coverage)",
                basket_pe, valid_count, coverage_ratio * 100)

    return FetchResult(
        value=basket_pe,
        observed_at=today,
        series=[],
        provider=PROVIDER_FMP,
        series_id="MAG7_BASKET",
        series_name="Mag 7 Forward P/E Basket",
        frequency="daily",
        status="fresh",
        note=note,
        extra={
            "pe_basis": "forward",
            "metric_name": "Mag 7 Forward P/E",
            "object_label": "Mag 7 Basket",
            "provider": "fmp",
            "coverage_count": valid_count,
            "coverage_ratio": round(coverage_ratio, 4),
            "constituents": constituents,
        },
    )
