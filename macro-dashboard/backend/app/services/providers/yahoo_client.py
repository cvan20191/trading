"""
Yahoo Finance provider — fetches market data via yfinance.

yfinance is already in the existing 3StateMachineEngine requirements.
We add it to macro-dashboard/requirements.txt as well.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta

from app.services.ingestion.series_map import PROVIDER_YAHOO
from app.services.providers.base import FetchResult, ProviderError

logger = logging.getLogger(__name__)


def _import_yfinance():
    try:
        import yfinance as yf  # type: ignore[import]
        return yf
    except ImportError as exc:
        raise ProviderError("yfinance is not installed") from exc


def fetch_ticker(
    ticker: str,
    series_name: str,
    frequency: str = "daily",
    window_days: int = 90,
    timeout: int = 20,
) -> FetchResult:
    """
    Fetch latest close price and a recent price window for a Yahoo ticker.
    """
    yf = _import_yfinance()

    try:
        end = datetime.utcnow()
        start = end - timedelta(days=window_days)
        df = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
            timeout=timeout,
        )
    except Exception as exc:
        raise ProviderError(f"yfinance download failed for {ticker}: {exc}") from exc

    if df.empty:
        return FetchResult(
            value=None,
            observed_at=None,
            series=[],
            provider=PROVIDER_YAHOO,
            series_id=ticker,
            series_name=series_name,
            frequency=frequency,
            status="missing",
            note=f"No data returned by Yahoo Finance for {ticker}",
        )

    # yfinance >=1.x returns multi-level columns even for single tickers;
    # .squeeze() converts a single-column DataFrame to a plain Series.
    close = df["Close"].squeeze()
    # Build (date_str, price) pairs in chronological order
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
            frequency=frequency,
            status="missing",
            note=f"Close prices were all null for {ticker}",
        )
    latest_date_str, latest_value = series[-1]

    return FetchResult(
        value=latest_value,
        observed_at=latest_date_str,
        series=series,
        provider=PROVIDER_YAHOO,
        series_id=ticker,
        series_name=series_name,
        frequency=frequency,
        status="fresh",
    )


def _positive_finite(x: object) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x)) and float(x) > 0


def fetch_pe_ratio_proxy(
    ticker: str,
    series_name: str,
) -> FetchResult:
    """
    Forward or trailing P/E for an ETF / index proxy (e.g. QQQ for big-tech basket).

    Yahoo often omits ``forwardPE`` for ETFs; in that case we fall back to
    ``trailingPE``, then price / TTM EPS from the same quote payload.
    """
    yf = _import_yfinance()
    today = datetime.utcnow().strftime("%Y-%m-%d")

    try:
        t = yf.Ticker(ticker)
        # .info hits Yahoo quote summary; can be slower than download()
        info = t.info
    except Exception as exc:
        raise ProviderError(f"yfinance info failed for {ticker}: {exc}") from exc

    if not isinstance(info, dict):
        info = {}

    forward = info.get("forwardPE")
    trailing = info.get("trailingPE")

    pe: float | None = None
    basis = "unavailable"
    note = ""

    if _positive_finite(forward):
        pe = float(forward)
        basis = "forward"
        note = f"Forward P/E (Yahoo {ticker}) — QQQ proxy; Mag 7 basket preferred"
    elif _positive_finite(trailing):
        pe = float(trailing)
        basis = "trailing"
        note = (
            f"Trailing P/E proxy (Yahoo {ticker}) — forward P/E not available for this ETF; "
            "treat as a directional signal vs the speaker's 20× / 30× zones, not as an exact trigger."
        )
    else:
        # Last resort: price / TTM EPS (Yahoo still returns these for many ETFs)
        price = (
            info.get("regularMarketPrice")
            or info.get("currentPrice")
            or info.get("regularMarketPreviousClose")
            or info.get("previousClose")
        )
        eps = info.get("epsTrailingTwelveMonths") or info.get("trailingEps")
        if _positive_finite(price) and _positive_finite(eps):
            pe = float(price) / float(eps)
            basis = "ttm_derived"
            note = (
                f"Derived P/E from price ÷ TTM EPS (Yahoo {ticker}) — "
                "weakest proxy; treat as directional only vs the speaker's zones."
            )

    _yahoo_extra = {
        "pe_basis": basis,
        "metric_name": "QQQ P/E Proxy",
        "object_label": f"QQQ ({ticker})",
        "provider": "yahoo",
        "coverage_count": None,
        "coverage_ratio": None,
    }

    if pe is None:
        return FetchResult(
            value=None,
            observed_at=None,
            series=[],
            provider=PROVIDER_YAHOO,
            series_id=ticker,
            series_name=series_name,
            frequency="quote",
            status="missing",
            note=f"No forward or trailing P/E returned for {ticker} — Mag 7 basket unavailable",
            extra={**_yahoo_extra, "pe_basis": "unavailable"},
        )

    return FetchResult(
        value=pe,
        observed_at=today,
        series=[],
        provider=PROVIDER_YAHOO,
        series_id=ticker,
        series_name=series_name,
        frequency="quote",
        status="fresh",
        note=note,
        extra=_yahoo_extra,
    )
