"""
Replay outcome computation — forward price returns for SPY, QQQ, and WTI
at 1W, 1M, and 3M after the selected replay date.

Field names match the actual tickers:
  spy_*  — SPDR S&P 500 ETF (SPY)
  qqq_*  — Invesco Nasdaq-100 ETF (QQQ)
  wti_*  — WTI Crude front-month futures (CL=F)
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from app.schemas.replay import OutcomePoint, OutcomeReview

logger = logging.getLogger(__name__)

_TICKERS = {
    "spy": "SPY",
    "qqq": "QQQ",
    "wti": "CL=F",
}

# Horizon offsets
_HORIZONS = {
    "1w": 7,
    "1m": 30,
    "3m": 91,
}


def _import_yfinance():
    try:
        import yfinance as yf  # type: ignore[import]
        return yf
    except ImportError as exc:
        raise RuntimeError("yfinance is not installed") from exc


def _find_price_on_or_after(prices: list[tuple[str, float]], target_date: date) -> tuple[str, float] | None:
    """Return the first (date, price) entry on or after target_date."""
    target_str = target_date.isoformat()
    for d, p in prices:
        if d >= target_str:
            return d, p
    return None


def _compute_return(start: float | None, end: float | None) -> float | None:
    if start is None or end is None or start == 0:
        return None
    return round((end / start - 1) * 100, 2)


def _build_outcome_point(
    prices: dict[str, list[tuple[str, float]]],
    as_of_date: date,
    horizon_days: int,
) -> OutcomePoint:
    """
    Build one OutcomePoint for the given horizon.
    prices dict has keys: spy, qqq, wti — each is a list of (date_str, price).
    """
    today = date.today()
    horizon_date = as_of_date + timedelta(days=horizon_days)

    if horizon_date >= today:
        # Horizon not yet reached
        return OutcomePoint(
            date_end=horizon_date.isoformat(),
            spy_return_pct=None, spy_price_start=None, spy_price_end=None,
            qqq_return_pct=None, qqq_price_start=None, qqq_price_end=None,
            wti_change_pct=None, wti_price_start=None, wti_price_end=None,
        )

    outcomes: dict = {"date_end": horizon_date.isoformat()}

    for key in ("spy", "qqq", "wti"):
        series = prices.get(key, [])
        if not series:
            outcomes.update({
                f"{key}_{'change' if key == 'wti' else 'return'}_pct": None,
                f"{key}_price_start": None,
                f"{key}_price_end": None,
            })
            continue

        # Find the starting price (closest trading day on or after as_of_date)
        start_pair = _find_price_on_or_after(series, as_of_date)
        # Find the ending price (closest trading day on or after horizon_date)
        end_pair = _find_price_on_or_after(series, horizon_date)

        start_price = start_pair[1] if start_pair else None
        end_price = end_pair[1] if end_pair else None

        ret = _compute_return(start_price, end_price)
        ret_key = "wti_change_pct" if key == "wti" else f"{key}_return_pct"
        outcomes[ret_key] = ret
        outcomes[f"{key}_price_start"] = start_price
        outcomes[f"{key}_price_end"] = end_price

    return OutcomePoint(**outcomes)


async def compute_outcomes(as_of_date: date) -> OutcomeReview:
    """
    Fetch SPY, QQQ, and WTI price history spanning as_of_date through +91 days,
    then compute 1W, 1M, and 3M forward returns.

    Returns an OutcomeReview. Individual fields are None when the horizon is
    in the future or price data is unavailable.
    """
    import asyncio

    end_date = min(as_of_date + timedelta(days=100), date.today())
    start_date = as_of_date  # we only need forward data

    def _fetch_prices() -> dict[str, list[tuple[str, float]]]:
        yf = _import_yfinance()
        result: dict[str, list[tuple[str, float]]] = {}
        for key, ticker in _TICKERS.items():
            try:
                df = yf.download(
                    ticker,
                    start=start_date.strftime("%Y-%m-%d"),
                    end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
                    progress=False,
                    auto_adjust=True,
                    timeout=20,
                )
                if df.empty:
                    result[key] = []
                    continue
                close = df["Close"].squeeze()
                result[key] = [
                    (str(idx.date()), float(val))
                    for idx, val in zip(close.index, close.values)
                    if val is not None
                ]
            except Exception as exc:
                logger.warning("Outcomes fetch failed for %s: %s", ticker, exc)
                result[key] = []
        return result

    prices = await asyncio.to_thread(_fetch_prices)

    outcomes_1w = _build_outcome_point(prices, as_of_date, _HORIZONS["1w"])
    outcomes_1m = _build_outcome_point(prices, as_of_date, _HORIZONS["1m"])
    outcomes_3m = _build_outcome_point(prices, as_of_date, _HORIZONS["3m"])

    today = date.today()
    if as_of_date + timedelta(days=_HORIZONS["3m"]) >= today:
        data_note = "Some outcome horizons are not yet available — date too recent."
    else:
        data_note = "Outcomes show SPY (S&P 500 ETF), QQQ (Nasdaq-100 ETF), and WTI crude returns."

    return OutcomeReview(
        as_of=as_of_date.isoformat(),
        outcomes_1w=outcomes_1w,
        outcomes_1m=outcomes_1m,
        outcomes_3m=outcomes_3m,
        data_note=data_note,
    )
