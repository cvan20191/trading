"""
FRED provider — fetches macro series from the FRED REST API.

Public access works without an API key (rate-limited).
Set FRED_API_KEY in .env for higher limits.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import httpx

from app.services.ingestion.series_map import PROVIDER_FRED
from app.services.providers.base import FetchResult, ProviderError

logger = logging.getLogger(__name__)

_BASE = "https://api.stlouisfed.org/fred"
_OBS_URL = f"{_BASE}/series/observations"


def _api_key_param(api_key: str) -> dict[str, str]:
    if api_key:
        return {"api_key": api_key}
    return {}


def fetch_series(
    series_id: str,
    series_name: str,
    frequency: str = "unknown",
    observation_window_days: int = 90,
    api_key: str = "",
    timeout: int = 20,
    observation_end_date: str | None = None,
) -> FetchResult:
    """
    Fetch observations for a FRED series.

    Returns a FetchResult with the latest value and a short recent window.
    Raises ProviderError on network or API failure.

    observation_end_date: optional ISO date string (YYYY-MM-DD). When provided,
    FRED will only return observations up to (and including) that date. Used
    by the Replay Lab to freeze data to a historical as_of date.

    Note: FRED requires a free API key (https://fred.stlouisfed.org/docs/api/api_key.html).
    Set FRED_API_KEY in your .env file. Without a key, all FRED fetches will fail with 400.
    """
    if not api_key:
        return FetchResult(
            value=None,
            observed_at=None,
            series=[],
            provider=PROVIDER_FRED,
            series_id=series_id,
            series_name=series_name,
            frequency=frequency,
            status="missing",
            note=(
                "FRED_API_KEY not set. Get a free key at "
                "https://fred.stlouisfed.org/docs/api/api_key.html "
                "and add it to your .env file."
            ),
        )

    if observation_end_date:
        # Historical replay: window is relative to as_of date
        from datetime import date as _date
        end_dt = _date.fromisoformat(observation_end_date)
        start_dt = end_dt - timedelta(days=observation_window_days)
        start_date = start_dt.strftime("%Y-%m-%d")
    else:
        start_date = (datetime.utcnow() - timedelta(days=observation_window_days)).strftime("%Y-%m-%d")

    params: dict[str, str] = {
        "series_id": series_id,
        "observation_start": start_date,
        "sort_order": "desc",
        "file_type": "json",
        **_api_key_param(api_key),
    }
    if observation_end_date:
        params["observation_end"] = observation_end_date

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(_OBS_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        raise ProviderError(f"FRED HTTP {exc.response.status_code} for {series_id}") from exc
    except httpx.RequestError as exc:
        raise ProviderError(f"FRED request failed for {series_id}: {exc}") from exc

    observations = data.get("observations", [])
    # Filter out missing values (".")
    valid = [
        (obs["date"], float(obs["value"]))
        for obs in observations
        if obs.get("value", ".") != "."
    ]

    if not valid:
        return FetchResult(
            value=None,
            observed_at=None,
            series=[],
            provider=PROVIDER_FRED,
            series_id=series_id,
            series_name=series_name,
            frequency=frequency,
            status="missing",
            note="No valid observations returned by FRED",
        )

    # valid is desc order (newest first)
    latest_date, latest_value = valid[0]
    # Reverse for chronological order for trend computation
    recent_series = list(reversed(valid))

    return FetchResult(
        value=latest_value,
        observed_at=latest_date,
        series=recent_series,
        provider=PROVIDER_FRED,
        series_id=series_id,
        series_name=series_name,
        frequency=frequency,
        status="fresh",
    )
