"""
CPI provider — fetches core CPI and subcomponent series from FRED.

Subcomponent statuses (sticky / cooling / rising / unknown) are derived
from a short trend window rather than a single point.
"""

from __future__ import annotations

import logging

from app.services.ingestion.series_map import FRED_SERIES, PROVIDER_FRED, PROVIDER_STUB
from app.services.providers.base import FetchResult, ProviderError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Status thresholds (YoY % change to label subcomponent trend)
# ---------------------------------------------------------------------------
_STICKY_THRESHOLD = 3.5     # YoY% above which a subcomponent is "sticky"
_RISING_DELTA = 0.1         # MoM sequential change to call "rising"
_COOLING_DELTA = -0.05      # MoM sequential change to call "cooling"


def _label_status(yoy: float | None, mom_delta: float | None) -> str:
    """Return sticky/rising/cooling/unknown from subcomponent values."""
    if yoy is None:
        return "unknown"
    if yoy >= _STICKY_THRESHOLD:
        return "sticky"
    if mom_delta is not None and mom_delta >= _RISING_DELTA:
        return "rising"
    if mom_delta is not None and mom_delta <= _COOLING_DELTA:
        return "cooling"
    return "sticky" if yoy >= 3.0 else "cooling"


def fetch_cpi_components(
    api_key: str = "",
    timeout: int = 20,
) -> dict[str, FetchResult]:
    """
    Fetch core CPI, shelter CPI, and services-less-energy CPI from FRED.

    Returns a dict keyed by metric name.
    Each failed fetch is returned as a missing FetchResult rather than raising.
    """
    from app.services.providers.fred_client import fetch_series

    results: dict[str, FetchResult] = {}

    for key in ("core_cpi", "shelter_cpi", "services_ex_energy"):
        series_id = FRED_SERIES[key]
        try:
            results[key] = fetch_series(
                series_id=series_id,
                series_name=key,
                frequency="monthly",
                observation_window_days=400,   # ~13 months for YoY calc
                api_key=api_key,
                timeout=timeout,
            )
        except ProviderError as exc:
            logger.warning("CPI provider error for %s: %s", key, exc)
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


def compute_yoy_and_status(
    result: FetchResult,
) -> tuple[float | None, float | None, str]:
    """
    Given a FetchResult for a CPI index, compute:
    - YoY % change (using value ~12 observations ago)
    - MoM change (latest vs prior)
    - status label

    CPI series are index levels, so YoY = (latest / year_ago - 1) * 100.
    """
    series = result.series
    if not series or result.value is None:
        return None, None, "unknown"

    latest = result.value

    yoy: float | None = None
    if len(series) >= 12:
        year_ago = series[-13][1] if len(series) >= 13 else series[0][1]
        if year_ago and year_ago != 0:
            yoy = round((latest / year_ago - 1) * 100, 2)

    mom_delta: float | None = None
    if len(series) >= 2:
        prior = series[-2][1]
        if prior and prior != 0:
            mom_delta = round((latest / prior - 1) * 100, 3)

    status = _label_status(yoy, mom_delta)
    return yoy, mom_delta, status
