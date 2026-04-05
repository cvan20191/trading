"""
PMI provider abstraction.

PMI data (ISM Manufacturing / Services) is not cleanly available from free APIs.
This module:
  1. Attempts to use an environment-configurable stub value if set
  2. Returns explicit "missing" status if no source is available
  3. Exposes a configurable override path for future integration

The architecture is in place to swap in a real data source later
(e.g., a paid API, a manually entered value, or a scraped BLS proxy)
without changing the normalizer.
"""

from __future__ import annotations

import logging
import os

from app.services.ingestion.series_map import PROVIDER_STUB
from app.services.providers.base import FetchResult

logger = logging.getLogger(__name__)

_MISSING_RESULT = FetchResult(
    value=None,
    observed_at=None,
    series=[],
    provider=PROVIDER_STUB,
    series_id="n/a",
    series_name="pmi",
    frequency="monthly",
    status="missing",
    note=(
        "PMI (ISM Manufacturing / Services) is not available from a free public API. "
        "Set PMI_MANUFACTURING or PMI_SERVICES env vars to inject manual values."
    ),
)


def _read_env_float(key: str) -> float | None:
    val = os.environ.get(key, "").strip()
    if val:
        try:
            return float(val)
        except ValueError:
            logger.warning("Invalid float for env var %s: %r", key, val)
    return None


def fetch_pmi_manufacturing() -> FetchResult:
    val = _read_env_float("PMI_MANUFACTURING")
    if val is not None:
        return FetchResult(
            value=val,
            observed_at=None,
            series=[],
            provider=PROVIDER_STUB,
            series_id="PMI_MANUFACTURING",
            series_name="pmi_manufacturing",
            frequency="monthly",
            status="fallback",
            note="Value injected from PMI_MANUFACTURING environment variable",
        )
    return _MISSING_RESULT


def fetch_pmi_services() -> FetchResult:
    val = _read_env_float("PMI_SERVICES")
    if val is not None:
        return FetchResult(
            value=val,
            observed_at=None,
            series=[],
            provider=PROVIDER_STUB,
            series_id="PMI_SERVICES",
            series_name="pmi_services",
            frequency="monthly",
            status="fallback",
            note="Value injected from PMI_SERVICES environment variable",
        )
    return _MISSING_RESULT
