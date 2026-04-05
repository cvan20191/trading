"""
Freshness classification — determines whether each metric and the overall
snapshot are fresh, stale, or missing.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.services.ingestion.series_map import FRESHNESS_RULES
from app.services.providers.base import FetchResult


def _days_since(observed_at: str | None) -> float | None:
    """Return calendar days between observed_at and now, or None if unparseable."""
    if not observed_at:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(observed_at, fmt).replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - dt).total_seconds() / 86400
        except ValueError:
            continue
    return None


def classify_result_freshness(
    key: str,
    result: FetchResult,
) -> str:
    """
    Return 'fresh', 'stale', 'missing', or 'error' for a single FetchResult.
    """
    if result.status in ("missing", "error"):
        return result.status

    allowed_days = FRESHNESS_RULES.get(key)
    age = _days_since(result.observed_at)

    if age is None or allowed_days is None:
        return result.status   # trust whatever the provider reported

    if age > allowed_days:
        return "stale"

    return "fresh"


def compute_overall_freshness(
    statuses: dict[str, str],
    core_keys: tuple[str, ...] = (
        "fed_funds_rate",
        "balance_sheet",
        "unemployment_rate",
        "core_cpi",
        "yield_curve",
    ),
) -> tuple[str, list[str]]:
    """
    Compute overall_status and stale_series list from per-metric status dict.

    - fresh  → all core metrics fresh, few secondary misses
    - mixed  → some stale/missing but app still useful
    - stale  → too many core metrics degraded
    """
    stale_series = [k for k, v in statuses.items() if v in ("stale", "missing", "error")]

    core_bad = [k for k in core_keys if statuses.get(k) in ("stale", "missing", "error")]
    total = len(statuses)
    bad_count = len(stale_series)

    if not core_bad and bad_count <= 2:
        return "fresh", stale_series
    if len(core_bad) <= 2 and bad_count <= total // 2:
        return "mixed", stale_series
    return "stale", stale_series
