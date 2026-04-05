"""
Source metadata — tracks origin, freshness, and status for each fetched metric.
"""

from __future__ import annotations

from pydantic import BaseModel


class SourceMeta(BaseModel):
    provider: str
    series_name: str
    series_id: str | None = None
    fetched_at: str
    observed_at: str | None = None
    frequency: str | None = None
    status: str = "unknown"   # fresh | stale | missing | error | fallback
    note: str | None = None
    # Valuation-specific: forward | trailing | ttm_derived | unavailable | None (non-valuation metrics)
    basis: str | None = None
