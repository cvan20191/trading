"""
Provider base helpers — shared data containers and exceptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class ProviderError(Exception):
    """Raised when a provider cannot return usable data."""


@dataclass
class FetchResult:
    """Holds a single fetched value plus provenance."""
    value: float | None
    observed_at: str | None          # ISO date string of the observation
    series: list[tuple[str, float]]  # [(date_str, value), ...] recent window
    provider: str
    series_id: str
    series_name: str
    frequency: str = "unknown"
    note: str | None = None
    status: str = "fresh"            # fresh | stale | missing | error | fallback
    extra: dict = field(default_factory=dict)
