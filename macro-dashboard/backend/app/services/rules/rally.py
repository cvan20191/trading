"""
Rally Conditions Engine — Module 6.

Estimates whether the market has "rally fuel" using a deterministic
scoring model grounded in the speaker's framework.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.dashboard_state import RallyConditions
from app.schemas.indicator_snapshot import PolicySupportInput
from app.services.rules.chessboard import ChessboardResult
from app.services.rules.stagflation import StagflationResult
from app.services.rules.stress import StressResult
from app.services.rules.valuation import ValuationResult

# Score starts at 50 and is adjusted by each factor
_BASE_SCORE = 50


@dataclass
class RallyResult:
    conditions: RallyConditions
    rally_fuel_score: int


def compute_rally(
    cb: ChessboardResult,
    stag: StagflationResult,
    val: ValuationResult,
    stress: StressResult,
    policy: PolicySupportInput,
) -> RallyResult:
    score = _BASE_SCORE

    # ── Positive contributors ────────────────────────────────────────────────
    if cb.quadrant == "A":
        score += 15   # max liquidity: best environment
    elif cb.quadrant == "C":
        score += 10   # transition: forward-looking rally possible

    fed_put = bool(policy.fed_put)
    treasury_put = bool(policy.treasury_put)
    political_put = bool(policy.political_put)

    if fed_put:
        score += 10
    if treasury_put:
        score += 8
    if political_put:
        score += 8

    # ── Negative contributors ────────────────────────────────────────────────
    if val.is_stretched:
        score -= 15   # Red zone: new buyers pay a steep premium
    if stag.sticky_inflation:
        score -= 10   # sticky CPI limits Fed's room
    if stag.oil_risk_active:
        score -= 10   # oil above 100 adds inflation pressure
    if stress.stress_warning_active:
        score -= 10   # structural gauges deteriorating

    # Additional penalties for severe regime
    if stress.stress_severe:
        score -= 8
    if stag.trap.active:
        score -= 5    # trap is already partially priced in via sticky + growth

    score = max(0, min(100, score))

    # Market ignoring bad news: rally fuel is meaningful AND a negative signal exists
    negative_present = (
        stag.growth_weakening or stag.sticky_inflation or stress.stress_warning_active
    )
    market_ignoring = score >= 60 and negative_present

    return RallyResult(
        conditions=RallyConditions(
            rally_fuel_score=score,
            fed_put=fed_put,
            treasury_put=treasury_put,
            political_put=political_put,
            market_ignoring_bad_news=market_ignoring,
        ),
        rally_fuel_score=score,
    )
