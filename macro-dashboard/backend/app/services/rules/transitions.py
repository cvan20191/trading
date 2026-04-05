"""
What Changed & What Changes the Call — Module 9 & 10.

Both functions return exactly 3 bullets grounded in indicator trends and
the current regime. No LLM involvement.
"""

from __future__ import annotations

from app.schemas.indicator_snapshot import IndicatorSnapshot
from app.services.rules.chessboard import ChessboardResult
from app.services.rules.regime import RegimeResult
from app.services.rules.stagflation import StagflationResult
from app.services.rules.stress import StressResult
from app.services.rules.valuation import ValuationResult

_FILLER_CHANGED = "No significant recent change detected for this indicator"
_FILLER_TRIGGER = "No additional change trigger identified"


def compute_what_changed(
    snapshot: IndicatorSnapshot,
    cb: ChessboardResult,
    stag: StagflationResult,
    val: ValuationResult,
    stress: StressResult,
) -> list[str]:
    """Return exactly 3 bullets describing recent changes in indicator state."""
    bullets: list[str] = []

    liq = snapshot.liquidity
    infl = snapshot.inflation

    # Rate direction
    r1 = (liq.rate_trend_1m or "").lower()
    if r1 == "down":
        bullets.append("Rate direction has softened over the last month")
    elif r1 == "up":
        bullets.append("Rates moved higher over the last month, adding pressure to valuations")
    elif r1 == "flat":
        bullets.append("Rates have held steady — no material change in the cost of capital")

    # Balance sheet
    b1 = (liq.balance_sheet_trend_1m or "").lower()
    if b1 == "down":
        bullets.append(
            "Fed balance sheet is still contracting — net liquidity support has not arrived"
        )
    elif b1 == "flat":
        bullets.append(
            "Balance-sheet contraction has paused; while liquidity remains "
            "supportive, the trend has not yet clearly turned into a sustained expansion."
        )
    elif b1 == "up":
        bullets.append(
            "Fed balance sheet has begun expanding — a liquidity tailwind is building"
        )

    # Oil
    if stag.oil_risk_active:
        wti = infl.wti_oil
        wti_str = f" at ${wti:.0f}" if wti else ""
        bullets.append(f"WTI crude{wti_str} has moved into the speaker's inflation-risk zone")
    elif infl.wti_oil is not None and infl.wti_oil < 100 and infl.wti_oil > 85:
        bullets.append(
            f"WTI crude at ${infl.wti_oil:.0f} — below the risk threshold but worth monitoring"
        )

    # Valuation
    _is_proxy = val.valuation.is_fallback
    _val_label = "Proxy valuation" if _is_proxy else "Forward valuation"
    _proxy_note = " (directional proxy — confirm with true Forward P/E before acting)" if _is_proxy else ""
    if val.is_stretched:
        pe = val.valuation.forward_pe
        pe_str = f" at {pe:.1f}x" if pe else ""
        bullets.append(f"{_val_label} remains stretched{pe_str} — above the pause threshold{_proxy_note}")
    elif val.is_buy_zone:
        bullets.append(
            f"{_val_label} has compressed into the historical accumulation zone{_proxy_note}"
        )

    # Unemployment trend
    u_trend = (snapshot.growth.unemployment_trend or "").lower()
    if u_trend == "up":
        bullets.append("Labor slack is beginning to build — unemployment has been trending higher")
    elif u_trend == "down":
        bullets.append("Labor market has tightened — unemployment is still falling")

    # PMI contraction
    pmi = snapshot.growth.pmi_manufacturing
    if pmi is not None and pmi < 50:
        bullets.append(
            f"Manufacturing PMI at {pmi:.1f} — remains in contraction territory"
        )

    # Yield curve
    if stress.stress.yield_curve_inverted:
        yc = stress.stress.yield_curve_value
        yc_str = f" at {yc:.2f}%" if yc is not None else ""
        bullets.append(f"Yield curve remains inverted{yc_str} — recession-watch signal persists")

    # CPI stickiness
    cpi = snapshot.inflation.core_cpi_yoy
    if cpi is not None and cpi > 3.0:
        bullets.append(
            f"Core CPI at {cpi:.1f}% — remains above the speaker's sticky threshold"
        )

    # Pad or truncate to exactly 3
    while len(bullets) < 3:
        bullets.append(_FILLER_CHANGED)
    return bullets[:3]


def compute_what_changes_call(
    regime_result: RegimeResult,
    val: ValuationResult,
    stag: StagflationResult,
    stress: StressResult,
    cb: ChessboardResult,
) -> list[str]:
    """Return exactly 3 trigger bullets that would change the current posture."""
    bullets: list[str] = []

    regime = regime_result.primary_regime

    # Valuation trigger
    _is_proxy = val.valuation.is_fallback
    _confirm_note = " — confirm with true Forward P/E before acting" if _is_proxy else ""
    if val.is_stretched:
        bullets.append(
            f"{'Proxy valuation' if _is_proxy else 'Forward P/E'} compresses below 25x "
            f"through earnings growth or a price reset{_confirm_note}"
        )
    elif val.is_buy_zone:
        bullets.append(
            f"{'Proxy valuation' if _is_proxy else 'Valuation'} moves back above the buy zone "
            f"— reassess accumulation pace{_confirm_note}"
        )
    else:
        bullets.append(
            f"{'Proxy valuation' if _is_proxy else 'Forward P/E'} falls into the 20–25x "
            f"accumulation zone — a cleaner entry signal{_confirm_note}"
        )

    # Inflation / oil trigger
    if stag.oil_risk_active:
        bullets.append(
            "WTI crude drops back below $100 and sustains lower — removing the oil inflation risk"
        )
    if stag.sticky_inflation:
        bullets.append(
            "Core services inflation decisively rolls over toward the 2.5% range"
        )

    # Liquidity trigger
    bs = cb.chessboard.balance_sheet_trend_1m
    if bs in {"down", "flat"}:
        bullets.append(
            "Fed balance sheet turns clearly expansionary for multiple consecutive weeks"
        )
    elif bs == "up":
        bullets.append(
            "Fed balance sheet expansion reverses or stalls — reducing the liquidity tailwind"
        )

    # Trap / labor trigger
    if stag.trap.active:
        bullets.append(
            "Unemployment rises enough to reopen a cleaner rate-cut path, or CPI falls below 3%"
        )

    # Stress triggers
    if stress.stress.yield_curve_inverted:
        bullets.append(
            "Yield curve un-inverts as growth expectations improve alongside falling short rates"
        )
    if stress.stress.npl_zone in {"Caution", "Warning"}:
        bullets.append(
            f"NPL ratio accelerates toward or past the {1.5:.1f}% warning threshold"
        )

    # Regime-specific additions
    if regime == "Crash Watch":
        bullets.append(
            "Systemic stress gauges stabilize or reverse — NPL, Market Cap/M2 stop worsening"
        )
    if regime == "Max Liquidity":
        bullets.append(
            "Inflation re-accelerates and forces the Fed to pause or reverse cuts"
        )

    while len(bullets) < 3:
        bullets.append(_FILLER_TRIGGER)
    return bullets[:3]
