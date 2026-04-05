"""
Regime Classification — Module 7.

Computes primary_regime, secondary_overlays, confidence, and current_posture
from the outputs of all other rule modules.

All classification is deterministic. The LLM never touches this logic.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.services.rules.chessboard import ChessboardResult
from app.services.rules.rally import RallyResult
from app.services.rules.stagflation import StagflationResult
from app.services.rules.stress import DollarResult, StressResult
from app.services.rules.valuation import ValuationResult

# Primary regime labels — kept in a single place for easy search
R_MAX_LIQUIDITY = "Max Liquidity"
R_LIQUIDITY_TRANSITION = "Liquidity Transition"
R_STAGFLATION_TRAP = "Stagflation Trap"
R_VALUATION_STRETCHED = "Valuation Stretched"
R_BUY_THE_DIP = "Buy-the-Dip Window"
R_CRASH_WATCH = "Crash Watch"
R_DEFENSIVE = "Defensive / Illiquid Regime"
R_MIXED = "Mixed / Conflicted Regime"

# Overlay labels
O_STICKY_INFLATION = "Sticky Inflation"
O_GROWTH_WEAKENING = "Growth Weakening"
O_RALLY_FUEL = "Rally Fuel Active"
O_SYSTEMIC_STRESS = "Systemic Stress Rising"
O_DOLLAR_PRESSURE = "Dollar Pressure"
O_VAL_SUPPORTIVE = "Valuation Supportive"
O_VAL_DANGEROUS = "Valuation Stretched"   # matches zone_label language; non-sensational


@dataclass
class RegimeResult:
    primary_regime: str
    secondary_overlays: list[str]
    confidence: str
    current_posture: str


def compute_regime(
    cb: ChessboardResult,
    stag: StagflationResult,
    val: ValuationResult,
    stress: StressResult,
    dollar: DollarResult,
    rally: RallyResult,
) -> RegimeResult:
    # ── Primary regime — deterministic precedence ────────────────────────────

    if stress.stress_severe and cb.liquidity_tight:
        regime = R_CRASH_WATCH

    elif stag.trap.active:
        regime = R_STAGFLATION_TRAP

    elif cb.quadrant == "D" and not val.is_buy_zone:
        regime = R_DEFENSIVE

    elif cb.quadrant == "A":
        regime = R_MAX_LIQUIDITY

    elif val.is_buy_zone and (cb.liquidity_improving or not cb.liquidity_tight):
        regime = R_BUY_THE_DIP

    elif val.is_stretched and cb.quadrant not in {"A"}:
        regime = R_VALUATION_STRETCHED

    elif cb.quadrant == "C" or (cb.liquidity_improving and not stag.trap.active):
        regime = R_LIQUIDITY_TRANSITION

    else:
        regime = R_MIXED

    # ── Secondary overlays ───────────────────────────────────────────────────
    overlays: list[str] = []

    if stag.sticky_inflation:
        overlays.append(O_STICKY_INFLATION)
    if stag.growth_weakening:
        overlays.append(O_GROWTH_WEAKENING)
    if rally.rally_fuel_score >= 60:
        overlays.append(O_RALLY_FUEL)
    if stress.stress_warning_active:
        overlays.append(O_SYSTEMIC_STRESS)
    if dollar.dxy_pressure:
        overlays.append(O_DOLLAR_PRESSURE)
    if val.is_buy_zone:
        overlays.append(O_VAL_SUPPORTIVE)
    if val.is_stretched:
        overlays.append(O_VAL_DANGEROUS)

    # ── Confidence ───────────────────────────────────────────────────────────
    # Count how many signals align with the classified regime
    aligning = _count_aligning_signals(regime, cb, stag, val, stress, rally)
    if aligning >= 4:
        confidence = "High"
    elif aligning >= 2:
        confidence = "Medium"
    else:
        confidence = "Low"

    # ── Posture ───────────────────────────────────────────────────────────────
    posture = _derive_posture(regime, val, stag, stress, cb)

    return RegimeResult(
        primary_regime=regime,
        secondary_overlays=overlays,
        confidence=confidence,
        current_posture=posture,
    )


def _count_aligning_signals(
    regime: str,
    cb: ChessboardResult,
    stag: StagflationResult,
    val: ValuationResult,
    stress: StressResult,
    rally: RallyResult,
) -> int:
    count = 0
    if regime == R_CRASH_WATCH:
        if stress.stress_severe: count += 2
        if cb.liquidity_tight: count += 1
        if stag.growth_weakening: count += 1
        if val.is_stretched: count += 1
    elif regime == R_STAGFLATION_TRAP:
        if stag.sticky_inflation: count += 2
        if stag.growth_weakening: count += 2
        if stag.trap.active: count += 2
    elif regime == R_MAX_LIQUIDITY:
        if cb.quadrant == "A": count += 3
        if not val.is_stretched: count += 1
        if rally.rally_fuel_score >= 60: count += 1
    elif regime == R_BUY_THE_DIP:
        if val.is_buy_zone: count += 3
        if cb.liquidity_improving: count += 1
        if not stress.stress_severe: count += 1
    elif regime == R_VALUATION_STRETCHED:
        if val.is_stretched: count += 3
        if not cb.liquidity_improving: count += 1
    elif regime == R_LIQUIDITY_TRANSITION:
        if cb.quadrant == "C": count += 2
        if cb.liquidity_improving: count += 1
        if not stag.trap.active: count += 1
    elif regime == R_DEFENSIVE:
        if cb.quadrant == "D": count += 3
        if not val.is_buy_zone: count += 1
    else:  # mixed
        count = 1
    return count


def _derive_posture(
    regime: str,
    val: ValuationResult,
    stag: StagflationResult,
    stress: StressResult,
    cb: ChessboardResult,
) -> str:
    if regime == R_MAX_LIQUIDITY:
        return (
            "Risk-on conditions are supportive; favor growth names and be willing "
            "to hold higher-multiple positions with manageable debt."
        )
    if regime == R_LIQUIDITY_TRANSITION:
        if val.is_stretched:
            return (
                "Hold existing winners, avoid aggressive new buying, and wait for "
                "either valuation compression or cleaner liquidity confirmation."
            )
        return (
            "Liquidity is improving at the margin; accumulate selectively on "
            "pullbacks, with preference for profitable names over pure speculation."
        )
    if regime == R_STAGFLATION_TRAP:
        return (
            "Defensive patience; favor profitable, lower-debt names over "
            "high-multiple growth — the Fed cannot cleanly rescue either "
            "growth or valuations here."
        )
    if regime == R_BUY_THE_DIP:
        return (
            "Valuation has reset into the historical accumulation zone; "
            "accumulate slowly on pullbacks, prioritising quality over speed."
        )
    if regime == R_VALUATION_STRETCHED:
        return (
            "Halt new accumulation at current levels; be patient and wait for "
            "valuation to compress through earnings growth or a price reset."
        )
    if regime == R_CRASH_WATCH:
        return (
            "Preserve capital; reduce exposure to low-quality, high-debt names "
            "and favor cash, short-duration assets, or defensive quality."
        )
    if regime == R_DEFENSIVE:
        return (
            "Capital is scarce and expensive; stay in low-debt, high-profitability "
            "names and preserve dry powder for better entry conditions."
        )
    # Mixed / Conflicted
    return (
        "Signals are mixed; stay patient, be selective, and avoid "
        "aggressive positioning in either direction until the regime clarifies."
    )
