"""
PlaybookConclusion builder.

Phase 1 scope:
- pure deterministic derivation from already-computed rule outputs
- no orchestration wiring
- no API/summary/prompt/frontend dependencies
"""

from __future__ import annotations

from app.schemas.playbook_conclusion import (
    ArchetypeLabel,
    LeniencyNote,
    NewCashAction,
    PlaybookConclusion,
    WarningUrgency,
)
from app.services.rules.chessboard import ChessboardResult
from app.services.rules.rally import RallyResult
from app.services.rules.regime import RegimeResult
from app.services.rules.stagflation import StagflationResult
from app.services.rules.stress import StressResult
from app.services.rules.valuation import ValuationResult


_WHY_NOW_LIQUIDITY_BY_QUADRANT: dict[str, str] = {
    "A": "liquidity_quadrant_a_supportive",
    "B": "liquidity_quadrant_b_mixed_support",
    "C": "liquidity_quadrant_c_transition",
    "D": "liquidity_quadrant_d_tight",
}


def _quadrant_archetypes(
    quadrant: str,
) -> tuple[list[ArchetypeLabel], list[ArchetypeLabel]]:
    if quadrant == "A":
        return (
            [
                "hyper_growth_manageable_debt",
                "high_debt_unprofitable_growth",
            ],
            [
                "valuation_dependent_speculation",
            ],
        )
    if quadrant == "B":
        return (
            [
                "moderate_growth_moderate_leverage",
                "profitable_cashflow_compounders",
            ],
            [
                "high_debt_unprofitable_growth",
                "valuation_dependent_speculation",
            ],
        )
    if quadrant == "C":
        return (
            [
                "high_growth_refinancing_beneficiary",
                "profitable_cashflow_compounders",
            ],
            [
                "deep_cyclical_balance_sheet_risk",
                "high_debt_unprofitable_growth",
            ],
        )
    if quadrant == "D":
        return (
            [
                "defensive_low_debt_low_valuation",
                "balance_sheet_strength_priority",
            ],
            [
                "high_debt_unprofitable_growth",
                "valuation_dependent_speculation",
                "deep_cyclical_balance_sheet_risk",
            ],
        )
    return (
        [
            "balance_sheet_strength_priority",
        ],
        [
            "high_debt_unprofitable_growth",
            "valuation_dependent_speculation",
        ],
    )


def _defensive_override_needed(
    cb: ChessboardResult,
    val: ValuationResult,
    stag: StagflationResult,
    stress: StressResult,
) -> bool:
    # Doctrine leniency: in max-liquidity A with buy-zone valuation and no trap,
    # stress remains a caution overlay (warning light), not a full defensive flip.
    if cb.quadrant == "A" and val.is_buy_zone and not stag.trap.active:
        return False
    # Doctrine leniency extension: B/C are mixed regimes and should not auto-collapse
    # into D-style preservation from stress alone when trap is inactive.
    if cb.quadrant in {"B", "C"} and not stag.trap.active:
        return False
    return stress.stress_severe or (stag.trap.active and cb.liquidity_tight)


def _derive_new_cash_action(
    cb: ChessboardResult,
    val: ValuationResult,
    stag: StagflationResult,
    stress: StressResult,
) -> NewCashAction:
    # Precedence contract:
    # 1) severe stress defensive override
    # 2) trap + tight liquidity
    # 3) minimum-liquidity quadrant (D) — doctrine: ~80% cash / sidelines
    # 4) stretched valuation pause rule
    # 5) buy-zone accumulation
    # 6) max-liquidity quadrant (A) — doctrine: low cash / active deployment
    # 7) improving-liquidity selectivity
    # 8) default hold
    if cb.quadrant == "A" and val.is_buy_zone and not stag.trap.active:
        return "accumulate_selectively"
    if _defensive_override_needed(cb, val, stag, stress):
        return "defensive_preservation"
    if cb.quadrant == "D":
        return "defensive_preservation"
    if val.is_stretched:
        return "pause_new_buying"
    if val.is_buy_zone:
        return "accumulate_selectively"
    if cb.quadrant == "A":
        return "accumulate_selectively"
    if cb.liquidity_improving:
        return "accumulate_selectively"
    return "hold_and_wait"


def _derive_warning_urgency(
    cb: ChessboardResult,
    val: ValuationResult,
    stag: StagflationResult,
    stress: StressResult,
) -> WarningUrgency:
    # Doctrine leniency: A + buy-zone + no trap keeps stress as a warning light,
    # not a full-urgency trigger. Cap at elevated — consistent with action/archetype leniency.
    if cb.quadrant == "A" and val.is_buy_zone and not stag.trap.active:
        if stress.stress_warning_active or stress.stress_severe:
            return "elevated"
        return "cautionary"
    # Doctrine leniency extension: in mixed B/C without trap, severe stress can
    # intensify caution but should not auto-escalate to full urgent posture.
    if cb.quadrant in {"B", "C"} and not stag.trap.active:
        if stress.stress_warning_active or stress.stress_severe or cb.liquidity_tight:
            return "elevated"
        return "cautionary"
    if stress.stress_severe or (stag.trap.active and cb.liquidity_tight):
        return "urgent"
    if stress.stress_warning_active or stag.trap.active or cb.liquidity_tight:
        return "elevated"
    return "cautionary"


def _derive_leniency_notes(
    cb: ChessboardResult,
    val: ValuationResult,
    stress: StressResult,
    regime: RegimeResult,
) -> list[LeniencyNote]:
    notes: list[LeniencyNote] = []

    if val.valuation.is_fallback:
        notes.append("valuation_proxy_not_true_forward_pe")

    if stress.stress_warning_active:
        notes.append("stress_gauges_warning_lights_not_timers")

    if cb.quadrant == "C":
        notes.append("transition_regime_can_overshoot")

    if val.is_stretched:
        notes.append("stretched_means_pause_not_forced_sell")

    if regime.confidence == "Low":
        notes.append("mixed_signals_reduce_conviction")

    return notes


def _build_why_now(
    cb: ChessboardResult,
    val: ValuationResult,
    stag: StagflationResult,
    stress: StressResult,
    rally: RallyResult,
) -> str:
    # Deterministic contract:
    # - max 3 drivers
    # - priority: liquidity, valuation, risk-context
    # - semicolon-joined canonical phrases only
    drivers: list[str] = []

    drivers.append(
        _WHY_NOW_LIQUIDITY_BY_QUADRANT.get(
            cb.quadrant,
            "liquidity_quadrant_c_transition",
        )
    )

    if val.is_stretched:
        drivers.append("valuation_stretched_pause_new_buying")
    elif val.is_buy_zone:
        drivers.append("valuation_buy_zone")
    else:
        drivers.append("valuation_neutral_wait_for_edge")

    # Risk-context priority order:
    # trap / severe stress / warning stress / rally despite bad news
    if stag.trap.active:
        drivers.append("stagflation_trap_active")
    elif stress.stress_severe:
        drivers.append("systemic_stress_severe")
    elif stress.stress_warning_active:
        drivers.append("systemic_stress_warning_active")
    elif rally.conditions.market_ignoring_bad_news:
        drivers.append("market_ignoring_bad_news")

    return "; ".join(drivers[:3])


def build_playbook_conclusion(
    cb: ChessboardResult,
    val: ValuationResult,
    stag: StagflationResult,
    stress: StressResult,
    rally: RallyResult,
    regime: RegimeResult,
) -> PlaybookConclusion:
    preferred, avoid = _quadrant_archetypes(cb.quadrant)

    if _defensive_override_needed(cb, val, stag, stress):
        preferred = [
            "defensive_low_debt_low_valuation",
            "balance_sheet_strength_priority",
            "profitable_cashflow_compounders",
        ]
        avoid = [
            "high_debt_unprofitable_growth",
            "valuation_dependent_speculation",
            "deep_cyclical_balance_sheet_risk",
        ]
    elif cb.quadrant == "A" and stress.stress_severe:
        # Stress-quality tightening: severe stress in Quadrant A narrows prefer to
        # manageable-debt names only. high_debt_unprofitable_growth moves to avoid
        # because "accumulate_selectively" implies a quality filter is active.
        # Not a full D-style flip — hyper_growth_manageable_debt stays preferred.
        preferred = [a for a in preferred if a != "high_debt_unprofitable_growth"]
        if "high_debt_unprofitable_growth" not in avoid:
            avoid = list(avoid) + ["high_debt_unprofitable_growth"]

    return PlaybookConclusion(
        conclusion_label=regime.primary_regime,
        new_cash_action=_derive_new_cash_action(cb, val, stag, stress),
        stock_archetype_preferred=preferred,
        stock_archetype_avoid=avoid,
        can_rally_despite_bad_news=rally.conditions.market_ignoring_bad_news,
        warning_urgency=_derive_warning_urgency(cb, val, stag, stress),
        leniency_notes=_derive_leniency_notes(cb, val, stress, regime),
        why_now=_build_why_now(cb, val, stag, stress, rally),
    )
