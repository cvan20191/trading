"""
Deterministic fallback summary generator.

Used whenever the LLM call fails (network error, timeout, schema mismatch,
or any other exception). Returns a valid PlaybookSummary that is calm,
educational, and grounded entirely in the supplied DashboardState — no LLM
involved, never fails for well-formed input.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.schemas.dashboard_state import DashboardState
from app.schemas.playbook_conclusion import PlaybookConclusion
from app.schemas.summary import (
    FILLER_CHANGED,
    FILLER_TRIGGER,
    FILLER_WATCH,
    PlaybookSummary,
    SummaryMeta,
    pad_or_truncate,
)

# ---------------------------------------------------------------------------
# Regime-keyed teaching notes
# ---------------------------------------------------------------------------
_REGIME_TEACHING_NOTES: dict[str, str] = {
    "Max Liquidity": (
        "In max-liquidity setups, rates are generally easier and balance-sheet "
        "pressure has eased, contributing to a supportive liquidity regime."
    ),
    "Liquidity Transition": (
        "A liquidity transition means one lever is improving while another remains "
        "tight; markets can rally on expectations of better conditions even before "
        "they fully arrive."
    ),
    "Stagflation Trap": (
        "The stagflation trap describes a Fed caught between weak growth and sticky "
        "inflation — cutting rates risks reigniting inflation, while keeping rates "
        "tight risks deepening the slowdown."
    ),
    "Valuation Stretched": (
        "A stretched forward P/E is a pause signal for new accumulation, not an "
        "automatic sell signal, because strong earnings can compress the multiple "
        "over time even without a price decline."
    ),
    "Buy-the-Dip Window": (
        "A buy-the-dip window appears when valuation has reset into the historical "
        "accumulation zone and liquidity conditions are stabilising or improving — "
        "the combination matters more than either factor alone."
    ),
    "Crash Watch": (
        "Crash watch means multiple structural stress gauges are worsening "
        "simultaneously; these are warning lights, not exact timers, but they signal "
        "rising systemic fragility."
    ),
    "Defensive / Illiquid Regime": (
        "In a defensive regime, capital is expensive and scarce; the playbook favours "
        "low-debt, high-profitability names and preserving dry powder rather than "
        "reaching for yield."
    ),
    "Mixed / Conflicted Regime": (
        "A mixed regime means indicators are pulling in different directions; patience "
        "and selectivity matter more than aggressive directional positioning."
    ),
}

_DEFAULT_TEACHING_NOTE = (
    "In this playbook, thresholds such as the forward P/E level or oil above a key "
    "zone are action zones and warning bands, not deterministic prophecies — the "
    "market frequently overshoots both before reversing."
)

_ACTION_POSTURE: dict[str, str] = {
    "accumulate": "Accumulate selectively",
    "accumulate_selectively": "Accumulate with criteria",
    "hold_and_wait": "Hold and wait",
    "pause_new_buying": "Pause new buying — not a sell signal",
    "defensive_preservation": "Defensive capital preservation",
}

_LENIENCY_TEACHING: dict[str, str] = {
    "valuation_proxy_not_true_forward_pe": (
        "This valuation signal is a proxy, not a pure forward P/E series, so treat "
        "it as a disciplined zone marker rather than a precision trigger."
    ),
    "stress_gauges_warning_lights_not_timers": (
        "Systemic stress gauges are warning lights, not timers; they raise caution "
        "without predicting exact turning points."
    ),
    "transition_regime_can_overshoot": (
        "Transition regimes can overshoot in both directions before settling, so "
        "positioning should stay selective and adaptive."
    ),
    "stretched_means_pause_not_forced_sell": (
        "Stretched valuation means pause new buying, not forced selling; price can "
        "normalize through time as fundamentals catch up."
    ),
    "mixed_signals_reduce_conviction": (
        "Mixed signals reduce conviction; when indicators disagree, prioritize risk "
        "control over aggressive directional bets."
    ),
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _derive_main_tension(state: DashboardState) -> str:
    tensions: list[str] = []

    if state.stagflation_trap and state.stagflation_trap.active:
        tensions.append(
            "the stagflation trap keeps the Fed constrained between weak growth "
            "and sticky inflation"
        )
    if state.valuation and state.valuation.zone == "Red":
        tensions.append(
            "big-tech valuation remains stretched above the speaker's pause threshold"
        )
    if state.systemic_stress and state.systemic_stress.yield_curve_inverted:
        tensions.append(
            "the yield curve is flagging a recession-watch horizon"
        )
    if state.dollar_context and state.dollar_context.dxy_pressure:
        tensions.append(
            "dollar pressure is adding friction to the macro backdrop"
        )

    if not tensions:
        return (
            "macro conditions are mixed and require monitoring across multiple "
            "indicators before the picture clears"
        )
    return "; ".join(tensions)


def _derive_expanded_summary(
    state: DashboardState,
    tension: str,
    can_rally: bool | None = None,
) -> str:
    regime = state.primary_regime
    parts: list[str] = [
        f"The current regime is classified as {regime}, driven by the interaction "
        f"of liquidity conditions, inflation data, and valuation signals.",
        f"The primary tension is that {tension}.",
    ]

    trap = state.stagflation_trap
    if trap and trap.active:
        parts.append(
            "The Fed remains constrained: cutting rates risks reigniting inflation, "
            "while holding rates risks further economic softening, so the playbook "
            "does not expect clean policy relief in the near term."
        )

    if can_rally is True:
        parts.append(
            "Rally conditions are partially active, which means the market may "
            "continue looking through near-term weak data if forward liquidity "
            "expectations remain supportive."
        )
    elif can_rally is False:
        parts.append(
            "Rally fuel is limited in the current setup, making the market more "
            "vulnerable to negative surprises than the headline price level "
            "might suggest."
        )
    else:
        rc = state.rally_conditions
        if rc and rc.rally_fuel_score is not None:
            if rc.rally_fuel_score >= 60:
                parts.append(
                    "Rally conditions are partially active, which means the market may "
                    "continue looking through near-term weak data if forward liquidity "
                    "expectations remain supportive."
                )
            else:
                parts.append(
                    "Rally fuel is limited in the current setup, making the market more "
                    "vulnerable to negative surprises than the headline price level "
                    "might suggest."
                )

    posture_snippet = state.current_posture[:110].rstrip(".")
    parts.append(
        f"The implied posture is to {posture_snippet.lower()} until conditions "
        f"improve more clearly."
    )

    return " ".join(parts[:5])


def _derive_posture_label(posture: str) -> str:
    if "," in posture:
        return posture.split(",")[0].strip()
    return posture[:60].strip()


def _derive_risk_flags(state: DashboardState) -> list[str]:
    flags = list(state.secondary_overlays)
    trap = state.stagflation_trap
    if trap and trap.active and "Stagflation Trap Active" not in flags:
        flags.append("Stagflation Trap Active")
    val = state.valuation
    if val and val.zone == "Red" and "Valuation Stretched" not in flags:
        flags.append("Valuation Stretched")
    stress = state.systemic_stress
    if stress and stress.yield_curve_inverted and "Yield Curve Warning" not in flags:
        flags.append("Yield Curve Warning")
    dc = state.dollar_context
    if dc and dc.dxy_pressure and "Dollar Pressure" not in flags:
        flags.append("Dollar Pressure")
    return flags


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_fallback_summary(
    state: DashboardState,
    model_name: str | None = None,
    conclusion: PlaybookConclusion | None = None,
) -> PlaybookSummary:
    """
    Return a valid PlaybookSummary without calling any external service.

    This is the guaranteed-safe path. It must never raise for well-formed input.
    """
    regime = state.primary_regime
    tension = _derive_main_tension(state)
    posture_label = (
        _ACTION_POSTURE.get(conclusion.new_cash_action, _derive_posture_label(state.current_posture))
        if conclusion is not None
        else _derive_posture_label(state.current_posture)
    )

    headline = (
        f"The macro dashboard is currently in a {regime} regime. "
        f"The framework implies: {posture_label}."
    )
    expanded = _derive_expanded_summary(
        state,
        tension,
        can_rally=conclusion.can_rally_despite_bad_news if conclusion is not None else None,
    )

    watch_now = pad_or_truncate(state.top_watchpoints[:3], 3, FILLER_WATCH)
    what_changed_bullets = pad_or_truncate(state.what_changed[:3], 3, FILLER_CHANGED)
    what_changes_call_bullets = pad_or_truncate(
        state.what_changes_call[:3], 3, FILLER_TRIGGER
    )

    risk_flags = _derive_risk_flags(state)
    if conclusion is not None and conclusion.warning_urgency != "cautionary":
        urgency_flag = conclusion.warning_urgency.upper()
        if urgency_flag not in risk_flags:
            risk_flags = [urgency_flag, *risk_flags]

    teaching_note = _REGIME_TEACHING_NOTES.get(regime, _DEFAULT_TEACHING_NOTE)
    if conclusion is not None and conclusion.leniency_notes:
        first_note = conclusion.leniency_notes[0]
        teaching_note = _LENIENCY_TEACHING.get(first_note, teaching_note)

    meta = SummaryMeta(
        used_fallback=True,
        generated_at=datetime.now(timezone.utc).isoformat(),
        model=model_name,
        data_status=state.data_freshness.overall_status,
    )

    return PlaybookSummary(
        headline_summary=headline,
        expanded_summary=expanded,
        regime_label=regime,
        posture_label=posture_label,
        watch_now=watch_now,
        what_changed_bullets=what_changed_bullets,
        what_changes_call_bullets=what_changes_call_bullets,
        risk_flags=risk_flags,
        teaching_note=teaching_note,
        meta=meta,
    )
