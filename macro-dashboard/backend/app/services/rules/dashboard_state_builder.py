"""
DashboardState Builder — the orchestrator.

Accepts an IndicatorSnapshot and runs all rule modules in sequence to
produce a fully valid DashboardState. This is the single deterministic
entry point for all rule logic.

Flow:
  1. chessboard
  2. stagflation trap
  3. valuation
  4. systemic stress
  5. dollar context
  6. rally conditions
  7. regime + overlays + confidence + posture
  8. top watchpoints
  9. what changed
  10. what changes the call
  11. assemble DashboardState

Public API:
  build_dashboard_state(snapshot)                -> DashboardState
  build_dashboard_state_with_conclusion(snapshot) -> (DashboardState, PlaybookConclusion)
"""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.dashboard_state import DataFreshness, DashboardState
from app.schemas.indicator_snapshot import IndicatorSnapshot
from app.schemas.playbook_conclusion import PlaybookConclusion
from app.services.rules.chessboard import ChessboardResult, compute_chessboard
from app.services.rules.playbook_conclusion import build_playbook_conclusion
from app.services.rules.rally import RallyResult, compute_rally
from app.services.rules.regime import RegimeResult, compute_regime
from app.services.rules.stagflation import StagflationResult, compute_stagflation
from app.services.rules.stress import StressResult, compute_dollar, compute_stress
from app.services.rules.transitions import compute_what_changed, compute_what_changes_call
from app.services.rules.valuation import ValuationResult, compute_valuation
from app.services.rules.watchpoints import compute_watchpoints


@dataclass
class _RuleOutputs:
    """Private container for all intermediate rule results."""
    cb: ChessboardResult
    stag: StagflationResult
    val: ValuationResult
    stress: StressResult
    dollar: object
    rally: RallyResult
    regime: RegimeResult
    watchpoints: list
    what_changed: list
    what_changes_call: list
    freshness: DataFreshness


def _run_rules(snapshot: IndicatorSnapshot) -> _RuleOutputs:
    """
    Run all rule modules in order and return every intermediate result.

    This is the single source of truth for rule computation. Both public
    builder functions delegate here so computation is never duplicated.
    """
    # ── Step 1: Fed Chessboard ────────────────────────────────────────────────
    cb = compute_chessboard(snapshot.liquidity)

    # ── Step 2: Stagflation Trap ──────────────────────────────────────────────
    stag = compute_stagflation(snapshot.growth, snapshot.inflation)

    # ── Step 3: Valuation ─────────────────────────────────────────────────────
    val = compute_valuation(snapshot.valuation)

    # ── Step 4: Systemic Stress ───────────────────────────────────────────────
    stress = compute_stress(snapshot.systemic_stress)

    # ── Step 5: Dollar Context ────────────────────────────────────────────────
    dollar = compute_dollar(snapshot.dollar_context)

    # ── Step 6: Rally Conditions ──────────────────────────────────────────────
    rally = compute_rally(cb, stag, val, stress, snapshot.policy_support)

    # ── Step 7: Regime ────────────────────────────────────────────────────────
    regime = compute_regime(cb, stag, val, stress, dollar, rally)

    # ── Step 8: Top Watchpoints ───────────────────────────────────────────────
    watchpoints = compute_watchpoints(
        cb, stag, val, stress, dollar, regime.primary_regime
    )

    # ── Step 9: What Changed ──────────────────────────────────────────────────
    what_changed = compute_what_changed(snapshot, cb, stag, val, stress)

    # ── Step 10: What Changes the Call ────────────────────────────────────────
    what_changes_call = compute_what_changes_call(regime, val, stag, stress, cb)

    # ── Step 11: Freshness ────────────────────────────────────────────────────
    freshness = DataFreshness(
        overall_status=snapshot.data_freshness.overall_status or "unknown",
        stale_series=snapshot.data_freshness.stale_series,
    )

    return _RuleOutputs(
        cb=cb,
        stag=stag,
        val=val,
        stress=stress,
        dollar=dollar,
        rally=rally,
        regime=regime,
        watchpoints=watchpoints,
        what_changed=what_changed,
        what_changes_call=what_changes_call,
        freshness=freshness,
    )


def _assemble_state(snapshot: IndicatorSnapshot, r: _RuleOutputs) -> DashboardState:
    """Assemble a DashboardState from pre-computed rule outputs."""
    return DashboardState(
        as_of=snapshot.as_of,
        data_freshness=r.freshness,
        primary_regime=r.regime.primary_regime,
        secondary_overlays=r.regime.secondary_overlays,
        confidence=r.regime.confidence,
        current_posture=r.regime.current_posture,
        fed_chessboard=r.cb.chessboard,
        stagflation_trap=r.stag.trap,
        valuation=r.val.valuation,
        systemic_stress=r.stress.stress,
        dollar_context=r.dollar.dollar,
        rally_conditions=r.rally.conditions,
        top_watchpoints=r.watchpoints,
        what_changed=r.what_changed,
        what_changes_call=r.what_changes_call,
    )


def build_dashboard_state(snapshot: IndicatorSnapshot) -> DashboardState:
    """
    Deterministic, side-effect-free orchestrator.

    All regime classification, overlay detection, watchpoint ranking,
    and change detection happens here. The LLM never sees raw indicators.
    """
    r = _run_rules(snapshot)
    return _assemble_state(snapshot, r)


def build_dashboard_state_with_conclusion(
    snapshot: IndicatorSnapshot,
) -> tuple[DashboardState, PlaybookConclusion]:
    """
    Companion builder returning both DashboardState and PlaybookConclusion.

    Runs all rule modules exactly once. Use this in any call site that needs
    the structured PlaybookConclusion alongside the standard DashboardState.
    Existing callers of build_dashboard_state() are unaffected.
    """
    r = _run_rules(snapshot)
    state = _assemble_state(snapshot, r)
    conclusion = build_playbook_conclusion(
        cb=r.cb,
        val=r.val,
        stag=r.stag,
        stress=r.stress,
        rally=r.rally,
        regime=r.regime,
    )
    return state, conclusion
