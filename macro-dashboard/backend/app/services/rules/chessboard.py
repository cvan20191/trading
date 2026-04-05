"""
Fed Chessboard — Module 1.

Two-layer model:
  Layer 1 — Primary Quadrant: A / B / C / D
            based on directional rate impulse + balance_sheet_direction
  Layer 2 — Transition Tag: Improving / Stable / Deteriorating
            based on the same impulse signals, contextualized by quadrant

Speaker-faithful directional meanings:
  A  rates down + balance sheet up   → MAX LIQUIDITY
  B  rates up   + balance sheet up   → MIXED LIQUIDITY
  C  rates down + balance sheet down → TRANSITION / MIXED
  D  rates up   + balance sheet down → MAX ILLIQUIDITY

Policy stance remains a secondary context signal for downstream overlays/copy.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.dashboard_state import FedChessboard
from app.schemas.indicator_snapshot import LiquidityInput

# ── Heuristic implementation helpers ─────────────────────────────────────────
# These constants are heuristic defaults — implementation helpers, not
# transcript-exact speaker doctrine. Do not surface these in the UI as if they
# were universal economic laws.

# Zero-bound shortcut: rate at or near ZLB is unambiguously "easy"
_POLICY_ZERO_BOUND: float = 0.50

# Cycle-aware bands applied to 0–1 normalized cycle position
# (bottom 30% of trailing cycle range → easy; top 30% → restrictive)
_CYCLE_EASY_MAX: float = 0.30
_CYCLE_RESTRICTIVE_MIN: float = 0.70

# Fallback fixed bands used only when cycle history is unavailable.
# Labelled explicitly as fallback heuristics; the cycle-aware path is preferred.
_POLICY_EASY_FALLBACK: float = 1.0
_POLICY_RESTRICTIVE_FALLBACK: float = 3.5


# ── Helper functions ──────────────────────────────────────────────────────────

def _policy_stance(rate: float | None, cycle_pos: float | None) -> str:
    """
    Classify policy stance as 'easy' | 'middle' | 'restrictive'.

    Inference order (most to least preferred):
    1. Zero-bound shortcut — rate at or near ZLB is unambiguously easy.
    2. Cycle-aware inference — rate position within trailing 36-month range.
    3. Fallback fixed bands — used only when cycle history is unavailable.

    Rate impulse does NOT remap the middle band. Trend is reserved for the
    transition tag only.
    """
    # 1. Zero-bound shortcut
    if rate is not None and rate <= _POLICY_ZERO_BOUND:
        return "easy"

    # 2. Cycle-aware inference (preferred over fixed cutoffs when available)
    if cycle_pos is not None:
        if cycle_pos <= _CYCLE_EASY_MAX:
            return "easy"
        if cycle_pos >= _CYCLE_RESTRICTIVE_MIN:
            return "restrictive"
        return "middle"

    # 3. Fallback fixed bands (cycle history unavailable)
    if rate is None:
        return "middle"
    if rate <= _POLICY_EASY_FALLBACK:
        return "easy"
    if rate >= _POLICY_RESTRICTIVE_FALLBACK:
        return "restrictive"
    return "middle"


def _rate_impulse(liq: LiquidityInput) -> str:
    """
    Secondary directional signal for the transition tag.
    Returns 'easing' | 'stable' | 'tightening'.
    Uses 1m trend as primary; 3m as confirmation.
    """
    t1 = (liq.rate_trend_1m or "").lower()
    t3 = (liq.rate_trend_3m or "").lower()

    if t1 == "down" or t3 == "down":
        return "easing"
    if t1 == "up" or t3 == "up":
        return "tightening"
    return "stable"


def _bs_direction(liq: LiquidityInput) -> str:
    """
    Classify balance sheet direction as 'expanding' | 'flat_or_mixed' | 'contracting'.

    Mixed-signal-aware: if 1m and 3m disagree, returns 'flat_or_mixed'
    rather than letting one noisy trend dominate.
    """
    b1 = (liq.balance_sheet_trend_1m or "").lower()
    b3 = (liq.balance_sheet_trend_3m or "").lower()

    has_up = b1 == "up" or b3 == "up"
    has_down = b1 == "down" or b3 == "down"

    if has_up and not has_down:
        return "expanding"
    if has_down and not has_up:
        return "contracting"
    return "flat_or_mixed"


@dataclass
class ChessboardResult:
    chessboard: FedChessboard
    liquidity_improving: bool
    liquidity_tight: bool
    quadrant: str  # "A" | "B" | "C" | "D" | "Unknown"


def compute_chessboard(liq: LiquidityInput) -> ChessboardResult:
    """
    Determine the Fed Chessboard quadrant and transition tag.

    Primary quadrant is directional: rate impulse + balance-sheet direction.
    Policy stance is retained as secondary context only.
    """
    stance = _policy_stance(liq.fed_funds_rate, liq.rate_cycle_position)
    impulse = _rate_impulse(liq)
    bs_dir = _bs_direction(liq)

    # ── Primary Quadrant (directional doctrine map) ──────────────────────────
    if impulse == "easing" and bs_dir == "expanding":
        quadrant = "A"
        label = "MAX LIQUIDITY"
    elif impulse == "tightening" and bs_dir == "expanding":
        quadrant = "B"
        label = "MIXED LIQUIDITY: BALANCE SHEET SUPPORT"
    elif impulse == "easing" and bs_dir == "contracting":
        quadrant = "C"
        label = "TRANSITION TO EASIER MONEY"
    elif impulse == "tightening" and bs_dir == "contracting":
        quadrant = "D"
        label = "MAX ILLIQUIDITY"
    # Stable/flat rate impulse is directional ambiguity. Keep smallest-safe
    # fallback behavior by using policy stance only as a tie-breaker.
    elif bs_dir == "expanding":
        if stance == "easy":
            quadrant = "A"
            label = "MAX LIQUIDITY"
        else:
            quadrant = "B"
            label = "MIXED LIQUIDITY: BALANCE SHEET SUPPORT"
    elif bs_dir == "contracting":
        if stance == "restrictive":
            quadrant = "D"
            label = "MAX ILLIQUIDITY"
        else:
            quadrant = "C"
            label = "TRANSITION TO EASIER MONEY"
    else:
        quadrant = "C"
        label = "TRANSITION TO EASIER MONEY"

    # ── Transition Tag ────────────────────────────────────────────────────────
    # Mixed-signal-aware: conflicting signals → Stable rather than forced direction
    supportive = impulse == "easing" or (
        bs_dir == "expanding" and quadrant in ("B", "C")
    )
    adverse = impulse == "tightening" or (
        bs_dir == "contracting" and quadrant in ("C", "D")
    )

    if supportive and not adverse:
        transition_tag = "Improving"
    elif adverse and not supportive:
        transition_tag = "Deteriorating"
    else:
        transition_tag = "Stable"

    # ── Downstream compatibility booleans ────────────────────────────────────
    # Derived from quadrant + transition_tag rather than raw trend strings,
    # keeping downstream logic aligned with the new two-layer model.
    liquidity_improving = quadrant == "A" or transition_tag == "Improving"
    liquidity_tight = quadrant == "D" or (
        quadrant == "C" and transition_tag == "Deteriorating"
    )

    # ── Direction field (preserved for existing consumers) ───────────────────
    direction = _derive_direction(liq)

    cb = FedChessboard(
        quadrant=quadrant,
        label=label,
        rate_trend_1m=liq.rate_trend_1m,
        rate_trend_3m=liq.rate_trend_3m,
        balance_sheet_trend_1m=liq.balance_sheet_trend_1m,
        balance_sheet_trend_3m=liq.balance_sheet_trend_3m,
        direction_vs_1m_ago=direction,
        policy_stance=stance,
        rate_impulse=impulse,
        balance_sheet_direction=bs_dir,
        transition_tag=transition_tag,
    )

    return ChessboardResult(
        chessboard=cb,
        liquidity_improving=liquidity_improving,
        liquidity_tight=liquidity_tight,
        quadrant=quadrant,
    )


def _derive_direction(liq: LiquidityInput) -> str | None:
    """
    Compare 1m vs 3m trends to determine if conditions are improving or
    deteriorating vs a month ago. Preserved for existing consumers.
    """
    r1, r3 = liq.rate_trend_1m, liq.rate_trend_3m
    b1, b3 = liq.balance_sheet_trend_1m, liq.balance_sheet_trend_3m

    if r1 is None and b1 is None:
        return None

    rate_improved = (r1 or "").lower() == "down" and (r3 or "").lower() == "up"
    bs_improved = (b1 or "").lower() == "up" and (b3 or "").lower() in {"down", "flat"}

    rate_worsened = (r1 or "").lower() == "up" and (r3 or "").lower() == "down"
    bs_worsened = (b1 or "").lower() in {"down", "flat"} and (b3 or "").lower() == "up"

    if rate_improved or bs_improved:
        return "improving"
    if rate_worsened or bs_worsened:
        return "deteriorating"
    return "stable"
