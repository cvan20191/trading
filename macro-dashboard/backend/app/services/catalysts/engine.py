"""
Catalyst logic engine.

`build_catalyst_state` is a pure, synchronous function that consumes:
  - raw catalyst config dict (from config_loader)
  - current IndicatorSnapshot (live or mock)
  - current DashboardState (from macro rule engine)

and returns a fully typed CatalystState.

All logic is deterministic. No I/O happens here.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from app.schemas.catalysts import (
    CatalystState,
    CleanCutWatch,
    FedChairCandidate,
    FedChairWatch,
    MegaIPOItem,
    MegaIPOWatch,
    PlumbingWatch,
    TariffTwinDeficitWatch,
)
from app.schemas.dashboard_state import DashboardState
from app.schemas.indicator_snapshot import IndicatorSnapshot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds (speaker-faithful anchor numbers)
# ---------------------------------------------------------------------------
_UNEMPLOYMENT_CLEAN_CUT_THRESHOLD = 5.0   # unemployment must be AT or ABOVE this
_INFLATION_CLEAN_CUT_THRESHOLD = 2.5      # core CPI YoY must be BELOW this
_DXY_PRESSURE_THRESHOLD = 100.0

# Mega-IPO statuses that indicate liquidity weakening
_IPO_WEAKENING_STATUSES = {"delayed", "weak_demand", "missed_target"}
_IPO_SUPPORTIVE_STATUSES = {"strong_demand", "completed"}


# ---------------------------------------------------------------------------
# Module 1: Mega-IPO Signal
# ---------------------------------------------------------------------------

def _build_mega_ipo_watch(config: dict[str, Any]) -> MegaIPOWatch:
    raw_items: list[dict] = config.get("mega_ipos", [])
    items = [
        MegaIPOItem(
            name=i.get("name", "Unknown"),
            status=i.get("status", "unknown"),
            target_valuation=i.get("target_valuation"),
            notes=i.get("notes"),
        )
        for i in raw_items
        if isinstance(i, dict)
    ]

    weakening = any(item.status in _IPO_WEAKENING_STATUSES for item in items)
    supportive = (
        not weakening
        and any(item.status in _IPO_SUPPORTIVE_STATUSES for item in items)
    )

    if weakening:
        overall_signal = "Weakening"
        playbook_impact = (
            "AI thematic liquidity may be softening; treat speculative "
            "enthusiasm with more caution."
        )
    elif supportive:
        overall_signal = "Supportive"
        playbook_impact = "Risk appetite remains broad enough to support thematic issuance."
    else:
        overall_signal = "Neutral"
        playbook_impact = "No clean signal yet from upcoming issuance."

    return MegaIPOWatch(
        items=items,
        overall_signal=overall_signal,
        why_it_matters=(
            "The speaker treats mega-IPOs as a live test of whether market "
            "liquidity is deep enough to absorb new trillion-dollar-scale narratives."
        ),
        playbook_impact=playbook_impact,
        provenance="config-backed",
    )


# ---------------------------------------------------------------------------
# Module 2: Fed Chair Bias
# ---------------------------------------------------------------------------

def _build_fed_chair_watch(config: dict[str, Any]) -> FedChairWatch:
    raw_candidates: list[dict] = config.get("fed_chair", [])
    candidates = [
        FedChairCandidate(
            name=c.get("name", "Unknown"),
            status=c.get("status", "unknown"),
            tone=c.get("tone", "unknown"),
            notes=c.get("notes"),
        )
        for c in raw_candidates
        if isinstance(c, dict)
    ]

    # Determine current bias by front-runner precedence
    front_runners = [c for c in candidates if c.status == "front_runner"]

    current_bias = "Unknown / Mixed"
    playbook_impact = "No front-runner identified; Fed Chair transition risk remains open."

    if front_runners:
        fr = front_runners[0]
        if fr.tone == "bullish_fast_cuts":
            current_bias = "Bullish / Fast Cuts"
            playbook_impact = (
                f"{fr.name} as front-runner supports a faster-cut path and "
                "equity multiple expansion narrative."
            )
        elif fr.tone == "hawkish_inflation_wary":
            current_bias = "Hawkish / Inflation Wary"
            playbook_impact = (
                f"{fr.name} as front-runner suggests a more cautious easing path "
                "that limits how much rate-cut tailwind the market can price in."
            )
        else:
            current_bias = "Neutral / Unknown"
            playbook_impact = f"{fr.name} as front-runner; policy tone not yet clear."

    return FedChairWatch(
        candidates=candidates,
        current_bias=current_bias,
        why_it_matters=(
            "The speaker treats the next Fed Chair as a major determinant of "
            "the future cut path and the overall liquidity tone for equities."
        ),
        playbook_impact=playbook_impact,
        provenance="config-backed",
    )


# ---------------------------------------------------------------------------
# Module 3: Tariff / Twin Deficit / Dollar
# ---------------------------------------------------------------------------

def _build_tariff_watch(
    config: dict[str, Any],
    snapshot: IndicatorSnapshot,
) -> TariffTwinDeficitWatch:
    raw_tariffs: dict = config.get("tariffs", {})
    tariff_status: str = raw_tariffs.get("tariff_status", "unknown")
    tariff_pressure_active: bool = bool(raw_tariffs.get("tariff_pressure_active", False))
    notes: str | None = raw_tariffs.get("notes")

    dxy = snapshot.dollar_context.dxy
    dxy_pressure = (dxy is not None and dxy > _DXY_PRESSURE_THRESHOLD)

    # Build playbook impact from combined signals
    if dxy_pressure and tariff_pressure_active:
        playbook_impact = (
            "Dollar and tariff pressure are both working against a clean "
            "disinflation path — inflation may stay stickier even as growth softens."
        )
    elif dxy_pressure:
        playbook_impact = (
            "Dollar strength is a macro headwind. Watch for export drag and "
            "twin-deficit pressure to complicate the growth picture."
        )
    elif tariff_pressure_active:
        playbook_impact = (
            "Tariff pass-through risk can keep inflation sticky even if growth "
            "weakens — a direct complication for the stagflation trap."
        )
    else:
        playbook_impact = (
            "No active dollar or tariff pressure signal at this time. "
            "Continue monitoring for escalation."
        )

    return TariffTwinDeficitWatch(
        dxy=dxy,
        dxy_pressure=dxy_pressure,
        tariff_status=tariff_status,
        tariff_pressure_active=tariff_pressure_active,
        notes=notes,
        why_it_matters=(
            "The speaker links a strong dollar and tariff escalation to export "
            "pressure, twin-deficit stress, and inflation pass-through that can "
            "complicate or prevent clean rate cuts."
        ),
        playbook_impact=playbook_impact,
        provenance="mixed",   # DXY from live data; tariff status from config
    )


# ---------------------------------------------------------------------------
# Module 4: Clean-Cut Watch
# ---------------------------------------------------------------------------

def _build_clean_cut_watch(snapshot: IndicatorSnapshot) -> CleanCutWatch:
    unemployment = snapshot.growth.unemployment_rate
    cpi_yoy = snapshot.inflation.core_cpi_yoy

    labor_met = (
        unemployment is not None
        and unemployment >= _UNEMPLOYMENT_CLEAN_CUT_THRESHOLD
    )
    inflation_met = (
        cpi_yoy is not None
        and cpi_yoy < _INFLATION_CLEAN_CUT_THRESHOLD
    )
    window_open = labor_met and inflation_met

    if window_open:
        why_not_open = None
        playbook_impact = (
            "Rate cuts would be easier to justify without immediately reviving "
            "inflation fears. A clean easing path is opening."
        )
    else:
        missing_parts = []
        if not labor_met:
            if unemployment is None:
                missing_parts.append("unemployment data unavailable")
            else:
                missing_parts.append(
                    f"unemployment at {unemployment:.1f}% — "
                    f"not yet at the {_UNEMPLOYMENT_CLEAN_CUT_THRESHOLD:.1f}% threshold"
                )
        if not inflation_met:
            if cpi_yoy is None:
                missing_parts.append("core CPI data unavailable")
            else:
                missing_parts.append(
                    f"core CPI at {cpi_yoy:.1f}% — "
                    f"not yet below the {_INFLATION_CLEAN_CUT_THRESHOLD:.1f}% threshold"
                )
        why_not_open = "; ".join(missing_parts)
        playbook_impact = (
            "The market may still hope for cuts, but the macro backdrop is not "
            "yet clean. Any cuts risk being seen as inflation-tolerant rather than "
            "economy-driven."
        )

    return CleanCutWatch(
        clean_cut_window_open=window_open,
        unemployment_condition_met=labor_met,
        inflation_condition_met=inflation_met,
        why_not_open=why_not_open,
        why_it_matters=(
            "The speaker argues the Fed only has room for truly clean cuts when "
            "labor slack is meaningful (unemployment near 5%) and inflation has "
            "cooled enough to preserve credibility (core CPI below ~2.5%)."
        ),
        playbook_impact=playbook_impact,
        provenance="live-linked",   # Both conditions derived from live snapshot data
    )


# ---------------------------------------------------------------------------
# Module 5: Plumbing Watch
# ---------------------------------------------------------------------------

def _build_plumbing_watch(config: dict[str, Any]) -> PlumbingWatch:
    raw: dict = config.get("plumbing", {})
    repo_status = raw.get("repo_status", "unknown")
    reverse_repo_status = raw.get("reverse_repo_status", "unknown")
    reserve_status = raw.get("reserve_status", "unknown")
    notes: str | None = raw.get("notes")

    all_statuses = [repo_status, reverse_repo_status, reserve_status]

    if "stress" in all_statuses:
        stress_label = "Stress"
        playbook_impact = (
            "Liquidity plumbing may be strained even if headline policy has "
            "not changed. Treat emergency operations as a stress signal, not "
            "as bullish QE."
        )
    elif "watch" in all_statuses:
        stress_label = "Watch"
        playbook_impact = (
            "Keep an eye on funding conditions and reserve stability. "
            "Plumbing tension can tighten financial conditions quietly."
        )
    elif all(s == "unknown" for s in all_statuses):
        stress_label = "Unknown"
        playbook_impact = (
            "Plumbing data not yet configured. Update catalysts.json with "
            "repo/reserve conditions as needed."
        )
    else:
        stress_label = "Normal"
        playbook_impact = "No visible plumbing stress from configured inputs."

    return PlumbingWatch(
        repo_status=repo_status,
        reverse_repo_status=reverse_repo_status,
        reserve_status=reserve_status,
        stress_label=stress_label,
        notes=notes,
        why_it_matters=(
            "The speaker distinguishes between true QE (genuine balance sheet "
            "expansion) and emergency plumbing operations (repo/reverse repo). "
            "The latter can signal stress rather than genuine bullish liquidity."
        ),
        playbook_impact=playbook_impact,
        provenance="config-backed",
    )


# ---------------------------------------------------------------------------
# Module 6: Catalyst Overlays
# ---------------------------------------------------------------------------

def _build_overlays(
    mega_ipos: MegaIPOWatch,
    fed_chair: FedChairWatch,
    tariff: TariffTwinDeficitWatch,
    clean_cut: CleanCutWatch,
    plumbing: PlumbingWatch,
) -> list[str]:
    overlays: list[str] = []

    if mega_ipos.overall_signal == "Weakening":
        overlays.append("AI Liquidity Test Weakening")
    elif mega_ipos.overall_signal == "Supportive":
        overlays.append("AI Liquidity Test Supportive")

    if "Fast Cuts" in fed_chair.current_bias:
        overlays.append("Fed Chair Bias: Fast Cuts")
    elif "Hawkish" in fed_chair.current_bias or "Inflation Wary" in fed_chair.current_bias:
        overlays.append("Fed Chair Bias: Inflation Wary")

    if tariff.dxy_pressure:
        overlays.append("Dollar Pressure")
    if tariff.tariff_pressure_active:
        overlays.append("Tariff Inflation Risk")

    if clean_cut.clean_cut_window_open:
        overlays.append("Clean Cut Window Open")
    else:
        overlays.append("Clean Cut Window Closed")

    if plumbing.stress_label == "Stress":
        overlays.append("Liquidity Plumbing Stress")

    return overlays


# ---------------------------------------------------------------------------
# Module 7: Next Lookouts (always exactly 3)
# ---------------------------------------------------------------------------

def _build_next_lookouts(
    mega_ipos: MegaIPOWatch,
    fed_chair: FedChairWatch,
    tariff: TariffTwinDeficitWatch,
    clean_cut: CleanCutWatch,
    plumbing: PlumbingWatch,
    state: DashboardState,
) -> list[str]:
    candidates: list[tuple[int, str]] = []

    # IPO lookout — prioritise if any item is non-neutral
    if mega_ipos.overall_signal == "Weakening":
        candidates.append((10, (
            "If IPO demand stays weak or delays continue, treat it as confirmation "
            "that thematic liquidity is thinning beneath the surface."
        )))
    elif mega_ipos.overall_signal == "Neutral":
        # Find the most prominent item
        primary = next(
            (i for i in mega_ipos.items if i.name == "SpaceX"),
            mega_ipos.items[0] if mega_ipos.items else None,
        )
        if primary:
            candidates.append((6, (
                f"If {primary.name} sees delays or weak demand, treat it as a warning "
                "that thematic liquidity is thinning."
            )))
        else:
            candidates.append((5, (
                "Watch mega-IPO reception: weak demand signals thinner real liquidity "
                "beneath the headline tape."
            )))
    else:
        candidates.append((4, (
            "Strong IPO reception is a positive liquidity signal — watch whether "
            "momentum continues or fades into the broader calendar."
        )))

    # Fed Chair lookout — always relevant
    no_front_runner = all(c.status != "front_runner" for c in fed_chair.candidates)
    if no_front_runner:
        candidates.append((9, (
            "If Kevin Hassett becomes the clear front-runner, expect markets to price "
            "a faster-cut path. A Waller or Warsh front-runner implies slower easing."
        )))
    else:
        front = next((c for c in fed_chair.candidates if c.status == "front_runner"), None)
        if front and front.tone == "bullish_fast_cuts":
            candidates.append((5, (
                f"Watch for any shift away from {front.name} — a change in the "
                "front-runner to a more hawkish candidate would reprice the cut path."
            )))
        else:
            candidates.append((5, (
                "A shift toward a dovish front-runner for Fed Chair would be a "
                "significant liquidity-supportive catalyst."
            )))

    # Clean-cut lookout — always relevant
    if not clean_cut.clean_cut_window_open:
        missing = []
        if not clean_cut.unemployment_condition_met:
            missing.append("unemployment rises toward 5%")
        if not clean_cut.inflation_condition_met:
            missing.append("core CPI falls below 2.5%")
        condition_str = " and ".join(missing) if missing else "conditions improve"
        candidates.append((8, (
            f"If {condition_str}, the clean-cut window begins reopening — "
            "watch for this as a signal that the Fed has credible room to ease."
        )))
    else:
        candidates.append((7, (
            "The clean-cut window is open — watch for any reversal (CPI re-acceleration "
            "or labor market tightening) that would close it again."
        )))

    # Tariff/DXY lookout
    if not tariff.dxy_pressure and not tariff.tariff_pressure_active:
        candidates.append((5, (
            "If DXY moves back above 100 alongside tariff escalation, inflation "
            "pressure risk rises — watch this combination carefully."
        )))
    elif tariff.tariff_pressure_active:
        candidates.append((7, (
            "If tariff escalation broadens, expect inflation pass-through to keep "
            "CPI stickier — complicating the path to clean cuts."
        )))

    # Plumbing lookout
    if plumbing.stress_label in ("Watch", "Unknown"):
        candidates.append((6, (
            "If repo or reserve stress shifts from watch to stress, treat it as a "
            "plumbing warning rather than bullish QE — liquidity conditions may be "
            "tighter than they appear."
        )))
    elif plumbing.stress_label == "Stress":
        candidates.append((9, (
            "Plumbing stress is active — distinguish between emergency stabilization "
            "operations and genuine liquidity expansion before interpreting as bullish."
        )))

    # Regime-specific bonus
    regime = state.primary_regime.lower()
    if "stagflation" in regime:
        candidates.append((9, (
            "In a stagflation regime, the key release to watch is monthly CPI — "
            "any decisive cooling would be the first signal that the trap is loosening."
        )))
    elif "transition" in regime:
        candidates.append((7, (
            "In a liquidity transition, watch the Fed balance sheet weekly — "
            "a clear turn from contraction to expansion would upgrade the regime."
        )))
    elif "crash" in regime or "stress" in regime:
        candidates.append((10, (
            "With stress gauges elevated, watch NPL trends and bank reserve data — "
            "any acceleration is a warning to reduce exposure to lower-quality assets."
        )))

    # Sort by priority (descending) and take top 3
    candidates.sort(key=lambda x: x[0], reverse=True)
    top3 = [msg for _, msg in candidates[:3]]

    # Deterministic fill if somehow fewer than 3
    fallbacks = [
        "Watch monthly CPI prints for any decisive cooling below 3% — that is the primary inflation unlock.",
        "Watch the Fed balance sheet weekly for any shift from contraction to genuine expansion.",
        "Watch labor market data — a rise in unemployment toward 5% reopens the clean-cut path.",
    ]
    while len(top3) < 3:
        fb = fallbacks[len(top3) % len(fallbacks)]
        if fb not in top3:
            top3.append(fb)
        else:
            top3.append(fallbacks[(len(top3) + 1) % len(fallbacks)])

    return top3[:3]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_catalyst_state(
    config: dict[str, Any],
    snapshot: IndicatorSnapshot,
    state: DashboardState,
) -> CatalystState:
    """
    Build a fully typed CatalystState from:
      - raw catalyst config dict
      - current IndicatorSnapshot
      - current DashboardState

    Pure, synchronous, no side effects.
    """
    try:
        mega_ipos = _build_mega_ipo_watch(config)
        fed_chair = _build_fed_chair_watch(config)
        tariff = _build_tariff_watch(config, snapshot)
        clean_cut = _build_clean_cut_watch(snapshot)
        plumbing = _build_plumbing_watch(config)

        overlays = _build_overlays(mega_ipos, fed_chair, tariff, clean_cut, plumbing)
        next_lookouts = _build_next_lookouts(
            mega_ipos, fed_chair, tariff, clean_cut, plumbing, state
        )

        return CatalystState(
            mega_ipos=mega_ipos,
            fed_chair=fed_chair,
            tariff_twin_deficit=tariff,
            clean_cut_watch=clean_cut,
            plumbing_watch=plumbing,
            catalyst_overlays=overlays,
            next_lookouts=next_lookouts,
            updated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
    except Exception as exc:
        logger.error("build_catalyst_state failed: %s", exc)
        # Return empty-but-valid default rather than crashing the playbook pipeline
        return CatalystState(
            updated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
