"""
Top Watchpoints Ranker — Module 8.

Scores a deterministic candidate list and returns the top 3.
"""

from __future__ import annotations

from app.services.rules.chessboard import ChessboardResult
from app.services.rules.stagflation import StagflationResult
from app.services.rules.stress import DollarResult, StressResult
from app.services.rules.valuation import ValuationResult


def compute_watchpoints(
    cb: ChessboardResult,
    stag: StagflationResult,
    val: ValuationResult,
    stress: StressResult,
    dollar: DollarResult,
    regime: str,
) -> list[str]:
    """Return exactly 3 watchpoint strings ranked by current relevance."""
    candidates: list[tuple[int, str]] = []

    # ── WTI Oil ──────────────────────────────────────────────────────────────
    wti = stag.trap.wti_oil
    if wti is not None:
        if stag.oil_risk_active:
            candidates.append((90, f"WTI crude is holding above $100 — inflation-risk zone active"))
        elif wti >= 95:
            candidates.append((60, f"WTI crude near $100 threshold at ${wti:.0f}"))
        else:
            candidates.append((20, f"WTI crude at ${wti:.0f} — below inflation-risk threshold"))

    # ── Shelter / Services CPI ───────────────────────────────────────────────
    if (stag.trap.shelter_status or "").lower() == "sticky":
        candidates.append((85, "Shelter CPI is not rolling over — services inflation remains sticky"))
    if (stag.trap.services_ex_energy_status or "").lower() == "sticky":
        candidates.append((80, "Services ex-energy inflation is still elevated and not easing"))

    # ── Forward Valuation ────────────────────────────────────────────────────
    pe = val.valuation.forward_pe
    if pe is not None:
        _is_proxy = val.valuation.is_fallback
        _pe_label = "Proxy valuation" if _is_proxy else "Forward big-tech P/E"
        _proxy_suffix = " (directional proxy — apply softer interpretation)" if _is_proxy else ""
        if val.is_stretched:
            candidates.append((88, f"{_pe_label} at {pe:.1f}x — above the speaker's pause threshold{_proxy_suffix}"))
        elif 25 < pe < 30:
            candidates.append((55, f"{_pe_label} at {pe:.1f}x — approaching the stretched zone{_proxy_suffix}"))
        elif val.is_buy_zone:
            candidates.append((40, f"{_pe_label} at {pe:.1f}x — inside the historical accumulation zone{_proxy_suffix}"))

    # ── Fed Balance Sheet ─────────────────────────────────────────────────────
    bs = cb.chessboard.balance_sheet_trend_1m
    if bs == "down":
        candidates.append((75, "Fed balance sheet is still contracting — net liquidity withdrawal ongoing"))
    elif bs == "flat":
        candidates.append((55, "Fed balance sheet is flat — contraction has paused but not reversed"))
    elif bs == "up":
        candidates.append((30, "Fed balance sheet is expanding — liquidity is being injected"))

    # ── Unemployment ─────────────────────────────────────────────────────────
    unemp = stag.trap.unemployment_rate
    if unemp is not None:
        if unemp > 4.3:
            candidates.append((70, f"Unemployment at {unemp:.1f}% — rising above the Fed's tolerated range"))
        elif 4.0 <= unemp <= 4.3:
            candidates.append((65, f"Unemployment at {unemp:.1f}% — in the trap band; not weak enough to justify easy cuts"))

    # ── Core CPI ─────────────────────────────────────────────────────────────
    cpi = stag.trap.core_cpi_yoy
    if cpi is not None:
        if cpi > 3.0:
            candidates.append((78, f"Core CPI at {cpi:.1f}% YoY — above the speaker's sticky threshold"))
        elif 2.5 <= cpi <= 3.0:
            candidates.append((50, f"Core CPI at {cpi:.1f}% — approaching the sticky threshold"))

    # ── NPL Ratio ────────────────────────────────────────────────────────────
    npl = stress.stress.npl_ratio
    if npl is not None:
        if stress.stress.npl_zone == "Warning":
            candidates.append((82, f"Bank NPL ratio at {npl:.2f}% — above systemic warning threshold"))
        elif stress.stress.npl_zone == "Caution":
            candidates.append((58, f"Bank NPL ratio at {npl:.2f}% — entering caution band"))

    # ── Market Cap / M2 ──────────────────────────────────────────────────────
    mcm2 = stress.stress.market_cap_m2_ratio
    if mcm2 is not None:
        if stress.stress.market_cap_m2_zone == "Extreme":
            candidates.append((88, f"Market Cap/M2 at {mcm2:.2f} — in extreme froth territory"))
        elif stress.stress.market_cap_m2_zone == "Warning":
            candidates.append((65, f"Market Cap/M2 at {mcm2:.2f} — above the overleveraged warning level"))

    # ── Yield Curve ──────────────────────────────────────────────────────────
    yc = stress.stress.yield_curve_value
    if yc is not None and stress.stress.yield_curve_inverted:
        candidates.append((72, f"Yield curve inverted at {yc:.2f}% — recession-watch signal active"))

    # ── DXY ──────────────────────────────────────────────────────────────────
    dxy = dollar.dollar.dxy
    if dxy is not None and dollar.dxy_pressure:
        candidates.append((48, f"DXY at {dxy:.1f} — strong dollar adding macro friction"))

    # ── Chessboard direction ─────────────────────────────────────────────────
    if cb.liquidity_tight and cb.quadrant == "D":
        candidates.append((70, "Fed Chessboard is in max illiquidity — rates elevated, balance sheet contracting"))

    # Sort descending, deduplicate, take top 3
    candidates.sort(key=lambda x: x[0], reverse=True)

    seen_starts: set[str] = set()
    unique: list[str] = []
    for _, text in candidates:
        key = text[:30]
        if key not in seen_starts:
            seen_starts.add(key)
            unique.append(text)
        if len(unique) == 3:
            break

    # Pad to exactly 3 if fewer candidates exist
    fallbacks = [
        "Monitor the Fed Chessboard for any shift in rate or balance sheet direction",
        "Watch Core CPI for signs of rolling over toward the 2.5% zone",
        "Track forward big-tech P/E for compression into the buy-zone range",
    ]
    for fb in fallbacks:
        if len(unique) >= 3:
            break
        unique.append(fb)

    return unique[:3]
