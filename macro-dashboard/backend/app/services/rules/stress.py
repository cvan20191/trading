"""
Systemic Stress + Dollar Context — Modules 4 & 5.

Crash-gauge style warnings: yield curve, NPL, Market Cap / M2, DXY.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.dashboard_state import DollarContext, SystemicStress
from app.schemas.indicator_snapshot import DollarContextInput, SystemicStressInput

# Speaker thresholds
_NPL_CAUTION = 1.0
_NPL_WARNING = 1.5
_MCM2_WARNING = 2.0
_MCM2_EXTREME = 3.0
_DXY_PRESSURE = 100.0


@dataclass
class StressResult:
    stress: SystemicStress
    stress_warning_active: bool
    stress_severe: bool


@dataclass
class DollarResult:
    dollar: DollarContext
    dxy_pressure: bool


def compute_stress(s: SystemicStressInput) -> StressResult:
    # Yield curve
    yc = s.yield_curve_10y_2y
    inverted = yc is not None and yc < 0.0

    # NPL zone
    npl = s.npl_ratio
    if npl is None:
        npl_zone = None
    elif npl < _NPL_CAUTION:
        npl_zone = "Normal"
    elif npl < _NPL_WARNING:
        npl_zone = "Caution"
    else:
        npl_zone = "Warning"

    # Market Cap / M2 zone
    mcm2 = s.market_cap_m2_ratio
    if mcm2 is None:
        mcm2_zone = None
    elif mcm2 < _MCM2_WARNING:
        mcm2_zone = "Normal"
    elif mcm2 < _MCM2_EXTREME:
        mcm2_zone = "Warning"
    else:
        mcm2_zone = "Extreme"

    # Helper booleans
    stress_warning_active = any([
        inverted,
        npl_zone in {"Caution", "Warning"},
        mcm2_zone in {"Warning", "Extreme"},
    ])
    stress_severe = any([
        npl_zone == "Warning",
        mcm2_zone == "Extreme",
    ])

    return StressResult(
        stress=SystemicStress(
            yield_curve_inverted=inverted,
            yield_curve_value=yc,
            npl_ratio=npl,
            npl_zone=npl_zone,
            market_cap_m2_ratio=mcm2,
            market_cap_m2_zone=mcm2_zone,
        ),
        stress_warning_active=stress_warning_active,
        stress_severe=stress_severe,
    )


def compute_dollar(d: DollarContextInput) -> DollarResult:
    dxy = d.dxy
    pressure = dxy is not None and dxy > _DXY_PRESSURE
    return DollarResult(
        dollar=DollarContext(dxy=dxy, dxy_pressure=pressure),
        dxy_pressure=pressure,
    )
