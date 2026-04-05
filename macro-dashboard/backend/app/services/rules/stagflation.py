"""
Stagflation Trap Monitor — Module 2.

Determines growth_weakening, sticky_inflation, and the full trap trigger.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.dashboard_state import StagflationTrap
from app.schemas.indicator_snapshot import GrowthInput, InflationInput

# Speaker thresholds
_CPI_STICKY_THRESHOLD = 3.0
_OIL_RISK_THRESHOLD = 100.0
_PMI_CONTRACTION = 50.0
_UNEMP_TRAP_LOW = 4.0
_UNEMP_TRAP_HIGH = 4.3


@dataclass
class StagflationResult:
    trap: StagflationTrap
    growth_weakening: bool
    sticky_inflation: bool
    oil_risk_active: bool


def compute_stagflation(
    growth: GrowthInput, inflation: InflationInput
) -> StagflationResult:
    pmi_mfg = growth.pmi_manufacturing
    pmi_svc = growth.pmi_services
    unemp = growth.unemployment_rate
    unemp_trend = (growth.unemployment_trend or "").lower()
    claims_trend = (growth.initial_claims_trend or "").lower()
    payrolls_trend = (growth.payrolls_trend or "").lower()

    # ── Growth Weakening ────────────────────────────────────────────────────
    # Primary: manufacturing PMI below 50
    mfg_weak = pmi_mfg is not None and pmi_mfg < _PMI_CONTRACTION

    # At least one secondary signal
    svc_softening = pmi_svc is not None and pmi_svc < _PMI_CONTRACTION + 1.0
    labor_softening = any([
        unemp_trend == "up",
        claims_trend == "up",
        payrolls_trend == "down",
    ])

    growth_weakening = mfg_weak and (svc_softening or labor_softening)

    # ── Oil Risk ────────────────────────────────────────────────────────────
    if inflation.oil_risk_active is not None:
        oil_risk_active = inflation.oil_risk_active
    elif inflation.wti_oil is not None:
        oil_risk_active = inflation.wti_oil >= _OIL_RISK_THRESHOLD
    else:
        oil_risk_active = False

    # ── Sticky Inflation ────────────────────────────────────────────────────
    cpi_elevated = (
        inflation.core_cpi_yoy is not None
        and inflation.core_cpi_yoy > _CPI_STICKY_THRESHOLD
    )
    shelter_sticky = (inflation.shelter_status or "").lower() == "sticky"
    svc_ex_sticky = (inflation.services_ex_energy_status or "").lower() == "sticky"

    sticky_inflation = cpi_elevated and any([
        shelter_sticky,
        svc_ex_sticky,
        oil_risk_active,
        inflation.wti_oil is not None and inflation.wti_oil >= _OIL_RISK_THRESHOLD,
    ])

    # ── Full Trap ────────────────────────────────────────────────────────────
    # Requires all three: PMI contracting, unemployment in low-but-not-cracked band,
    # CPI above threshold
    unemp_in_trap_band = (
        unemp is not None
        and _UNEMP_TRAP_LOW <= unemp <= _UNEMP_TRAP_HIGH
    )
    trap_active = (
        mfg_weak
        and unemp_in_trap_band
        and cpi_elevated
    )

    trap = StagflationTrap(
        active=trap_active,
        growth_weakening=growth_weakening,
        sticky_inflation=sticky_inflation,
        pmi_manufacturing=pmi_mfg,
        pmi_services=pmi_svc,
        unemployment_rate=unemp,
        core_cpi_yoy=inflation.core_cpi_yoy,
        shelter_status=inflation.shelter_status,
        services_ex_energy_status=inflation.services_ex_energy_status,
        wti_oil=inflation.wti_oil,
        oil_risk_active=oil_risk_active,
    )

    return StagflationResult(
        trap=trap,
        growth_weakening=growth_weakening,
        sticky_inflation=sticky_inflation,
        oil_risk_active=oil_risk_active,
    )
