from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

NewCashAction = Literal[
    "accumulate",
    "accumulate_selectively",
    "hold_and_wait",
    "pause_new_buying",
    "defensive_preservation",
]

WarningUrgency = Literal[
    "cautionary",
    "elevated",
    "urgent",
]

ArchetypeLabel = Literal[
    "hyper_growth_manageable_debt",
    "moderate_growth_moderate_leverage",
    "high_growth_refinancing_beneficiary",
    "defensive_low_debt_low_valuation",
    "profitable_cashflow_compounders",
    "balance_sheet_strength_priority",
    "high_debt_unprofitable_growth",
    "valuation_dependent_speculation",
    "deep_cyclical_balance_sheet_risk",
]

LeniencyNote = Literal[
    "valuation_proxy_not_true_forward_pe",
    "stress_gauges_warning_lights_not_timers",
    "transition_regime_can_overshoot",
    "stretched_means_pause_not_forced_sell",
    "mixed_signals_reduce_conviction",
]


class PlaybookConclusion(BaseModel):
    conclusion_label: str
    new_cash_action: NewCashAction
    stock_archetype_preferred: list[ArchetypeLabel] = Field(default_factory=list)
    stock_archetype_avoid: list[ArchetypeLabel] = Field(default_factory=list)
    can_rally_despite_bad_news: bool = False
    warning_urgency: WarningUrgency
    leniency_notes: list[LeniencyNote] = Field(default_factory=list)
    why_now: str
