// PlaybookResponse — combined state + summary returned by POST /api/playbook
// LivePlaybookResponse — same, extended with source provenance from live endpoints

import type { DashboardState } from './summary'
import type { PlaybookSummary } from './summary'
import type { IndicatorSnapshot } from './indicator'
import type { CatalystState } from './catalysts'

export type { CatalystState }

export interface PlaybookConclusion {
  conclusion_label: string
  new_cash_action:
    | 'accumulate'
    | 'accumulate_selectively'
    | 'hold_and_wait'
    | 'pause_new_buying'
    | 'defensive_preservation'
  stock_archetype_preferred: Array<
    | 'hyper_growth_manageable_debt'
    | 'moderate_growth_moderate_leverage'
    | 'high_growth_refinancing_beneficiary'
    | 'defensive_low_debt_low_valuation'
    | 'profitable_cashflow_compounders'
    | 'balance_sheet_strength_priority'
    | 'high_debt_unprofitable_growth'
    | 'valuation_dependent_speculation'
    | 'deep_cyclical_balance_sheet_risk'
  >
  stock_archetype_avoid: Array<
    | 'hyper_growth_manageable_debt'
    | 'moderate_growth_moderate_leverage'
    | 'high_growth_refinancing_beneficiary'
    | 'defensive_low_debt_low_valuation'
    | 'profitable_cashflow_compounders'
    | 'balance_sheet_strength_priority'
    | 'high_debt_unprofitable_growth'
    | 'valuation_dependent_speculation'
    | 'deep_cyclical_balance_sheet_risk'
  >
  can_rally_despite_bad_news: boolean
  warning_urgency: 'cautionary' | 'elevated' | 'urgent'
  leniency_notes: Array<
    | 'valuation_proxy_not_true_forward_pe'
    | 'stress_gauges_warning_lights_not_timers'
    | 'transition_regime_can_overshoot'
    | 'stretched_means_pause_not_forced_sell'
    | 'mixed_signals_reduce_conviction'
  >
  why_now: string
}

export interface PlaybookResponse {
  state: DashboardState
  playbook_conclusion?: PlaybookConclusion
  summary: PlaybookSummary
  catalysts: CatalystState
}

// ---------------------------------------------------------------------------
// Source metadata (mirrors backend SourceMeta)
// ---------------------------------------------------------------------------
export interface SourceMeta {
  provider: string
  series_name: string
  series_id: string | null
  fetched_at: string
  observed_at: string | null
  frequency: string | null
  status: string   // fresh | stale | missing | error | fallback
  note: string | null
  // valuation-specific: forward | trailing | ttm_derived | unavailable | null (other metrics)
  basis: string | null
}

// ---------------------------------------------------------------------------
// Live response types (mirrors backend LiveSnapshotResponse / LivePlaybookResponse)
// ---------------------------------------------------------------------------
export interface LiveSnapshotResponse {
  snapshot: IndicatorSnapshot
  sources: Record<string, SourceMeta>
  overall_status: string
  stale_series: string[]
  generated_at: string
}

export interface LivePlaybookResponse {
  snapshot: IndicatorSnapshot
  state: DashboardState
  playbook_conclusion?: PlaybookConclusion
  summary: PlaybookSummary
  catalysts: CatalystState
  sources: Record<string, SourceMeta>
  overall_status: string
  stale_series: string[]
  generated_at: string
}
