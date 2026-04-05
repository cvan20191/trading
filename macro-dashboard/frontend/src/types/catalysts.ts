// ---------------------------------------------------------------------------
// Catalyst types — mirror backend app/schemas/catalysts.py exactly
// ---------------------------------------------------------------------------

export interface MegaIPOItem {
  name: string
  status: string   // unknown | rumored | on_track | delayed | weak_demand | strong_demand | missed_target | completed
  target_valuation: string | null
  notes: string | null
  impact_label: string | null
}

export interface MegaIPOWatch {
  items: MegaIPOItem[]
  overall_signal: string   // Weakening | Supportive | Neutral
  why_it_matters: string
  playbook_impact: string
  provenance: string       // config-backed | live-linked | mixed
}

export interface FedChairCandidate {
  name: string
  status: string   // front_runner | contender | unlikely | unknown
  tone: string     // bullish_fast_cuts | hawkish_inflation_wary | neutral | unknown
  notes: string | null
}

export interface FedChairWatch {
  candidates: FedChairCandidate[]
  current_bias: string
  why_it_matters: string
  playbook_impact: string
  provenance: string
}

export interface TariffTwinDeficitWatch {
  dxy: number | null
  dxy_pressure: boolean
  tariff_status: string   // inactive | watch | active | escalating | paused | unknown
  tariff_pressure_active: boolean
  notes: string | null
  why_it_matters: string
  playbook_impact: string
  provenance: string      // mixed — DXY is live-linked; tariff status is config-backed
}

export interface CleanCutWatch {
  clean_cut_window_open: boolean
  unemployment_condition_met: boolean
  inflation_condition_met: boolean
  why_not_open: string | null
  why_it_matters: string
  playbook_impact: string
  provenance: string   // live-linked
}

export interface PlumbingWatch {
  repo_status: string        // normal | watch | stress | unknown
  reverse_repo_status: string
  reserve_status: string
  stress_label: string       // Normal | Watch | Stress | Unknown
  notes: string | null
  why_it_matters: string
  playbook_impact: string
  provenance: string         // config-backed
}

export interface CatalystState {
  mega_ipos: MegaIPOWatch
  fed_chair: FedChairWatch
  tariff_twin_deficit: TariffTwinDeficitWatch
  clean_cut_watch: CleanCutWatch
  plumbing_watch: PlumbingWatch
  catalyst_overlays: string[]
  next_lookouts: string[]
  updated_at: string
}
