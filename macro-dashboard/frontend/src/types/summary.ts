// ---------------------------------------------------------------------------
// Response types (mirror backend Pydantic models exactly)
// ---------------------------------------------------------------------------

export interface SummaryMeta {
  used_fallback: boolean
  generated_at: string
  model: string | null
  data_status: string
}

export interface PlaybookSummary {
  headline_summary: string
  expanded_summary: string
  regime_label: string
  posture_label: string
  watch_now: [string, string, string]
  what_changed_bullets: [string, string, string]
  what_changes_call_bullets: [string, string, string]
  risk_flags: string[]
  teaching_note: string
  meta: SummaryMeta
}

// ---------------------------------------------------------------------------
// Request types (mirror backend DashboardState input schema)
// ---------------------------------------------------------------------------

export interface DataFreshness {
  overall_status: string
  stale_series: string[]
}

export interface FedChessboard {
  quadrant?: string
  label?: string
  rate_trend_1m?: string
  rate_trend_3m?: string
  balance_sheet_trend_1m?: string
  balance_sheet_trend_3m?: string
  direction_vs_1m_ago?: string
  // Two-layer chessboard fields — policy stance refactor
  policy_stance?: string          // "easy" | "middle" | "restrictive"
  rate_impulse?: string           // "easing" | "stable" | "tightening"
  balance_sheet_direction?: string // "expanding" | "flat_or_mixed" | "contracting"
  transition_tag?: string         // "Improving" | "Stable" | "Deteriorating"
}

export interface StagflationTrap {
  active: boolean
  growth_weakening: boolean
  sticky_inflation: boolean
  pmi_manufacturing?: number
  pmi_services?: number
  unemployment_rate?: number
  core_cpi_yoy?: number
  shelter_status?: string
  services_ex_energy_status?: string
  wti_oil?: number
  oil_risk_active: boolean
}

export interface ValuationConstituent {
  ticker: string
  price?: number
  forward_eps?: number
  forward_pe?: number
}

export interface Valuation {
  forward_pe?: number
  zone?: string
  zone_label?: string
  buy_zone_low?: number
  buy_zone_high?: number
  pause_threshold?: number
  // basis metadata — forward | trailing | ttm_derived | unavailable
  basis?: string
  basis_label?: string
  source_note?: string
  is_fallback?: boolean
  // metric identity
  metric_name?: string       // e.g. "Mag 7 Forward P/E" or "QQQ P/E Proxy"
  object_label?: string      // e.g. "Mag 7 Basket" or "QQQ (QQQ)"
  provider?: string          // e.g. "fmp" or "yahoo"
  coverage_count?: number    // number of basket constituents used
  coverage_ratio?: number    // fraction of Mag 7 market cap included
  // per-ticker breakdown — only populated for FMP Mag 7 basket
  constituents?: ValuationConstituent[]
}

export interface SystemicStress {
  yield_curve_inverted: boolean
  yield_curve_value?: number
  npl_ratio?: number
  npl_zone?: string
  market_cap_m2_ratio?: number
  market_cap_m2_zone?: string
}

export interface DollarContext {
  dxy?: number
  dxy_pressure: boolean
}

export interface RallyConditions {
  rally_fuel_score?: number
  fed_put: boolean
  treasury_put: boolean
  political_put: boolean
  market_ignoring_bad_news: boolean
}

export interface DashboardState {
  as_of?: string
  data_freshness: DataFreshness
  primary_regime: string
  secondary_overlays: string[]
  confidence: string
  current_posture: string
  fed_chessboard?: FedChessboard
  stagflation_trap?: StagflationTrap
  valuation?: Valuation
  systemic_stress?: SystemicStress
  dollar_context?: DollarContext
  rally_conditions?: RallyConditions
  top_watchpoints: string[]
  what_changed: string[]
  what_changes_call: string[]
}
