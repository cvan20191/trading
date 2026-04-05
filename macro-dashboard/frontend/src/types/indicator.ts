// Raw indicator input types — mirror backend IndicatorSnapshot schema

export interface DataFreshnessInput {
  overall_status?: string
  stale_series?: string[]
}

export interface LiquidityInput {
  fed_funds_rate?: number
  rate_trend_1m?: string
  rate_trend_3m?: string
  balance_sheet_assets?: number
  balance_sheet_trend_1m?: string
  balance_sheet_trend_3m?: string
}

export interface GrowthInput {
  pmi_manufacturing?: number
  pmi_services?: number
  unemployment_rate?: number
  unemployment_trend?: string
  initial_claims_trend?: string
  payrolls_trend?: string
}

export interface InflationInput {
  core_cpi_yoy?: number
  core_cpi_mom?: number
  shelter_status?: string
  services_ex_energy_status?: string
  wti_oil?: number
  oil_risk_active?: boolean
}

export interface ValuationInput {
  forward_pe?: number
  basis?: string
  basis_label?: string
  pe_basis?: string
  source_note?: string
  pe_source_note?: string
  is_fallback?: boolean
  metric_name?: string
  object_label?: string
  provider?: string
  pe_provider?: string
  coverage_count?: number
  coverage_ratio?: number
}

export interface SystemicStressInput {
  yield_curve_10y_2y?: number
  npl_ratio?: number
  market_cap_m2_ratio?: number
}

export interface DollarContextInput {
  dxy?: number
}

export interface PolicySupportInput {
  fed_put?: boolean
  treasury_put?: boolean
  political_put?: boolean
}

export interface IndicatorSnapshot {
  as_of?: string
  data_freshness: DataFreshnessInput
  liquidity: LiquidityInput
  growth: GrowthInput
  inflation: InflationInput
  valuation: ValuationInput
  systemic_stress: SystemicStressInput
  dollar_context: DollarContextInput
  policy_support: PolicySupportInput
}
