// Replay Lab types — mirror backend replay.py schemas exactly.
// Note: no `catalysts` field — build_catalyst_state() is not date-aware
// and is intentionally omitted from replay responses.

import type { DashboardState, PlaybookSummary } from './summary'
import type { IndicatorSnapshot } from './indicator'
import type { PlaybookConclusion, SourceMeta } from './playbook'

export interface ContextNote {
  title: string
  body: string
  tags: string[]
}

export interface OutcomePoint {
  date_end: string
  // SPY (SPDR S&P 500 ETF)
  spy_return_pct: number | null
  spy_price_start: number | null
  spy_price_end: number | null
  // QQQ (Invesco Nasdaq-100 ETF)
  qqq_return_pct: number | null
  qqq_price_start: number | null
  qqq_price_end: number | null
  // WTI crude (CL=F)
  wti_change_pct: number | null
  wti_price_start: number | null
  wti_price_end: number | null
}

export interface OutcomeReview {
  as_of: string
  outcomes_1w: OutcomePoint
  outcomes_1m: OutcomePoint
  outcomes_3m: OutcomePoint
  data_note: string
}

export interface ReplayPlaybookResponse {
  as_of: string
  snapshot: IndicatorSnapshot
  state: DashboardState
  playbook_conclusion?: PlaybookConclusion
  summary: PlaybookSummary
  sources: Record<string, SourceMeta>
  context: ContextNote[]
  data_notes: string[]
  generated_at: string
}

// Local-only type — stored in localStorage, never sent to the server
export interface ReplayAssessment {
  date: string                            // YYYY-MM-DD
  regime: string
  posture: string
  watchpoints: [string, string, string]
  confidence: 'High' | 'Medium' | 'Low'
  notes: string
  saved_at: string                        // ISO timestamp
  revealed: boolean
}
