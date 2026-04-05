import { useState, useCallback, useEffect, useRef } from 'react'
import { fetchPlaybook, fetchLivePlaybook, triggerManualRefresh } from '../api/playbook'
import type { IndicatorSnapshot } from '../types/indicator'
import type { CatalystState, LivePlaybookResponse } from '../types/playbook'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type AppMode = 'live' | 'mock'
export type LoadStatus = 'idle' | 'loading' | 'success' | 'error'

export interface PlaybookData {
  snapshot?: LivePlaybookResponse['snapshot']
  state: LivePlaybookResponse['state']
  playbook_conclusion?: LivePlaybookResponse['playbook_conclusion']
  summary: LivePlaybookResponse['summary']
  catalysts?: CatalystState
  sources?: LivePlaybookResponse['sources']
  overall_status?: string
  stale_series?: string[]
  generated_at?: string
}

export interface UsePlaybookDataReturn {
  mode: AppMode
  setMode: (m: AppMode) => void
  pmiMfgOverride: number | null
  pmiSvcOverride: number | null
  setPmiMfgOverride: (v: number | null) => void
  setPmiSvcOverride: (v: number | null) => void
  status: LoadStatus
  data: PlaybookData | null
  errorMessage: string
  refreshing: boolean
  showDebug: boolean
  setShowDebug: (v: boolean) => void
  refresh: () => Promise<void>
  reset: () => void
}

// ---------------------------------------------------------------------------
// Mock snapshot (dev fallback)
// ---------------------------------------------------------------------------

const MOCK_SNAPSHOT: IndicatorSnapshot = {
  as_of: '2026-02-14T08:30:00Z',
  data_freshness: { overall_status: 'fresh', stale_series: [] },
  liquidity: {
    fed_funds_rate: 4.75, rate_trend_1m: 'down', rate_trend_3m: 'down',
    balance_sheet_assets: 7_200_000_000_000, balance_sheet_trend_1m: 'down', balance_sheet_trend_3m: 'flat',
  },
  growth: {
    pmi_manufacturing: 48.9, pmi_services: 50.1, unemployment_rate: 4.2,
    unemployment_trend: 'flat', initial_claims_trend: 'up', payrolls_trend: 'down',
  },
  inflation: {
    core_cpi_yoy: 3.3, core_cpi_mom: 0.3, shelter_status: 'sticky',
    services_ex_energy_status: 'sticky', wti_oil: 101.4, oil_risk_active: true,
  },
  valuation: {
    forward_pe: 30.8,
    basis: 'forward',
    basis_label: 'Forward P/E',
    source_note: 'Mag 7 market-cap-weighted forward P/E — 7/7 constituents, 100% market-cap coverage (demo)',
    is_fallback: false,
    metric_name: 'Mag 7 Forward P/E',
    object_label: 'Mag 7 Basket',
    provider: 'fmp',
    coverage_count: 7,
    coverage_ratio: 1.0,
  },
  systemic_stress: { yield_curve_10y_2y: -0.42, npl_ratio: 1.1, market_cap_m2_ratio: 2.35 },
  dollar_context: { dxy: 101.2 },
  policy_support: { fed_put: true, treasury_put: false, political_put: true },
}

const DOCTRINE_FIXTURE_A_BUYZONE_STRESS: IndicatorSnapshot = {
  as_of: '2026-03-29T15:00:00Z',
  data_freshness: { overall_status: 'fresh', stale_series: [] },
  liquidity: {
    // Force Quadrant A in rules: easy policy + expanding balance sheet.
    fed_funds_rate: 0.25, rate_trend_1m: 'down', rate_trend_3m: 'down',
    balance_sheet_assets: 7_800_000_000_000, balance_sheet_trend_1m: 'up', balance_sheet_trend_3m: 'up',
  },
  growth: {
    pmi_manufacturing: 51.0, pmi_services: 52.0, unemployment_rate: 4.2,
    unemployment_trend: 'flat', initial_claims_trend: 'flat', payrolls_trend: 'flat',
  },
  inflation: {
    core_cpi_yoy: 2.8, core_cpi_mom: 0.2, shelter_status: 'rising',
    services_ex_energy_status: 'rising', wti_oil: 92.0, oil_risk_active: false,
  },
  valuation: {
    forward_pe: 23.0,
    basis: 'forward',
    basis_label: 'Forward P/E',
    source_note: 'Synthetic doctrine fixture (manual test): A + buy-zone + severe-stress overlay',
    is_fallback: false,
    metric_name: 'Mag 7 Forward P/E',
    object_label: 'Mag 7 Basket',
    provider: 'fixture',
    coverage_count: 7,
    coverage_ratio: 1.0,
  },
  systemic_stress: { yield_curve_10y_2y: 0.56, npl_ratio: 1.57, market_cap_m2_ratio: 2.51 },
  dollar_context: { dxy: 102.0 },
  // Weak-to-mixed backstops (not strong): keeps rally condition-sensitive.
  policy_support: { fed_put: true, treasury_put: false, political_put: false },
}

const MOCK_FIXTURE_QUERY_KEY = 'mock_fixture'
const DOCTRINE_FIXTURE_ID = 'a_buyzone_stress'

function resolveMockSnapshot(): IndicatorSnapshot {
  if (typeof window === 'undefined') return MOCK_SNAPSHOT
  const fixture = new URLSearchParams(window.location.search).get(MOCK_FIXTURE_QUERY_KEY)
  if (fixture === DOCTRINE_FIXTURE_ID) {
    // Dev/manual synthetic fixture path. Does not affect live mode.
    return DOCTRINE_FIXTURE_A_BUYZONE_STRESS
  }
  return MOCK_SNAPSHOT
}

const PMI_MFG_KEY = 'pmi_mfg_override'
const PMI_SVC_KEY = 'pmi_svc_override'
const PMI_MIN = 30
const PMI_MAX = 70

function readStoredPmi(key: string): number | null {
  if (typeof window === 'undefined') return null
  const raw = window.localStorage.getItem(key)
  if (!raw) return null
  const parsed = Number(raw)
  return Number.isFinite(parsed) && parsed >= PMI_MIN && parsed <= PMI_MAX ? parsed : null
}

function normalizePmi(value: number | null): number | null {
  if (value === null) return null
  return Number.isFinite(value) && value >= PMI_MIN && value <= PMI_MAX ? value : null
}

function persistPmi(key: string, value: number | null): void {
  if (typeof window === 'undefined') return
  if (value === null) window.localStorage.removeItem(key)
  else window.localStorage.setItem(key, String(value))
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function usePlaybookData(): UsePlaybookDataReturn {
  const [mode, setMode] = useState<AppMode>('live')
  const [pmiMfgOverride, setPmiMfgOverrideState] = useState<number | null>(() => readStoredPmi(PMI_MFG_KEY))
  const [pmiSvcOverride, setPmiSvcOverrideState] = useState<number | null>(() => readStoredPmi(PMI_SVC_KEY))
  const [status, setStatus] = useState<LoadStatus>('idle')
  const [data, setData] = useState<PlaybookData | null>(null)
  const [errorMessage, setErrorMessage] = useState('')
  const [refreshing, setRefreshing] = useState(false)
  const [showDebug, setShowDebug] = useState(false)
  const mountedRef = useRef(true)

  useEffect(() => {
    mountedRef.current = true
    return () => { mountedRef.current = false }
  }, [])

  const load = useCallback(async (currentMode: AppMode, forceRefresh = false) => {
    if (!mountedRef.current) return
    setStatus('loading')
    setErrorMessage('')
    try {
      if (currentMode === 'live') {
        const live = await fetchLivePlaybook(forceRefresh, pmiMfgOverride, pmiSvcOverride)
        if (!mountedRef.current) return
        setData({
          snapshot: live.snapshot,
          state: live.state,
          playbook_conclusion: live.playbook_conclusion,
          summary: live.summary,
          catalysts: live.catalysts,
          sources: live.sources,
          overall_status: live.overall_status,
          stale_series: live.stale_series,
          generated_at: live.generated_at,
        })
      } else {
        const mock = await fetchPlaybook(resolveMockSnapshot())
        if (!mountedRef.current) return
        setData({
          state: mock.state,
          playbook_conclusion: mock.playbook_conclusion,
          summary: mock.summary,
          catalysts: mock.catalysts,
        })
      }
      setStatus('success')
    } catch (err: unknown) {
      if (!mountedRef.current) return
      setErrorMessage(err instanceof Error ? err.message : 'Unknown error')
      setStatus('error')
    }
  }, [pmiMfgOverride, pmiSvcOverride])

  // Load on mount and mode change
  useEffect(() => { load(mode) }, [mode, load, pmiMfgOverride, pmiSvcOverride])

  const setPmiMfgOverride = useCallback((value: number | null) => {
    const normalized = normalizePmi(value)
    setPmiMfgOverrideState(normalized)
    persistPmi(PMI_MFG_KEY, normalized)
  }, [])

  const setPmiSvcOverride = useCallback((value: number | null) => {
    const normalized = normalizePmi(value)
    setPmiSvcOverrideState(normalized)
    persistPmi(PMI_SVC_KEY, normalized)
  }, [])

  const refresh = useCallback(async () => {
    setRefreshing(true)
    try {
      if (mode === 'live') await triggerManualRefresh()
      await load(mode, true)
    } catch (err: unknown) {
      if (mountedRef.current) {
        setErrorMessage(err instanceof Error ? err.message : 'Refresh failed')
        setStatus('error')
      }
    } finally {
      if (mountedRef.current) setRefreshing(false)
    }
  }, [mode, load])

  // Hard reset: clears data to show loading skeletons, then re-fetches.
  const reset = useCallback(() => {
    setData(null)
    setStatus('idle')
    setErrorMessage('')
    load(mode)
  }, [mode, load])

  return {
    mode,
    setMode,
    pmiMfgOverride,
    pmiSvcOverride,
    setPmiMfgOverride,
    setPmiSvcOverride,
    status,
    data,
    errorMessage,
    refreshing,
    showDebug,
    setShowDebug,
    refresh,
    reset,
  }
}
