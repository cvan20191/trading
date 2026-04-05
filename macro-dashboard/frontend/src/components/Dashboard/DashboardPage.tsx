import type { DashboardState, FedChessboard, StagflationTrap, Valuation, SystemicStress, RallyConditions, DollarContext, PlaybookSummary } from '../../types/summary'
import type { SourceMeta } from '../../types/playbook'
import { usePlaybookData } from '../../hooks/usePlaybookData'
import { SourceStatusStrip } from './SourceStatusStrip'
import { CurrentPlaybookStrip } from './CurrentPlaybookStrip'
import { FedChessboardCard } from './FedChessboardCard'
import { FedLiquidityOverviewCard } from './FedLiquidityOverviewCard'
import { StagflationTrapCard } from './StagflationTrapCard'
import { ValuationTriggerCard } from './ValuationTriggerCard'
import { Mag7ValuationHistoryCard } from './Mag7ValuationHistoryCard'
import { RallyConditionsCard } from './RallyConditionsCard'
import { SystemicStressCard } from './SystemicStressCard'
import { WatchlistCard } from './WatchlistCard'
import { ThingsToLookOutForCard } from './ThingsToLookOutForCard'
import { DashboardStateDebug } from '../DashboardStateDebug'

// ---------------------------------------------------------------------------
// Safe default sections — used when backend returns partial DashboardState
// ---------------------------------------------------------------------------

const DEFAULT_CHESSBOARD: FedChessboard = {
  quadrant: undefined, label: 'Unknown', rate_trend_1m: undefined,
  rate_trend_3m: undefined, balance_sheet_trend_1m: undefined, balance_sheet_trend_3m: undefined,
}

const DEFAULT_TRAP: StagflationTrap = {
  active: false, growth_weakening: false, sticky_inflation: false, oil_risk_active: false,
}

const DEFAULT_VALUATION: Valuation = {
  forward_pe: undefined, zone: undefined, zone_label: undefined,
  buy_zone_low: 20, buy_zone_high: 25, pause_threshold: 30,
}

const DEFAULT_STRESS: SystemicStress = {
  yield_curve_inverted: false, yield_curve_value: undefined,
  npl_ratio: undefined, npl_zone: undefined, market_cap_m2_ratio: undefined, market_cap_m2_zone: undefined,
}

const DEFAULT_DOLLAR: DollarContext = { dxy: undefined, dxy_pressure: false }

const DEFAULT_RALLY: RallyConditions = {
  rally_fuel_score: undefined, fed_put: false, treasury_put: false,
  political_put: false, market_ignoring_bad_news: false,
}

function normalizeState(state: DashboardState): Required<DashboardState> {
  return {
    ...state,
    fed_chessboard:   state.fed_chessboard   ?? DEFAULT_CHESSBOARD,
    stagflation_trap: state.stagflation_trap ?? DEFAULT_TRAP,
    valuation:        state.valuation        ?? DEFAULT_VALUATION,
    systemic_stress:  state.systemic_stress  ?? DEFAULT_STRESS,
    dollar_context:   state.dollar_context   ?? DEFAULT_DOLLAR,
    rally_conditions: state.rally_conditions ?? DEFAULT_RALLY,
  } as Required<DashboardState>
}

// ---------------------------------------------------------------------------
// Skeleton loading
// ---------------------------------------------------------------------------
function SkeletonCard({ height = 160 }: { height?: number }) {
  return (
    <div style={{
      height, borderRadius: 'var(--radius-lg)',
      background: 'var(--bg-card)',
      border: '1px solid var(--border)',
      overflow: 'hidden',
    }}>
      <div style={{
        height: '100%',
        background: 'linear-gradient(90deg, var(--bg-card) 25%, var(--bg-card-raised) 50%, var(--bg-card) 75%)',
        backgroundSize: '200% 100%',
        animation: 'shimmer 1.4s ease-in-out infinite',
      }} />
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export function DashboardPage() {
  const {
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
  } = usePlaybookData()

  const staleCount = data?.stale_series?.length ?? 0
  const isLoading = status === 'idle' || status === 'loading'

  // Derive sources for debug panel
  const sources = data?.sources as Record<string, SourceMeta> | undefined
  const snapshotJson = data?.snapshot ? JSON.stringify(data.snapshot, null, 2) : undefined

  return (
    <>
      <style>{`
        @keyframes shimmer {
          0%   { background-position: 200% 0; }
          100% { background-position: -200% 0; }
        }
      `}</style>

      {/* Sticky header strip */}
      <SourceStatusStrip
        mode={mode}
        onSetMode={setMode}
        overallStatus={data?.overall_status}
        staleCount={staleCount}
        staleSeries={data?.stale_series ?? []}
        sources={sources}
        generatedAt={data?.generated_at}
        refreshing={refreshing}
        onRefresh={refresh}
        onReset={reset}
        showDebug={showDebug}
        onToggleDebug={() => setShowDebug(!showDebug)}
      />

      <main style={s.page}>
        {/* Error banner */}
        {status === 'error' && (
          <div style={s.errorBanner}>
            <strong>Dashboard unavailable</strong> — {errorMessage}
            {mode === 'live' && (
              <div style={{ marginTop: '6px', fontSize: '12px', opacity: 0.8 }}>
                Switch to <button style={s.inlineSwitchBtn} onClick={() => setMode('mock')}>Mock mode</button> to use offline data.
              </div>
            )}
          </div>
        )}

        {/* Loading skeletons */}
        {isLoading && (
          <div className="dashboard-grid" style={s.grid}>
            <div style={{ gridColumn: '1 / -1' }}><SkeletonCard height={180} /></div>
            <SkeletonCard height={280} />
            <SkeletonCard height={280} />
            <SkeletonCard height={220} />
            <SkeletonCard height={220} />
            <SkeletonCard height={200} />
            <SkeletonCard height={220} />
          </div>
        )}

        {/* Main dashboard */}
        {status === 'success' && data && (() => {
          const state = normalizeState(data.state)
          const summary = data.summary

          return (
            <div className="dashboard-grid" style={s.grid}>
              {/* Row 1: Current Playbook strip — full width */}
              <div style={{ gridColumn: '1 / -1' }}>
                <CurrentPlaybookStrip
                  summary={summary}
                  state={state}
                  playbookConclusion={data.playbook_conclusion}
                />
              </div>

              {/* Row 2: Fed Chessboard + Stagflation Trap */}
              <FedChessboardCard cb={state.fed_chessboard} />
              <StagflationTrapCard
                trap={state.stagflation_trap}
                pmiMfgOverride={pmiMfgOverride}
                pmiSvcOverride={pmiSvcOverride}
                onPmiOverrideChange={(mfg, svc) => {
                  setPmiMfgOverride(mfg)
                  setPmiSvcOverride(svc)
                }}
              />

              {/* Row 2.5: Fed liquidity overview — full width */}
              <div style={{ gridColumn: '1 / -1' }}>
                <FedLiquidityOverviewCard />
              </div>

              {/* Row 3: Valuation + Rally Conditions */}
              <ValuationTriggerCard val={state.valuation} />
              <RallyConditionsCard rally={state.rally_conditions} />

              {/* Row 3.5: Mag 7 cached valuation history — full width, local JSON only */}
              <div style={{ gridColumn: '1 / -1' }}>
                <Mag7ValuationHistoryCard />
              </div>

              {/* Row 4: Systemic Stress + Watchlist */}
              <SystemicStressCard stress={state.systemic_stress} />
              <WatchlistCard summary={summary as PlaybookSummary} />

              {/* Row 5: Things to Look Out For — full width */}
              {data.catalysts && (
                <div style={{ gridColumn: '1 / -1' }}>
                  <ThingsToLookOutForCard catalysts={data.catalysts} />
                </div>
              )}

              {/* Debug panel — full width, toggled */}
              {showDebug && (
                <div style={{ gridColumn: '1 / -1' }}>
                  <DashboardStateDebug
                    state={data.state}
                    sources={sources}
                    snapshotJson={snapshotJson}
                  />
                </div>
              )}
            </div>
          )
        })()}
      </main>
    </>
  )
}

const s: Record<string, React.CSSProperties> = {
  page: {
    maxWidth: '1200px',
    margin: '0 auto',
    padding: '24px 20px 80px',
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: '16px',
  },
  errorBanner: {
    gridColumn: '1 / -1',
    background: 'var(--red-bg)',
    border: '1px solid var(--red-dim)',
    color: 'var(--red)',
    borderRadius: 'var(--radius-md)',
    padding: '14px 18px',
    fontSize: '13px',
    lineHeight: 1.6,
    marginBottom: '16px',
  },
  inlineSwitchBtn: {
    background: 'none',
    border: 'none',
    color: 'var(--red)',
    textDecoration: 'underline',
    cursor: 'pointer',
    fontSize: 'inherit',
    padding: 0,
  },
}
