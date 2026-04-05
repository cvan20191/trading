// Replay Lab main page — blind mode and reveal mode.
//
// Blind mode:  shows date picker + all 5 signal cards + assessment form.
//              Hides CurrentPlaybookStrip, WatchlistCard, ReplayContextPanel.
// Reveal mode: shows full dashboard (minus ThingsToLookOutForCard) +
//              assessment comparison + OutcomeReviewCard + ReplayContextPanel.

import { useState } from 'react'
import type { FedChessboard, StagflationTrap, Valuation, SystemicStress, DollarContext, RallyConditions, PlaybookSummary, DashboardState } from '../../types/summary'
import { useReplayData } from '../../hooks/useReplayData'
import { FedChessboardCard } from '../Dashboard/FedChessboardCard'
import { StagflationTrapCard } from '../Dashboard/StagflationTrapCard'
import { ValuationTriggerCard } from '../Dashboard/ValuationTriggerCard'
import { SystemicStressCard } from '../Dashboard/SystemicStressCard'
import { RallyConditionsCard } from '../Dashboard/RallyConditionsCard'
import { CurrentPlaybookStrip } from '../Dashboard/CurrentPlaybookStrip'
import { WatchlistCard } from '../Dashboard/WatchlistCard'
import { ReplayDatePicker } from './ReplayDatePicker'
import { AssessmentForm } from './AssessmentForm'
import { OutcomeReviewCard } from './OutcomeReviewCard'
import { ReplayContextPanel } from './ReplayContextPanel'

// Default values for optional DashboardState sub-objects
const DEFAULT_CHESSBOARD: FedChessboard = { label: 'Unknown' }
const DEFAULT_TRAP: StagflationTrap = { active: false, growth_weakening: false, sticky_inflation: false, oil_risk_active: false }
const DEFAULT_VALUATION: Valuation = { forward_pe: undefined, zone: undefined, zone_label: undefined, buy_zone_low: 20, buy_zone_high: 25, pause_threshold: 30 }
const DEFAULT_STRESS: SystemicStress = { yield_curve_inverted: false }
const DEFAULT_DOLLAR: DollarContext = { dxy: undefined, dxy_pressure: false }
const DEFAULT_RALLY: RallyConditions = { fed_put: false, treasury_put: false, political_put: false, market_ignoring_bad_news: false }

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

// Default initial date — most recent preset macro moment
const DEFAULT_DATE = '2022-10-14'

interface Props {
  onGoLive: () => void
}

export function ReplayLabPage({ onGoLive }: Props) {
  const {
    selectedDate,
    setSelectedDate,
    playbook,
    outcomes,
    assessment,
    revealed,
    loadingPlaybook,
    loadingOutcomes,
    errorPlaybook,
    errorOutcomes,
    reveal,
    resetForDate,
    updateAssessment,
    saveCurrentAssessment,
  } = useReplayData(DEFAULT_DATE)

  const [pmiMfgOverride, setPmiMfgOverride] = useState<number | null>(null)
  const [pmiSvcOverride, setPmiSvcOverride] = useState<number | null>(null)

  const hasSavedAssessment = Boolean(assessment?.saved_at)

  return (
    <>
      {/* Page header */}
      <div style={s.pageHeader}>
        <div style={s.headerLeft}>
          <button onClick={onGoLive} style={s.backBtn}>← Live Dashboard</button>
          <div>
            <h2 style={s.pageTitle}>Replay Lab</h2>
            <p style={s.pageSubtitle}>
              Choose a historical date and test your macro reading.
              Reveal the engine's computed playbook and review what happened next.
            </p>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          {revealed && (
            <span style={s.revealedBadge}>REVEALED</span>
          )}
          {(playbook || revealed) && (
            <button
              style={s.resetBtn}
              onClick={() => resetForDate(selectedDate)}
              title="Clear this session and return to blind mode"
            >
              Reset
            </button>
          )}
        </div>
      </div>

      <main style={s.page}>
        {/* Date picker */}
        <ReplayDatePicker
          selectedDate={selectedDate}
          onChange={setSelectedDate}
          disabled={loadingPlaybook}
        />

        {/* Date banner */}
        {selectedDate && (
          <div style={s.dateBanner}>
            <span style={s.dateBannerDate}>{selectedDate}</span>
            {loadingPlaybook && <span style={s.loadingNote}>Loading historical data…</span>}
            {revealed && <span style={s.revealedNote}>Playbook revealed</span>}
            {!revealed && !loadingPlaybook && playbook && (
              <span style={s.blindNote}>Blind mode — playbook hidden</span>
            )}
          </div>
        )}

        {/* Error banner */}
        {errorPlaybook && (
          <div style={s.errorBanner}>
            <strong>Failed to load historical data</strong> — {errorPlaybook}
          </div>
        )}

        {/* Data notes (always visible once loaded — transparency first) */}
        {playbook && playbook.data_notes.length > 0 && (
          <div style={s.dataNotes}>
            <span style={s.dataNotesLabel}>Data Disclosures</span>
            <ul style={s.dataNotesUl}>
              {playbook.data_notes.map((n, i) => (
                <li key={i} style={s.dataNoteLi}>{n}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Dashboard cards — always visible when data is loaded */}
        {!loadingPlaybook && playbook && (() => {
          const state = normalizeState(playbook.state)
          const summary = playbook.summary

          return (
            <div style={s.grid}>
              {/* In revealed mode: show playbook strip (full width) */}
              {revealed && (
                <div style={{ gridColumn: '1 / -1' }}>
                  <CurrentPlaybookStrip
                    summary={summary}
                    state={state}
                    playbookConclusion={playbook.playbook_conclusion}
                  />
                </div>
              )}

              {/* 5 signal cards — always visible in both modes */}
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
              <ValuationTriggerCard val={state.valuation} />
              <RallyConditionsCard rally={state.rally_conditions} />
              <SystemicStressCard stress={state.systemic_stress} />

              {/* In revealed mode: show Watchlist */}
              {revealed && (
                <WatchlistCard summary={summary as PlaybookSummary} />
              )}

              {/* In blind mode: assessment form — full width */}
              {!revealed && assessment && (
                <div style={{ gridColumn: '1 / -1' }}>
                  <AssessmentForm
                    assessment={assessment}
                    onUpdate={updateAssessment}
                    onSave={saveCurrentAssessment}
                    onReveal={reveal}
                    saved={hasSavedAssessment}
                  />
                </div>
              )}

              {/* In revealed mode: show assessment comparison */}
              {revealed && assessment && (
                <div style={{ gridColumn: '1 / -1' }}>
                  <AssessmentComparison
                    assessment={assessment}
                    computedRegime={state.primary_regime}
                    computedPosture={state.current_posture}
                    computedWatchpoints={state.top_watchpoints}
                  />
                </div>
              )}

              {/* Outcome review — full width, only in revealed mode */}
              {revealed && (
                <div style={{ gridColumn: '1 / -1' }}>
                  {loadingOutcomes && (
                    <div style={s.loadingOutcomes}>Loading outcome data…</div>
                  )}
                  {errorOutcomes && (
                    <div style={s.errorBanner}>
                      <strong>Outcomes unavailable</strong> — {errorOutcomes}
                    </div>
                  )}
                  {outcomes && <OutcomeReviewCard outcomes={outcomes} />}
                </div>
              )}

              {/* Replay context panel — full width, only in revealed mode */}
              {revealed && (
                <div style={{ gridColumn: '1 / -1' }}>
                  <ReplayContextPanel notes={playbook.context} asOf={selectedDate} />
                </div>
              )}
            </div>
          )
        })()}

        {/* Loading state */}
        {loadingPlaybook && (
          <div style={s.loadingBlock}>
            <div style={s.loadingSpinner} />
            <p style={s.loadingText}>
              Fetching historical macro data for {selectedDate}…<br />
              <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                This may take 15–30 seconds on first load.
              </span>
            </p>
          </div>
        )}
      </main>
    </>
  )
}

// ── Assessment comparison shown after reveal ─────────────────────────────────

interface ComparisonProps {
  assessment: import('../../types/replay').ReplayAssessment
  computedRegime: string
  computedPosture: string
  computedWatchpoints: string[]
}

function AssessmentComparison({ assessment, computedRegime, computedPosture, computedWatchpoints }: ComparisonProps) {
  if (!assessment.saved_at) {
    return (
      <div style={s.comparison}>
        <h4 style={s.compTitle}>Assessment Comparison</h4>
        <p style={{ fontSize: '12px', color: 'var(--text-muted)', margin: 0 }}>
          You did not save an assessment before revealing. Next time, fill in your reading
          in blind mode to compare it against the engine.
        </p>
      </div>
    )
  }

  return (
    <div style={s.comparison}>
      <h4 style={s.compTitle}>Assessment Comparison</h4>
      <div style={s.compGrid}>
        <CompRow label="Regime" yours={assessment.regime || '—'} computed={computedRegime} />
        <CompRow label="Posture" yours={assessment.posture || '—'} computed={computedPosture} />
        {([0, 1, 2] as const).map((i) => (
          <CompRow
            key={i}
            label={`Watchpoint ${i + 1}`}
            yours={assessment.watchpoints[i] || '—'}
            computed={computedWatchpoints[i] || '—'}
          />
        ))}
        <div style={s.compRow}>
          <span style={s.compLabel}>Confidence</span>
          <span style={s.compYours}>{assessment.confidence}</span>
          <span style={s.compComputed}>—</span>
        </div>
      </div>
      {assessment.notes && (
        <div style={{ marginTop: '10px' }}>
          <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Your notes: </span>
          <span style={{ fontSize: '12px', color: 'var(--text)', fontStyle: 'italic' }}>{assessment.notes}</span>
        </div>
      )}
    </div>
  )
}

function CompRow({ label, yours, computed }: { label: string; yours: string; computed: string }) {
  const match = yours.toLowerCase() === computed.toLowerCase()
  return (
    <div style={s.compRow}>
      <span style={s.compLabel}>{label}</span>
      <span style={s.compYours}>{yours}</span>
      <span style={{ ...s.compComputed, color: match ? 'var(--green, #22c55e)' : 'var(--text)' }}>
        {computed}
        {match && <span style={{ marginLeft: '5px', fontSize: '10px' }}>✓</span>}
      </span>
    </div>
  )
}

// ── Styles ───────────────────────────────────────────────────────────────────

const s: Record<string, React.CSSProperties> = {
  pageHeader: {
    display: 'flex',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    padding: '16px 20px 0',
    maxWidth: '1200px',
    margin: '0 auto',
  },
  headerLeft: {
    display: 'flex',
    gap: '12px',
    alignItems: 'flex-start',
  },
  backBtn: {
    background: 'none',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-sm)',
    color: 'var(--text-muted)',
    cursor: 'pointer',
    fontSize: '12px',
    padding: '5px 10px',
    whiteSpace: 'nowrap',
    marginTop: '4px',
  },
  pageTitle: {
    margin: 0,
    fontSize: '20px',
    fontWeight: 800,
    color: 'var(--text)',
  },
  pageSubtitle: {
    margin: '3px 0 0',
    fontSize: '12px',
    color: 'var(--text-muted)',
  },
  revealedBadge: {
    fontSize: '11px',
    fontWeight: 800,
    letterSpacing: '0.1em',
    color: 'var(--green, #22c55e)',
    border: '1px solid var(--green, #22c55e)',
    borderRadius: '4px',
    padding: '3px 8px',
    marginTop: '4px',
  },
  resetBtn: {
    background: 'none',
    border: '1px solid var(--border)',
    borderRadius: '6px',
    color: 'var(--text-muted)',
    cursor: 'pointer',
    fontSize: '12px',
    fontWeight: 600,
    padding: '4px 12px',
    transition: 'border-color 0.15s, color 0.15s',
  } as React.CSSProperties,
  page: {
    maxWidth: '1200px',
    margin: '0 auto',
    padding: '16px 20px 80px',
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '14px',
  },
  dateBanner: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    background: 'var(--bg-card-raised)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-md)',
    padding: '8px 16px',
  },
  dateBannerDate: {
    fontSize: '14px',
    fontWeight: 700,
    color: 'var(--text)',
    fontVariantNumeric: 'tabular-nums',
  },
  loadingNote: {
    fontSize: '12px',
    color: 'var(--text-muted)',
  },
  revealedNote: {
    fontSize: '12px',
    color: 'var(--green, #22c55e)',
    fontWeight: 600,
  },
  blindNote: {
    fontSize: '12px',
    color: 'var(--yellow, #eab308)',
    fontWeight: 600,
  },
  errorBanner: {
    background: 'var(--red-bg, #2d0f0f)',
    border: '1px solid var(--red-dim, #7f1d1d)',
    color: 'var(--red, #ef4444)',
    borderRadius: 'var(--radius-md)',
    padding: '12px 16px',
    fontSize: '13px',
    lineHeight: 1.6,
  },
  dataNotes: {
    background: 'var(--bg-card-raised)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-md)',
    padding: '12px 16px',
  },
  dataNotesLabel: {
    fontSize: '10px',
    fontWeight: 700,
    textTransform: 'uppercase' as const,
    letterSpacing: '0.08em',
    color: 'var(--text-muted)',
  },
  dataNotesUl: {
    margin: '6px 0 0',
    paddingLeft: '16px',
  },
  dataNoteLi: {
    fontSize: '11px',
    color: 'var(--text-muted)',
    lineHeight: 1.6,
    marginBottom: '2px',
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: '14px',
  },
  loadingBlock: {
    display: 'flex',
    flexDirection: 'column' as const,
    alignItems: 'center',
    gap: '12px',
    padding: '40px 20px',
    color: 'var(--text-muted)',
  },
  loadingSpinner: {
    width: '24px',
    height: '24px',
    border: '2px solid var(--border)',
    borderTopColor: 'var(--blue, #3b82f6)',
    borderRadius: '50%',
    animation: 'spin 0.8s linear infinite',
  },
  loadingText: {
    margin: 0,
    fontSize: '13px',
    color: 'var(--text-muted)',
    textAlign: 'center' as const,
    lineHeight: 1.6,
  },
  loadingOutcomes: {
    padding: '16px',
    fontSize: '12px',
    color: 'var(--text-muted)',
    textAlign: 'center' as const,
  },
  comparison: {
    background: 'var(--bg-card)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-lg)',
    padding: '18px 22px',
  },
  compTitle: {
    margin: '0 0 12px',
    fontSize: '14px',
    fontWeight: 700,
    color: 'var(--text)',
  },
  compGrid: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '6px',
  },
  compRow: {
    display: 'grid',
    gridTemplateColumns: '120px 1fr 1fr',
    gap: '8px',
    alignItems: 'baseline',
    borderBottom: '1px solid var(--border)',
    paddingBottom: '5px',
  },
  compLabel: {
    fontSize: '11px',
    color: 'var(--text-muted)',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.05em',
  },
  compYours: {
    fontSize: '12px',
    color: 'var(--text-muted)',
    fontStyle: 'italic' as const,
  },
  compComputed: {
    fontSize: '12px',
    color: 'var(--text)',
    fontWeight: 600,
  },
}
