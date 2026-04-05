import type { PlaybookSummary } from '../../types/summary'
import type { DashboardState } from '../../types/summary'
import type { PlaybookConclusion } from '../../types/playbook'
import { StatusPill } from '../ui/StatusPill'
import { regimeColor, confidenceColor, colorVars } from '../../lib/colors'

interface Props {
  summary: PlaybookSummary
  state: DashboardState
  playbookConclusion?: PlaybookConclusion
}

const ACTION_LABEL: Record<PlaybookConclusion['new_cash_action'], string> = {
  accumulate: 'Accumulate',
  accumulate_selectively: 'Accumulate selectively',
  hold_and_wait: 'Hold and wait',
  pause_new_buying: 'Pause new buying (not auto-sell)',
  defensive_preservation: 'Defensive preservation',
}

const URGENCY_LABEL: Record<PlaybookConclusion['warning_urgency'], string> = {
  cautionary: 'Cautionary warning posture',
  elevated: 'Elevated warning posture',
  urgent: 'Urgent warning posture',
}

const WHY_NOW_LABEL: Record<string, string> = {
  liquidity_quadrant_a_supportive: 'Liquidity: Quadrant A supportive',
  liquidity_quadrant_b_mixed_support: 'Liquidity: Quadrant B mixed support',
  liquidity_quadrant_c_transition: 'Liquidity: Quadrant C transition',
  liquidity_quadrant_d_tight: 'Liquidity: Quadrant D tight',
  valuation_buy_zone: 'Valuation: buy zone',
  valuation_stretched_pause_new_buying: 'Valuation: stretched, pause new buying',
  valuation_neutral_wait_for_edge: 'Valuation: neutral, wait for edge',
  stagflation_trap_active: 'Stagflation trap active',
  systemic_stress_severe: 'Systemic stress severe',
  systemic_stress_warning_active: 'Systemic stress warning active',
  market_ignoring_bad_news: 'Market can ignore bad news',
}

const LENIENCY_LABEL: Record<string, string> = {
  valuation_proxy_not_true_forward_pe: 'Valuation is using a proxy, not true forward P/E',
  stress_gauges_warning_lights_not_timers: 'Stress gauges are warning lights, not timers',
  transition_regime_can_overshoot: 'Transition regime can overshoot',
  stretched_means_pause_not_forced_sell: 'Stretched means pause, not forced sell',
  mixed_signals_reduce_conviction: 'Mixed signals reduce conviction',
}

function humanizeTag(value: string): string {
  return value.split('_').join(' ')
}

function formatWhyNow(whyNow: string): string {
  return whyNow
    .split(';')
    .map(clause => clause.trim())
    .filter(Boolean)
    .map(clause => WHY_NOW_LABEL[clause] ?? humanizeTag(clause))
    .join(' • ')
}

function formatLeniencyNotes(notes: string[]): string {
  return notes.map(note => LENIENCY_LABEL[note] ?? humanizeTag(note)).join(', ')
}

export function CurrentPlaybookStrip({ summary, state, playbookConclusion }: Props) {
  const rColor = regimeColor(state.primary_regime)
  const rVars = colorVars(rColor)
  const isFallback = summary.meta?.used_fallback

  return (
    <div style={{ ...s.wrapper, borderLeft: `3px solid ${rVars.fg}` }}>
      {/* Top row: regime + confidence + fallback note */}
      <div style={s.topRow}>
        <div style={s.badges}>
          <StatusPill label={state.primary_regime} colorKey={rColor} />
          <StatusPill label={`${state.confidence} Confidence`} colorKey={confidenceColor(state.confidence)} size="sm" />
          {state.secondary_overlays.slice(0, 3).map(o => (
            <StatusPill key={o} label={o} colorKey="muted" size="sm" />
          ))}
        </div>
        {isFallback && (
          <span style={s.fallbackNote}>Deterministic summary (LLM offline)</span>
        )}
      </div>

      {/* Headline */}
      <p style={s.headline}>{summary.headline_summary}</p>

      {/* Posture */}
      <div style={s.postureRow}>
        <span style={s.postureLabel}>Posture</span>
        <span style={s.postureText}>{state.current_posture}</span>
      </div>

      {/* Structured conclusion (optional) */}
      {playbookConclusion && (
        <div style={s.conclusionBox}>
          <div style={s.conclusionTitle}>Playbook Conclusion</div>
          <div style={s.conclusionGrid}>
            <div><strong>New cash action:</strong> {ACTION_LABEL[playbookConclusion.new_cash_action]}</div>
            <div><strong>Warning:</strong> {URGENCY_LABEL[playbookConclusion.warning_urgency]}</div>
            <div><strong>Can rally despite bad news:</strong> {playbookConclusion.can_rally_despite_bad_news ? 'Yes' : 'No'}</div>
            <div><strong>Why now:</strong> {formatWhyNow(playbookConclusion.why_now)}</div>
            {playbookConclusion.stock_archetype_preferred.length > 0 && (
              <div>
                <strong>Prefer:</strong> {playbookConclusion.stock_archetype_preferred.map(humanizeTag).join(', ')}
              </div>
            )}
            {playbookConclusion.stock_archetype_avoid.length > 0 && (
              <div>
                <strong>Avoid:</strong> {playbookConclusion.stock_archetype_avoid.map(humanizeTag).join(', ')}
              </div>
            )}
            {playbookConclusion.leniency_notes.length > 0 && (
              <div>
                <strong>Caveats:</strong> {formatLeniencyNotes(playbookConclusion.leniency_notes)}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Expanded summary */}
      <p style={s.expanded}>{summary.expanded_summary}</p>

      {/* Risk flags */}
      {summary.risk_flags.length > 0 && (
        <div style={s.flagsRow}>
          {summary.risk_flags.map(f => (
            <RiskFlag key={f} label={f} />
          ))}
        </div>
      )}

      {/* Teaching note */}
      <div style={s.teachingNote}>
        <span style={s.teachingIcon}>⟳</span>
        <span style={s.teachingText}>{summary.teaching_note}</span>
      </div>
    </div>
  )
}

function RiskFlag({ label }: { label: string }) {
  const isRed = /stretched|warning|stress|crash|stagflation|illiquid/i.test(label)
  const isGreen = /active|supportive|rally/i.test(label)
  const color = isRed ? 'var(--red)' : isGreen ? 'var(--green)' : 'var(--yellow)'
  const bg = isRed ? 'var(--red-bg)' : isGreen ? 'var(--green-bg)' : 'var(--yellow-bg)'
  const border = isRed ? 'var(--red-dim)' : isGreen ? 'var(--green-dim)' : 'var(--yellow-dim)'
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center',
      padding: '2px 9px', borderRadius: '3px',
      fontSize: '11px', fontWeight: 600,
      background: bg, color, border: `1px solid ${border}`,
    }}>
      {label}
    </span>
  )
}

const s: Record<string, React.CSSProperties> = {
  wrapper: {
    background: 'var(--bg-card)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-lg)',
    padding: '22px 24px',
    display: 'flex',
    flexDirection: 'column',
    gap: '14px',
  },
  topRow: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    flexWrap: 'wrap',
    gap: '8px',
  },
  badges: {
    display: 'flex', flexWrap: 'wrap', gap: '6px', alignItems: 'center',
  },
  fallbackNote: {
    fontSize: '10px',
    color: 'var(--text-muted)',
    fontStyle: 'italic',
    background: 'var(--bg-card-raised)',
    padding: '2px 8px',
    borderRadius: '3px',
    border: '1px solid var(--border-subtle)',
  },
  headline: {
    fontSize: '17px',
    fontWeight: 600,
    color: 'var(--text-primary)',
    lineHeight: 1.5,
    letterSpacing: '-0.01em',
  },
  postureRow: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: '10px',
    padding: '10px 14px',
    background: 'var(--bg-card-raised)',
    borderRadius: 'var(--radius-sm)',
    border: '1px solid var(--border-subtle)',
  },
  postureLabel: {
    fontSize: '10px',
    fontWeight: 700,
    letterSpacing: '0.08em',
    textTransform: 'uppercase',
    color: 'var(--text-muted)',
    flexShrink: 0,
    paddingTop: '2px',
  },
  postureText: {
    fontSize: '13px',
    color: 'var(--text-primary)',
    lineHeight: 1.5,
  },
  conclusionBox: {
    padding: '10px 14px',
    background: 'var(--bg-card-raised)',
    borderRadius: 'var(--radius-sm)',
    border: '1px solid var(--border-subtle)',
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  conclusionTitle: {
    fontSize: '10px',
    fontWeight: 700,
    letterSpacing: '0.08em',
    textTransform: 'uppercase',
    color: 'var(--text-muted)',
  },
  conclusionGrid: {
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
    fontSize: '12px',
    color: 'var(--text-secondary)',
    lineHeight: 1.5,
  },
  expanded: {
    fontSize: '13px',
    color: 'var(--text-secondary)',
    lineHeight: 1.7,
  },
  flagsRow: {
    display: 'flex', flexWrap: 'wrap', gap: '6px',
  },
  teachingNote: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: '8px',
    padding: '10px 14px',
    background: 'var(--blue-bg)',
    borderRadius: 'var(--radius-sm)',
    border: '1px solid var(--blue-dim)',
  },
  teachingIcon: {
    fontSize: '13px',
    color: 'var(--blue)',
    flexShrink: 0,
  },
  teachingText: {
    fontSize: '12px',
    color: 'var(--blue)',
    lineHeight: 1.6,
    fontStyle: 'italic',
  },
}
