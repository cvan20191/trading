import { useState } from 'react'
import type { DashboardState } from '../../types/summary'
import type { SourceMeta } from '../../types/playbook'

interface Props {
  state: DashboardState
  sources?: Record<string, SourceMeta>
  snapshotJson?: string
}

export function DashboardStateDebug({ state, sources, snapshotJson }: Props) {
  const [open, setOpen] = useState(false)
  const [showRaw, setShowRaw] = useState(false)

  return (
    <div style={styles.container}>
      <button style={styles.toggle} onClick={() => setOpen((v) => !v)}>
        <span style={styles.toggleIcon}>{open ? '▾' : '▸'}</span>
        <span>Rule Engine State</span>
        <span style={styles.toggleMeta}>
          {state.primary_regime} · {state.confidence} confidence
        </span>
      </button>

      {open && (
        <div style={styles.grid}>
          <DebugRow label="Primary Regime" value={state.primary_regime} accent="var(--blue)" />
          <DebugRow label="Confidence" value={state.confidence} />
          <DebugRow label="Overlays" value={state.secondary_overlays.join(', ') || '—'} />
          <DebugRow
            label="Posture"
            value={state.current_posture}
            mono={false}
            wrap
          />

          <Divider />

          <DebugRow
            label="Chessboard"
            value={`Quadrant ${state.fed_chessboard?.quadrant ?? '?'} — ${state.fed_chessboard?.label ?? '—'}`}
          />
          <DebugRow
            label="Rate / BS trends"
            value={`Fed target short (4 daily): ${state.fed_chessboard?.rate_trend_1m ?? '—'} | Fed target wider (12 daily): ${state.fed_chessboard?.rate_trend_3m ?? '—'} | BS 1M: ${state.fed_chessboard?.balance_sheet_trend_1m ?? '—'} | BS 3M: ${state.fed_chessboard?.balance_sheet_trend_3m ?? '—'}`}
          />

          <Divider />

          <DebugRow
            label="Stagflation trap"
            value={state.stagflation_trap?.active ? 'ACTIVE' : 'inactive'}
            accent={state.stagflation_trap?.active ? 'var(--yellow)' : undefined}
          />
          <DebugRow
            label="Growth / Inflation"
            value={`Growth weakening: ${state.stagflation_trap?.growth_weakening ? 'yes' : 'no'} | Sticky inflation: ${state.stagflation_trap?.sticky_inflation ? 'yes' : 'no'}`}
          />

          <Divider />

          <DebugRow
            label="Valuation zone"
            value={`${state.valuation?.zone ?? '—'} — ${state.valuation?.zone_label ?? '—'} (${state.valuation?.forward_pe?.toFixed(1) ?? '—'}x)`}
            accent={
              state.valuation?.zone === 'Red' ? 'var(--red)'
              : state.valuation?.zone === 'Green' ? 'var(--green)'
              : undefined
            }
          />

          <Divider />

          <DebugRow
            label="Yield curve"
            value={`${state.systemic_stress?.yield_curve_inverted ? 'Inverted' : 'Normal'} (${state.systemic_stress?.yield_curve_value?.toFixed(2) ?? '—'}%)`}
            accent={state.systemic_stress?.yield_curve_inverted ? 'var(--red)' : undefined}
          />
          <DebugRow
            label="NPL zone"
            value={`${state.systemic_stress?.npl_zone ?? '—'} (${state.systemic_stress?.npl_ratio?.toFixed(2) ?? '—'}%)`}
          />
          <DebugRow
            label="Market Cap / M2"
            value={`${state.systemic_stress?.market_cap_m2_zone ?? '—'} (${state.systemic_stress?.market_cap_m2_ratio?.toFixed(2) ?? '—'})`}
          />

          <Divider />

          <DebugRow
            label="Rally fuel score"
            value={`${state.rally_conditions?.rally_fuel_score ?? '—'} / 100`}
          />
          <DebugRow
            label="Puts active"
            value={[
              state.rally_conditions?.fed_put ? 'Fed' : null,
              state.rally_conditions?.treasury_put ? 'Treasury' : null,
              state.rally_conditions?.political_put ? 'Political' : null,
            ]
              .filter(Boolean)
              .join(', ') || 'None'}
          />
          <DebugRow
            label="Market ignoring bad news"
            value={state.rally_conditions?.market_ignoring_bad_news ? 'Yes' : 'No'}
          />

          <Divider />

          <div style={styles.listSection}>
            <div style={styles.listLabel}>Top watchpoints</div>
            {state.top_watchpoints.map((w, i) => (
              <div key={i} style={styles.listItem}>
                <span style={styles.listNum}>{i + 1}</span>
                {w}
              </div>
            ))}
          </div>

          {/* Optional: raw snapshot JSON */}
          {(sources || snapshotJson) && (
            <>
              <Divider />
              <button
                style={{ ...styles.rawToggle }}
                onClick={() => setShowRaw((v) => !v)}
              >
                {showRaw ? '▾ Hide raw JSON' : '▸ Show raw snapshot / sources JSON'}
              </button>
              {showRaw && snapshotJson && (
                <pre style={styles.rawJson}>{snapshotJson}</pre>
              )}
            </>
          )}
        </div>
      )}
    </div>
  )
}

function DebugRow({
  label,
  value,
  accent,
  mono = true,
  wrap = false,
}: {
  label: string
  value: string
  accent?: string
  mono?: boolean
  wrap?: boolean
}) {
  return (
    <div style={styles.row}>
      <span style={styles.rowLabel}>{label}</span>
      <span
        style={{
          ...styles.rowValue,
          fontFamily: mono ? 'var(--font-mono)' : 'var(--font-sans)',
          color: accent ?? 'var(--text-secondary)',
          whiteSpace: wrap ? 'normal' : 'nowrap',
          overflow: wrap ? 'visible' : 'hidden',
          textOverflow: wrap ? 'clip' : 'ellipsis',
        }}
      >
        {value}
      </span>
    </div>
  )
}

function Divider() {
  return <div style={styles.divider} />
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    background: 'var(--bg-card)',
    border: '1px solid var(--border-subtle)',
    borderRadius: 'var(--radius-md)',
    overflow: 'hidden',
  },
  toggle: {
    width: '100%',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '10px 16px',
    background: 'transparent',
    border: 'none',
    cursor: 'pointer',
    color: 'var(--text-muted)',
    fontSize: '11px',
    fontWeight: 600,
    letterSpacing: '0.06em',
    textTransform: 'uppercase',
    textAlign: 'left',
  },
  toggleIcon: {
    fontSize: '10px',
    color: 'var(--text-muted)',
  },
  toggleMeta: {
    marginLeft: 'auto',
    fontWeight: 400,
    textTransform: 'none',
    letterSpacing: 0,
    color: 'var(--text-muted)',
  },
  grid: {
    padding: '12px 16px 16px',
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
    borderTop: '1px solid var(--border-subtle)',
  },
  row: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    gap: '16px',
    minHeight: '20px',
  },
  rowLabel: {
    fontSize: '11px',
    color: 'var(--text-muted)',
    whiteSpace: 'nowrap',
    flexShrink: 0,
    paddingTop: '1px',
  },
  rowValue: {
    fontSize: '11px',
    textAlign: 'right',
    maxWidth: '60%',
  },
  divider: {
    height: '1px',
    background: 'var(--border-subtle)',
    margin: '4px 0',
  },
  listSection: {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
  },
  listLabel: {
    fontSize: '11px',
    color: 'var(--text-muted)',
    marginBottom: '2px',
  },
  listItem: {
    fontSize: '11px',
    color: 'var(--text-secondary)',
    display: 'flex',
    gap: '8px',
    lineHeight: 1.55,
  },
  listNum: {
    color: 'var(--blue)',
    fontWeight: 700,
    flexShrink: 0,
  },
  rawToggle: {
    background: 'transparent',
    border: 'none',
    cursor: 'pointer',
    color: 'var(--text-muted)',
    fontSize: '10px',
    padding: '2px 0',
    textAlign: 'left',
  },
  rawJson: {
    background: 'var(--bg-card-raised)',
    borderRadius: '4px',
    padding: '10px',
    fontSize: '10px',
    fontFamily: 'var(--font-mono)',
    color: 'var(--text-secondary)',
    overflow: 'auto',
    maxHeight: '280px',
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-all',
    margin: '4px 0 0',
  },
}
