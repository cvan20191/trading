import type { FedChessboard } from '../../types/summary'
import { Card } from '../ui/Card'
import { TrendChip } from '../ui/TrendChip'

interface Props { cb: FedChessboard }

/** Honest semantics: trends are last N daily DFEDTARU prints, not calendar-month slopes. */
const FED_TARGET_SHORT_TITLE =
  'Recent Fed funds target direction, not a monthly average. Uses the last four daily upper-bound readings (DFEDTARU). On a step-wise target, this mostly flags whether a recent target change falls inside that slice.'

const FED_TARGET_WIDER_TITLE =
  'Same idea over more daily prints — still not a calendar-month trend. Uses the last twelve daily readings. Between FOMC meetings this usually reads flat unless a change sits inside the window.'

// Grid layout: D=top-left, B=top-right, C=bottom-left, A=bottom-right
// Axes: horizontal = balance sheet, vertical = rates

export function FedChessboardCard({ cb }: Props) {
  const activeId = cb.quadrant

  const INTERPRETATION: Record<string, string> = {
    A: 'Maximum liquidity — both levers are accommodative. Growth and risk assets typically outperform.',
    B: 'Mixed: balance sheet supports, but rate pressure persists. Selectively supportive.',
    C: 'Transition toward easier money — rate direction improving, but balance sheet still contracting.',
    D: 'Maximum illiquidity — both rate pressure and balance sheet drain. Most hostile environment.',
  }

  return (
    <Card title="Fed Chessboard — Liquidity Quadrant">
      {/* 2×2 matrix: top row = Rates Up (D=left, B=right); bottom row = Rates Down (C=left, A=right) */}
      <div style={s.axisOuter}>
        {/* Y-axis label */}
        <div style={s.yAxisLabel}>
          <span style={s.axisText}>Rates ↑</span>
          <span style={s.axisArrow}>↕</span>
          <span style={s.axisText}>Rates ↓</span>
        </div>

        <div style={{ flex: 1 }}>
          {/* X-axis label */}
          <div style={s.xAxisLabel}>
            <span style={s.axisText}>BS ↓</span>
            <span style={s.axisArrow}>↔</span>
            <span style={s.axisText}>BS ↑</span>
          </div>

          {/* 2×2 grid */}
          <div style={s.grid}>
            {[
              { id: 'D', gridRow: 1, gridCol: 1, desc: 'Max Illiquidity', color: 'var(--red)' },
              { id: 'B', gridRow: 1, gridCol: 2, desc: 'Mixed Liquidity', color: 'var(--yellow)' },
              { id: 'C', gridRow: 2, gridCol: 1, desc: 'Transition', color: 'var(--blue)' },
              { id: 'A', gridRow: 2, gridCol: 2, desc: 'Max Liquidity', color: 'var(--green)' },
            ].map(q => {
              const isActive = q.id === activeId
              return (
                <div key={q.id} style={{
                  ...s.cell,
                  gridRow: q.gridRow,
                  gridColumn: q.gridCol,
                  background: isActive ? `${q.color}22` : 'var(--bg-card-raised)',
                  border: isActive ? `2px solid ${q.color}` : '1px solid var(--border)',
                }}>
                  <span style={{ fontSize: '16px', fontWeight: 800, color: isActive ? q.color : 'var(--border)' }}>
                    {q.id}
                  </span>
                  <span style={{ fontSize: '10px', color: isActive ? q.color : 'var(--text-muted)', textAlign: 'center' }}>
                    {q.desc}
                  </span>
                  {isActive && (
                    <span style={{ fontSize: '14px' }}>●</span>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      </div>

      {/* Trend chips */}
      <div style={s.chips}>
        <TrendChip label="Fed target — short window" title={FED_TARGET_SHORT_TITLE} trend={cb.rate_trend_1m} />
        <TrendChip label="Fed target — wider window" title={FED_TARGET_WIDER_TITLE} trend={cb.rate_trend_3m} />
        <TrendChip label="Balance Sheet 1M" trend={cb.balance_sheet_trend_1m} />
        <TrendChip label="Balance Sheet 3M" trend={cb.balance_sheet_trend_3m} />
      </div>

      {/* Active quadrant interpretation */}
      {activeId && (
        <div style={s.interp}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
            <span style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-muted)', letterSpacing: '0.06em', textTransform: 'uppercase' }}>
              Quadrant {activeId} — {cb.label ?? ''}
            </span>
            {cb.transition_tag && (
              <span style={{
                fontSize: '10px',
                fontWeight: 700,
                padding: '2px 7px',
                borderRadius: '10px',
                letterSpacing: '0.04em',
                color:
                  cb.transition_tag === 'Improving' ? 'var(--green)' :
                  cb.transition_tag === 'Deteriorating' ? 'var(--yellow)' :
                  'var(--text-muted)',
                background:
                  cb.transition_tag === 'Improving' ? 'color-mix(in srgb, var(--green) 15%, transparent)' :
                  cb.transition_tag === 'Deteriorating' ? 'color-mix(in srgb, var(--yellow) 15%, transparent)' :
                  'var(--bg-card-raised)',
                border: `1px solid ${
                  cb.transition_tag === 'Improving' ? 'color-mix(in srgb, var(--green) 40%, transparent)' :
                  cb.transition_tag === 'Deteriorating' ? 'color-mix(in srgb, var(--yellow) 40%, transparent)' :
                  'var(--border-subtle)'
                }`,
              }}>
                {cb.transition_tag}
              </span>
            )}
          </div>
          <p style={{ margin: 0, fontSize: '12px', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
            {INTERPRETATION[activeId] ?? cb.label}
          </p>
        </div>
      )}
    </Card>
  )
}

const s: Record<string, React.CSSProperties> = {
  axisOuter: { display: 'flex', gap: '10px', alignItems: 'stretch' },
  yAxisLabel: {
    display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
    gap: '6px', width: '30px', flexShrink: 0,
  },
  xAxisLabel: {
    display: 'flex', justifyContent: 'space-between', paddingLeft: '4px', paddingRight: '4px', marginBottom: '6px',
  },
  axisText: { fontSize: '9px', color: 'var(--text-muted)', fontWeight: 600, letterSpacing: '0.04em' },
  axisArrow: { fontSize: '10px', color: 'var(--border)', fontWeight: 400 },
  grid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gridTemplateRows: '1fr 1fr',
    gap: '4px',
    height: '140px',
  },
  cell: {
    display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
    borderRadius: '6px', gap: '3px', padding: '6px', transition: 'all 0.2s',
  },
  chips: { display: 'flex', flexWrap: 'wrap', gap: '6px' },
  interp: {
    display: 'flex', flexDirection: 'column', gap: '6px',
    padding: '12px 14px', background: 'var(--bg-card-raised)',
    borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)',
  },
}
