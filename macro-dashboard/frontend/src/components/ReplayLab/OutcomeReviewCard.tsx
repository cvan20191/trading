// Outcome review card — shows SPY, QQQ, and WTI price returns at 1W, 1M, 3M
// after the selected replay date. WTI closes the inflation loop for the user.

import type { OutcomePoint, OutcomeReview } from '../../types/replay'

interface Props {
  outcomes: OutcomeReview
}

export function OutcomeReviewCard({ outcomes }: Props) {
  return (
    <div style={s.wrapper}>
      <div style={s.header}>
        <h3 style={s.title}>What Happened Next</h3>
        <p style={s.subtitle}>
          Price returns from <strong>{outcomes.as_of}</strong>. WTI (crude oil) closes the
          inflation loop — did the energy component follow the playbook?
        </p>
      </div>

      {/* Table */}
      <div style={s.tableWrapper}>
        <table style={s.table}>
          <thead>
            <tr>
              <th style={s.th}>Asset</th>
              <th style={s.th}>+1 Week</th>
              <th style={s.th}>+1 Month</th>
              <th style={s.th}>+3 Months</th>
            </tr>
          </thead>
          <tbody>
            <OutcomeRow
              label="SPY (S&P 500)"
              ticker="SPY"
              w1={outcomes.outcomes_1w}
              m1={outcomes.outcomes_1m}
              m3={outcomes.outcomes_3m}
              retKey="spy_return_pct"
              startKey="spy_price_start"
              endKey="spy_price_end"
            />
            <OutcomeRow
              label="QQQ (Nasdaq-100)"
              ticker="QQQ"
              w1={outcomes.outcomes_1w}
              m1={outcomes.outcomes_1m}
              m3={outcomes.outcomes_3m}
              retKey="qqq_return_pct"
              startKey="qqq_price_start"
              endKey="qqq_price_end"
            />
            <OutcomeRow
              label="WTI Crude (CL=F)"
              ticker="WTI"
              w1={outcomes.outcomes_1w}
              m1={outcomes.outcomes_1m}
              m3={outcomes.outcomes_3m}
              retKey="wti_change_pct"
              startKey="wti_price_start"
              endKey="wti_price_end"
            />
          </tbody>
        </table>
      </div>

      {/* Horizon dates */}
      <div style={s.horizonDates}>
        <HorizonDate label="+1W" date={outcomes.outcomes_1w.date_end} />
        <HorizonDate label="+1M" date={outcomes.outcomes_1m.date_end} />
        <HorizonDate label="+3M" date={outcomes.outcomes_3m.date_end} />
      </div>

      {outcomes.data_note && (
        <p style={s.dataNote}>{outcomes.data_note}</p>
      )}
    </div>
  )
}

type ReturnKey = 'spy_return_pct' | 'qqq_return_pct' | 'wti_change_pct'
type PriceKey = 'spy_price_start' | 'spy_price_end' | 'qqq_price_start' | 'qqq_price_end' | 'wti_price_start' | 'wti_price_end'

interface RowProps {
  label: string
  ticker: string
  w1: OutcomePoint
  m1: OutcomePoint
  m3: OutcomePoint
  retKey: ReturnKey
  startKey: PriceKey
  endKey: PriceKey
}

function OutcomeRow({ label, w1, m1, m3, retKey, startKey, endKey }: RowProps) {
  return (
    <tr>
      <td style={s.tdLabel}>{label}</td>
      <OutcomeCell point={w1} retKey={retKey} startKey={startKey} endKey={endKey} />
      <OutcomeCell point={m1} retKey={retKey} startKey={startKey} endKey={endKey} />
      <OutcomeCell point={m3} retKey={retKey} startKey={startKey} endKey={endKey} />
    </tr>
  )
}

interface CellProps {
  point: OutcomePoint
  retKey: ReturnKey
  startKey: PriceKey
  endKey: PriceKey
}

function OutcomeCell({ point, retKey, startKey, endKey }: CellProps) {
  const ret = point[retKey]
  const start = point[startKey]
  const end = point[endKey]

  if (ret === null) {
    return (
      <td style={s.td}>
        <span style={s.unavail}>—</span>
      </td>
    )
  }

  const isPositive = ret >= 0
  const color = isPositive ? 'var(--green, #22c55e)' : 'var(--red, #ef4444)'
  const sign = isPositive ? '+' : ''

  return (
    <td style={s.td}>
      <span style={{ ...s.returnVal, color }}>{sign}{ret.toFixed(1)}%</span>
      {start !== null && end !== null && (
        <span style={s.prices}>${start.toFixed(0)} → ${end.toFixed(0)}</span>
      )}
    </td>
  )
}

function HorizonDate({ label, date }: { label: string; date: string }) {
  return (
    <span style={s.horizon}>
      <span style={s.horizonLabel}>{label}</span>
      <span style={s.horizonDate}>{date}</span>
    </span>
  )
}

const s: Record<string, React.CSSProperties> = {
  wrapper: {
    background: 'var(--bg-card)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-lg)',
    padding: '20px 24px',
    gridColumn: '1 / -1',
  },
  header: {
    marginBottom: '16px',
  },
  title: {
    margin: 0,
    fontSize: '15px',
    fontWeight: 700,
    color: 'var(--text)',
  },
  subtitle: {
    margin: '4px 0 0',
    fontSize: '12px',
    color: 'var(--text-muted)',
    lineHeight: 1.5,
  },
  tableWrapper: {
    overflowX: 'auto' as const,
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse' as const,
  },
  th: {
    fontSize: '11px',
    color: 'var(--text-muted)',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.05em',
    padding: '6px 12px',
    textAlign: 'left' as const,
    borderBottom: '1px solid var(--border)',
  },
  tdLabel: {
    fontSize: '13px',
    color: 'var(--text)',
    fontWeight: 600,
    padding: '8px 12px',
    borderBottom: '1px solid var(--border)',
    whiteSpace: 'nowrap' as const,
  },
  td: {
    padding: '8px 12px',
    borderBottom: '1px solid var(--border)',
    verticalAlign: 'top' as const,
  },
  returnVal: {
    display: 'block',
    fontSize: '15px',
    fontWeight: 700,
  },
  prices: {
    display: 'block',
    fontSize: '10px',
    color: 'var(--text-muted)',
    marginTop: '2px',
  },
  unavail: {
    color: 'var(--text-muted)',
    fontSize: '14px',
  },
  horizonDates: {
    display: 'flex',
    gap: '20px',
    marginTop: '10px',
  },
  horizon: {
    display: 'flex',
    gap: '5px',
    alignItems: 'baseline',
  },
  horizonLabel: {
    fontSize: '10px',
    color: 'var(--text-muted)',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.05em',
  },
  horizonDate: {
    fontSize: '11px',
    color: 'var(--text-muted)',
  },
  dataNote: {
    margin: '10px 0 0',
    fontSize: '11px',
    color: 'var(--text-muted)',
    fontStyle: 'italic' as const,
  },
}
