import type { PlaybookSummary } from '../../types/summary'

interface Props {
  summary: PlaybookSummary
}

function flagColor(flag: string): { bg: string; border: string; text: string } {
  const f = flag.toLowerCase()
  if (
    f.includes('dangerous') ||
    f.includes('stretched') ||
    f.includes('crash') ||
    f.includes('stress') ||
    f.includes('trap') ||
    f.includes('warning')
  ) {
    return { bg: 'var(--red-bg)', border: 'var(--red-dim)', text: 'var(--red)' }
  }
  if (
    f.includes('active') ||
    f.includes('supportive') ||
    f.includes('liquidity') && f.includes('max')
  ) {
    return { bg: 'var(--green-bg)', border: 'var(--green-dim)', text: 'var(--green)' }
  }
  return { bg: 'var(--yellow-bg)', border: 'var(--yellow-dim)', text: 'var(--yellow)' }
}

export function RiskFlags({ summary }: Props) {
  if (summary.risk_flags.length === 0) return null

  return (
    <div style={styles.container}>
      <span style={styles.label}>Active Signals</span>
      <div style={styles.flags}>
        {summary.risk_flags.map((flag) => {
          const color = flagColor(flag)
          return (
            <span
              key={flag}
              style={{
                ...styles.pill,
                background: color.bg,
                border: `1px solid ${color.border}`,
                color: color.text,
              }}
            >
              {flag}
            </span>
          )
        })}
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    background: 'var(--bg-card)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-md)',
    padding: '16px 20px',
    display: 'flex',
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: '12px',
  },
  label: {
    fontSize: '11px',
    fontWeight: 700,
    letterSpacing: '0.08em',
    textTransform: 'uppercase',
    color: 'var(--text-muted)',
    flexShrink: 0,
  },
  flags: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '6px',
  },
  pill: {
    fontSize: '11px',
    fontWeight: 600,
    letterSpacing: '0.04em',
    padding: '3px 10px',
    borderRadius: '20px',
    whiteSpace: 'nowrap',
  },
}
