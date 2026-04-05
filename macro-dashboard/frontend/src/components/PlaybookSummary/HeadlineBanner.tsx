import type { PlaybookSummary } from '../../types/summary'

interface Props {
  summary: PlaybookSummary
}

const REGIME_COLORS: Record<string, { bg: string; border: string; text: string }> = {
  'Max Liquidity':             { bg: 'var(--green-bg)',  border: 'var(--green-dim)',  text: 'var(--green)'  },
  'Liquidity Transition':      { bg: 'var(--blue-bg)',   border: 'var(--blue-dim)',   text: 'var(--blue)'   },
  'Buy-the-Dip Window':        { bg: 'var(--green-bg)',  border: 'var(--green-dim)',  text: 'var(--green)'  },
  'Stagflation Trap':          { bg: 'var(--yellow-bg)', border: 'var(--yellow-dim)', text: 'var(--yellow)' },
  'Valuation Stretched':       { bg: 'var(--red-bg)',    border: 'var(--red-dim)',    text: 'var(--red)'    },
  'Crash Watch':               { bg: 'var(--red-bg)',    border: 'var(--red-dim)',    text: 'var(--red)'    },
  'Defensive / Illiquid Regime':{ bg: 'var(--yellow-bg)',border: 'var(--yellow-dim)', text: 'var(--yellow)' },
  'Mixed / Conflicted Regime': { bg: 'var(--purple-bg)', border: 'var(--purple)',     text: 'var(--purple)' },
}

const DEFAULT_REGIME_COLOR = {
  bg: 'var(--blue-bg)',
  border: 'var(--blue-dim)',
  text: 'var(--blue)',
}

export function HeadlineBanner({ summary }: Props) {
  const regimeColor = REGIME_COLORS[summary.regime_label] ?? DEFAULT_REGIME_COLOR

  return (
    <div style={styles.container}>
      {/* Badge row */}
      <div style={styles.badgeRow}>
        <span
          style={{
            ...styles.badge,
            background: regimeColor.bg,
            border: `1px solid ${regimeColor.border}`,
            color: regimeColor.text,
          }}
        >
          {summary.regime_label}
        </span>
        <span style={{ ...styles.badge, ...styles.postureBadge }}>
          {summary.posture_label}
        </span>
      </div>

      {/* Headline */}
      <p style={styles.headline}>{summary.headline_summary}</p>

      {/* Expanded */}
      <p style={styles.expanded}>{summary.expanded_summary}</p>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    background: 'var(--bg-card)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-lg)',
    padding: '24px 28px',
    display: 'flex',
    flexDirection: 'column',
    gap: '14px',
  },
  badgeRow: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '8px',
    alignItems: 'center',
  },
  badge: {
    fontSize: '11px',
    fontWeight: 600,
    letterSpacing: '0.06em',
    textTransform: 'uppercase',
    padding: '3px 10px',
    borderRadius: '20px',
    whiteSpace: 'nowrap',
  },
  postureBadge: {
    background: 'var(--bg-card-raised)',
    border: '1px solid var(--border)',
    color: 'var(--text-secondary)',
    textTransform: 'none',
    letterSpacing: '0',
    fontWeight: 500,
    fontSize: '12px',
  },
  headline: {
    fontSize: '17px',
    fontWeight: 600,
    lineHeight: 1.5,
    color: 'var(--text-primary)',
  },
  expanded: {
    fontSize: '14px',
    lineHeight: 1.75,
    color: 'var(--text-secondary)',
  },
}
