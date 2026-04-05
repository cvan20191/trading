import type { CSSProperties } from 'react'

interface Props {
  label: string
  trend?: string   // 'up' | 'down' | 'flat' | 'unknown' | undefined
  /** Native tooltip (e.g. honest semantics for Fed target windows). */
  title?: string
  style?: CSSProperties
}

const ARROW: Record<string, string> = { up: '↑', down: '↓', flat: '→', unknown: '—' }

export function TrendChip({ label, trend, title, style }: Props) {
  const arrow = ARROW[trend ?? 'unknown'] ?? '—'
  const color =
    trend === 'down' ? 'var(--green)' :
    trend === 'up'   ? 'var(--red)' :
    'var(--text-muted)'

  return (
    <div
      title={title}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '5px',
        padding: '2px 8px',
        borderRadius: '4px',
        background: 'var(--bg-card-raised)',
        border: '1px solid var(--border-subtle)',
        fontSize: '11px',
        cursor: title ? 'help' : undefined,
        ...style,
      }}
    >
      <span style={{ color: 'var(--text-muted)' }}>{label}</span>
      <span style={{ color, fontWeight: 700 }}>{arrow}</span>
    </div>
  )
}
