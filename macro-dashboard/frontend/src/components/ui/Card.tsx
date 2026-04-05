import type { ReactNode, CSSProperties } from 'react'

interface Props {
  title?: string
  children: ReactNode
  style?: CSSProperties
  accent?: string   // optional left-border accent color
}

export function Card({ title, children, style, accent }: Props) {
  return (
    <div style={{
      background: 'var(--bg-card)',
      border: '1px solid var(--border)',
      borderLeft: accent ? `3px solid ${accent}` : '1px solid var(--border)',
      borderRadius: 'var(--radius-lg)',
      padding: '18px 20px',
      display: 'flex',
      flexDirection: 'column',
      gap: '14px',
      ...style,
    }}>
      {title && (
        <div style={{
          fontSize: '11px',
          fontWeight: 700,
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
          color: 'var(--text-muted)',
        }}>
          {title}
        </div>
      )}
      {children}
    </div>
  )
}
