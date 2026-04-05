import type { ColorKey } from '../../lib/colors'
import { colorVars } from '../../lib/colors'

interface Props {
  label: string
  colorKey: ColorKey
  size?: 'sm' | 'md'
}

export function StatusPill({ label, colorKey, size = 'md' }: Props) {
  const c = colorVars(colorKey)
  return (
    <span style={{
      display: 'inline-flex',
      alignItems: 'center',
      padding: size === 'sm' ? '2px 8px' : '3px 10px',
      borderRadius: '99px',
      fontSize: size === 'sm' ? '10px' : '11px',
      fontWeight: 600,
      letterSpacing: '0.04em',
      background: c.bg,
      color: c.fg,
      border: `1px solid ${c.dim}`,
      whiteSpace: 'nowrap',
    }}>
      {label}
    </span>
  )
}
