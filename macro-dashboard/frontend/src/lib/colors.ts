// Zone and status → CSS variable mappings

export type ColorKey = 'green' | 'yellow' | 'red' | 'blue' | 'muted'

export function zoneColor(zone: string | undefined): ColorKey {
  if (!zone) return 'muted'
  const z = zone.toLowerCase()
  if (z === 'green' || z === 'normal' || z === 'safe margin') return 'green'
  if (z === 'yellow' || z === 'caution' || z === 'neutral') return 'yellow'
  if (z === 'red' || z === 'warning' || z === 'extreme' || z === 'valuation stretched') return 'red'
  return 'muted'
}

export function regimeColor(regime: string | undefined): ColorKey {
  if (!regime) return 'muted'
  const r = regime.toLowerCase()
  if (r.includes('max liquidity') || r.includes('buy-the-dip')) return 'green'
  if (r.includes('transition') || r.includes('mixed')) return 'blue'
  if (r.includes('stagflation') || r.includes('valuation')) return 'yellow'
  if (r.includes('crash') || r.includes('defensive') || r.includes('illiquid')) return 'red'
  return 'blue'
}

export function confidenceColor(confidence: string | undefined): ColorKey {
  if (!confidence) return 'muted'
  const c = confidence.toLowerCase()
  if (c === 'high') return 'green'
  if (c === 'medium') return 'yellow'
  return 'red'
}

export function freshnessColor(status: string | undefined): ColorKey {
  if (!status) return 'muted'
  if (status === 'fresh') return 'green'
  if (status === 'mixed') return 'yellow'
  return 'red'
}

export function trendColor(trend: string | undefined): ColorKey {
  if (!trend) return 'muted'
  if (trend === 'up') return 'red'      // rising rates/BS = restrictive
  if (trend === 'down') return 'green'  // falling rates = easier
  return 'muted'
}

const CSS_VARS: Record<ColorKey, { fg: string; bg: string; dim: string }> = {
  green:  { fg: 'var(--green)',  bg: 'var(--green-bg)',  dim: 'var(--green-dim)' },
  yellow: { fg: 'var(--yellow)', bg: 'var(--yellow-bg)', dim: 'var(--yellow-dim)' },
  red:    { fg: 'var(--red)',    bg: 'var(--red-bg)',    dim: 'var(--red-dim)' },
  blue:   { fg: 'var(--blue)',   bg: 'var(--blue-bg)',   dim: 'var(--blue-dim)' },
  muted:  { fg: 'var(--text-muted)', bg: 'var(--bg-card-raised)', dim: 'var(--border-subtle)' },
}

export function colorVars(key: ColorKey) {
  return CSS_VARS[key]
}
