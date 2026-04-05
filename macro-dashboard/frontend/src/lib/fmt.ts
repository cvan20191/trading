// Formatting helpers for the dashboard

export function na(value: number | string | null | undefined, suffix = ''): string {
  if (value === null || value === undefined) return 'N/A'
  if (typeof value === 'number' && isNaN(value)) return 'N/A'
  return `${value}${suffix}`
}

export function fmtNum(value: number | null | undefined, decimals = 2, suffix = ''): string {
  if (value === null || value === undefined) return 'N/A'
  return `${value.toFixed(decimals)}${suffix}`
}

export function fmtPct(value: number | null | undefined, decimals = 1): string {
  return fmtNum(value, decimals, '%')
}

export function fmtPE(value: number | null | undefined): string {
  if (value === null || value === undefined) return 'N/A'
  return `${value.toFixed(1)}×`
}

export function fmtTime(isoString: string | null | undefined): string {
  if (!isoString) return '—'
  try {
    const d = new Date(isoString)
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  } catch {
    return '—'
  }
}

export function fmtDate(isoString: string | null | undefined): string {
  if (!isoString) return '—'
  try {
    const d = new Date(isoString)
    return d.toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' })
  } catch {
    return '—'
  }
}

export function fmtTrillions(value: number | null | undefined): string {
  if (value === null || value === undefined) return 'N/A'
  // WALCL is in millions
  if (value > 1_000_000) return `$${(value / 1_000_000).toFixed(2)}T`
  if (value > 1_000) return `$${(value / 1_000).toFixed(2)}B`
  return `$${value.toFixed(0)}`
}

export function trendLabel(trend: string | null | undefined): string {
  if (!trend) return '—'
  if (trend === 'up') return '↑'
  if (trend === 'down') return '↓'
  if (trend === 'flat') return '→'
  return '—'
}

export function rallyLabel(score: number | null | undefined): string {
  if (score === null || score === undefined) return 'Unknown'
  if (score >= 65) return 'High'
  if (score >= 40) return 'Medium'
  return 'Low'
}
