import type { DashboardState, PlaybookSummary } from '../types/summary'

const BASE_URL = import.meta.env.VITE_API_URL ?? ''

export async function fetchSummary(state: DashboardState): Promise<PlaybookSummary> {
  const response = await fetch(`${BASE_URL}/api/summary`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(state),
  })

  if (!response.ok) {
    const text = await response.text().catch(() => response.statusText)
    throw new Error(`Summary request failed (HTTP ${response.status}): ${text}`)
  }

  return response.json() as Promise<PlaybookSummary>
}
