import type { IndicatorSnapshot } from '../types/indicator'
import type { DashboardState } from '../types/summary'
import type {
  LivePlaybookResponse,
  LiveSnapshotResponse,
  PlaybookResponse,
} from '../types/playbook'

const BASE_URL = import.meta.env.VITE_API_URL ?? ''

// ---------------------------------------------------------------------------
// Mock-mode endpoints (POST with supplied IndicatorSnapshot)
// ---------------------------------------------------------------------------

export async function fetchPlaybook(snapshot: IndicatorSnapshot): Promise<PlaybookResponse> {
  const response = await fetch(`${BASE_URL}/api/playbook`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(snapshot),
  })
  if (!response.ok) {
    const text = await response.text().catch(() => response.statusText)
    throw new Error(`Playbook request failed (HTTP ${response.status}): ${text}`)
  }
  return response.json() as Promise<PlaybookResponse>
}

export async function fetchDashboardState(snapshot: IndicatorSnapshot): Promise<DashboardState> {
  const response = await fetch(`${BASE_URL}/api/dashboard-state`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(snapshot),
  })
  if (!response.ok) {
    const text = await response.text().catch(() => response.statusText)
    throw new Error(`Dashboard-state request failed (HTTP ${response.status}): ${text}`)
  }
  return response.json() as Promise<DashboardState>
}

// ---------------------------------------------------------------------------
// Live-mode endpoints (GET, backend fetches real data from FRED / Yahoo)
// ---------------------------------------------------------------------------

export async function fetchLivePlaybook(
  forceRefresh = false,
  pmiManufacturing: number | null = null,
  pmiServices: number | null = null,
): Promise<LivePlaybookResponse> {
  const params = new URLSearchParams()
  if (forceRefresh) params.set('force_refresh', 'true')
  if (Number.isFinite(pmiManufacturing)) params.set('pmi_manufacturing', String(pmiManufacturing))
  if (Number.isFinite(pmiServices)) params.set('pmi_services', String(pmiServices))
  const qs = params.toString()
  const url = `${BASE_URL}/api/live/playbook${qs ? `?${qs}` : ''}`
  const response = await fetch(url)
  if (!response.ok) {
    const text = await response.text().catch(() => response.statusText)
    throw new Error(`Live playbook request failed (HTTP ${response.status}): ${text}`)
  }
  return response.json() as Promise<LivePlaybookResponse>
}

export async function fetchLiveSnapshot(
  forceRefresh = false,
): Promise<LiveSnapshotResponse> {
  const url = `${BASE_URL}/api/live/snapshot${forceRefresh ? '?force_refresh=true' : ''}`
  const response = await fetch(url)
  if (!response.ok) {
    const text = await response.text().catch(() => response.statusText)
    throw new Error(`Live snapshot request failed (HTTP ${response.status}): ${text}`)
  }
  return response.json() as Promise<LiveSnapshotResponse>
}

export async function triggerManualRefresh(): Promise<{
  status: string
  overall_status: string
  generated_at: string
  stale_count: string
}> {
  const response = await fetch(`${BASE_URL}/api/live/refresh`, { method: 'POST' })
  if (!response.ok) {
    const text = await response.text().catch(() => response.statusText)
    throw new Error(`Manual refresh failed (HTTP ${response.status}): ${text}`)
  }
  return response.json()
}
