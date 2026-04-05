import type { OutcomeReview, ReplayPlaybookResponse } from '../types/replay'

const BASE_URL = import.meta.env.VITE_API_URL ?? ''

export async function fetchReplayPlaybook(asOf: string): Promise<ReplayPlaybookResponse> {
  const url = `${BASE_URL}/api/replay/playbook?as_of=${encodeURIComponent(asOf)}`
  const response = await fetch(url)
  if (!response.ok) {
    const text = await response.text().catch(() => response.statusText)
    throw new Error(`Replay playbook request failed (HTTP ${response.status}): ${text}`)
  }
  return response.json() as Promise<ReplayPlaybookResponse>
}

export async function fetchReplayOutcomes(asOf: string): Promise<OutcomeReview> {
  const url = `${BASE_URL}/api/replay/outcomes?as_of=${encodeURIComponent(asOf)}`
  const response = await fetch(url)
  if (!response.ok) {
    const text = await response.text().catch(() => response.statusText)
    throw new Error(`Replay outcomes request failed (HTTP ${response.status}): ${text}`)
  }
  return response.json() as Promise<OutcomeReview>
}
