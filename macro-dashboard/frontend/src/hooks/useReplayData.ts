import { useCallback, useEffect, useState } from 'react'
import { fetchReplayOutcomes, fetchReplayPlaybook } from '../api/replay'
import type { OutcomeReview, ReplayAssessment, ReplayPlaybookResponse } from '../types/replay'

const STORAGE_KEY = (date: string) => `replay_assessment_${date}`

function loadAssessment(date: string): ReplayAssessment | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY(date))
    if (!raw) return null
    return JSON.parse(raw) as ReplayAssessment
  } catch {
    return null
  }
}

function saveAssessment(assessment: ReplayAssessment): void {
  try {
    localStorage.setItem(STORAGE_KEY(assessment.date), JSON.stringify(assessment))
  } catch {
    // Storage quota exceeded or unavailable — fail silently
  }
}

interface UseReplayDataReturn {
  selectedDate: string
  setSelectedDate: (date: string) => void
  playbook: ReplayPlaybookResponse | null
  outcomes: OutcomeReview | null
  assessment: ReplayAssessment | null
  revealed: boolean
  loadingPlaybook: boolean
  loadingOutcomes: boolean
  errorPlaybook: string | null
  errorOutcomes: string | null
  reveal: () => void
  resetForDate: (date: string) => void
  updateAssessment: (partial: Partial<Omit<ReplayAssessment, 'date' | 'saved_at'>>) => void
  saveCurrentAssessment: () => void
}

// Default blank assessment for a given date
function blankAssessment(date: string): ReplayAssessment {
  return {
    date,
    regime: '',
    posture: '',
    watchpoints: ['', '', ''],
    confidence: 'Medium',
    notes: '',
    saved_at: '',
    revealed: false,
  }
}

export function useReplayData(initialDate: string): UseReplayDataReturn {
  const [selectedDate, setSelectedDateState] = useState<string>(initialDate)
  const [playbook, setPlaybook] = useState<ReplayPlaybookResponse | null>(null)
  const [outcomes, setOutcomes] = useState<OutcomeReview | null>(null)
  const [assessment, setAssessment] = useState<ReplayAssessment | null>(null)
  const [revealed, setRevealed] = useState<boolean>(false)
  const [loadingPlaybook, setLoadingPlaybook] = useState<boolean>(false)
  const [loadingOutcomes, setLoadingOutcomes] = useState<boolean>(false)
  const [errorPlaybook, setErrorPlaybook] = useState<string | null>(null)
  const [errorOutcomes, setErrorOutcomes] = useState<string | null>(null)

  // Load assessment from localStorage and reset state when date changes
  const resetForDate = useCallback((date: string) => {
    setPlaybook(null)
    setOutcomes(null)
    setErrorPlaybook(null)
    setErrorOutcomes(null)
    const saved = loadAssessment(date)
    setAssessment(saved ?? blankAssessment(date))
    setRevealed(saved?.revealed ?? false)
  }, [])

  const setSelectedDate = useCallback((date: string) => {
    setSelectedDateState(date)
    resetForDate(date)
  }, [resetForDate])

  // Fetch playbook when date changes
  useEffect(() => {
    if (!selectedDate) return

    let cancelled = false
    setLoadingPlaybook(true)
    setErrorPlaybook(null)

    fetchReplayPlaybook(selectedDate)
      .then((data) => {
        if (!cancelled) {
          setPlaybook(data)
          setLoadingPlaybook(false)
        }
      })
      .catch((err: Error) => {
        if (!cancelled) {
          setErrorPlaybook(err.message)
          setLoadingPlaybook(false)
        }
      })

    return () => {
      cancelled = true
    }
  }, [selectedDate])

  // Fetch outcomes lazily when revealed (avoids spoiling results before reveal)
  const reveal = useCallback(() => {
    setRevealed(true)

    if (outcomes || loadingOutcomes) return
    setLoadingOutcomes(true)
    setErrorOutcomes(null)

    fetchReplayOutcomes(selectedDate)
      .then((data) => {
        setOutcomes(data)
        setLoadingOutcomes(false)
      })
      .catch((err: Error) => {
        setErrorOutcomes(err.message)
        setLoadingOutcomes(false)
      })
  }, [selectedDate, outcomes, loadingOutcomes])

  // Update the in-memory assessment (does NOT auto-save)
  const updateAssessment = useCallback(
    (partial: Partial<Omit<ReplayAssessment, 'date' | 'saved_at'>>) => {
      setAssessment((prev) => {
        const base = prev ?? blankAssessment(selectedDate)
        return { ...base, ...partial }
      })
    },
    [selectedDate],
  )

  // Persist the current assessment to localStorage
  const saveCurrentAssessment = useCallback(() => {
    setAssessment((prev) => {
      const base = prev ?? blankAssessment(selectedDate)
      const toSave: ReplayAssessment = {
        ...base,
        saved_at: new Date().toISOString(),
        revealed,
      }
      saveAssessment(toSave)
      return toSave
    })
  }, [selectedDate, revealed])

  // Initialise assessment for the initial date on mount
  useEffect(() => {
    resetForDate(initialDate)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return {
    selectedDate,
    setSelectedDate,
    playbook,
    outcomes,
    assessment,
    revealed,
    loadingPlaybook,
    loadingOutcomes,
    errorPlaybook,
    errorOutcomes,
    reveal,
    resetForDate,
    updateAssessment,
    saveCurrentAssessment,
  }
}
