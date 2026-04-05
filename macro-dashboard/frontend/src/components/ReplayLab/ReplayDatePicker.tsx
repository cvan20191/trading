// Date picker with preset macro-moment buttons for the Replay Lab.
// Custom date input uses local state — only fires onChange when the user
// explicitly clicks "Load". Preset buttons fire immediately on click.

import { useEffect, useState } from 'react'

interface Props {
  selectedDate: string
  onChange: (date: string) => void
  disabled?: boolean
}

// Preset macro moments with brief context labels
const PRESETS = [
  { date: '2020-03-20', label: 'Mar 2020 — COVID crash' },
  { date: '2021-12-01', label: 'Dec 2021 — Inflation pivot' },
  { date: '2022-07-15', label: 'Jul 2022 — Max illiquidity' },
  { date: '2022-10-14', label: 'Oct 2022 — Peak CPI' },
  { date: '2023-03-17', label: 'Mar 2023 — SVB collapse' },
  { date: '2024-01-15', label: 'Jan 2024 — Soft landing' },
]

const MIN_DATE = '2015-01-01'
const yesterday = (() => {
  const d = new Date()
  d.setDate(d.getDate() - 1)
  return d.toISOString().split('T')[0]
})()

export function ReplayDatePicker({ selectedDate, onChange, disabled }: Props) {
  // Local draft state — decoupled from the submitted selectedDate
  const [draftDate, setDraftDate] = useState(selectedDate)

  // Keep draft in sync when a preset is clicked (parent updates selectedDate)
  useEffect(() => {
    setDraftDate(selectedDate)
  }, [selectedDate])

  const isDraftNew = draftDate !== selectedDate
  const isDraftValid = draftDate >= MIN_DATE && draftDate <= yesterday

  function handleLoad() {
    if (isDraftValid) onChange(draftDate)
  }

  function handlePreset(date: string) {
    setDraftDate(date)
    onChange(date)
  }

  return (
    <div style={s.wrapper}>
      <div style={s.header}>
        <h3 style={s.title}>Choose a Date to Replay</h3>
        <p style={s.subtitle}>
          The dashboard will freeze to what was knowable on that date.
        </p>
      </div>

      {/* Custom date input + Load button */}
      <div style={s.inputRow}>
        <label style={s.label} htmlFor="replay-date-input">
          Custom date
        </label>
        <input
          id="replay-date-input"
          type="date"
          value={draftDate}
          min={MIN_DATE}
          max={yesterday}
          onChange={(e) => setDraftDate(e.target.value)}
          disabled={disabled}
          style={s.input}
        />
        <button
          onClick={handleLoad}
          disabled={disabled || !isDraftValid || !isDraftNew}
          style={{
            ...s.loadBtn,
            ...((!isDraftNew || !isDraftValid || disabled) ? s.loadBtnDisabled : {}),
          }}
        >
          Load
        </button>
      </div>

      {/* Preset buttons — fire immediately on click */}
      <div style={s.presetsSection}>
        <span style={s.presetsLabel}>Macro moments</span>
        <div style={s.presets}>
          {PRESETS.map((p) => (
            <button
              key={p.date}
              onClick={() => handlePreset(p.date)}
              disabled={disabled}
              style={{
                ...s.preset,
                ...(selectedDate === p.date ? s.presetActive : {}),
              }}
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}

const s: Record<string, React.CSSProperties> = {
  wrapper: {
    background: 'var(--bg-card)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-lg)',
    padding: '20px 24px',
  },
  header: {
    marginBottom: '16px',
  },
  title: {
    margin: 0,
    fontSize: '15px',
    fontWeight: 700,
    color: 'var(--text)',
  },
  subtitle: {
    margin: '4px 0 0',
    fontSize: '12px',
    color: 'var(--text-muted)',
  },
  inputRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    marginBottom: '16px',
  },
  label: {
    fontSize: '12px',
    color: 'var(--text-muted)',
    whiteSpace: 'nowrap' as const,
  },
  input: {
    background: 'var(--bg-card-raised)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-sm)',
    color: 'var(--text)',
    fontSize: '13px',
    padding: '6px 10px',
    cursor: 'pointer',
    outline: 'none',
  },
  loadBtn: {
    background: 'var(--blue, #3b82f6)',
    border: 'none',
    borderRadius: 'var(--radius-sm)',
    color: '#fff',
    cursor: 'pointer',
    fontSize: '13px',
    fontWeight: 700,
    padding: '6px 16px',
    transition: 'opacity 0.15s',
    whiteSpace: 'nowrap' as const,
  },
  loadBtnDisabled: {
    opacity: 0.35,
    cursor: 'not-allowed',
  },
  presetsSection: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '8px',
  },
  presetsLabel: {
    fontSize: '11px',
    color: 'var(--text-muted)',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.06em',
  },
  presets: {
    display: 'flex',
    flexWrap: 'wrap' as const,
    gap: '6px',
  },
  preset: {
    background: 'var(--bg-card-raised)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-sm)',
    color: 'var(--text-muted)',
    cursor: 'pointer',
    fontSize: '11px',
    fontWeight: 500,
    padding: '5px 10px',
    transition: 'all 0.15s',
  },
  presetActive: {
    background: 'var(--blue-bg, #1e3a5f)',
    border: '1px solid var(--blue, #3b82f6)',
    color: 'var(--blue, #3b82f6)',
  },
}
