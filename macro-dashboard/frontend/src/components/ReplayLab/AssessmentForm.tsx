// Structured assessment form for the Replay Lab blind mode.
// Lets the user record their regime/posture judgment before revealing the app's computed answer.

import { useState } from 'react'
import type { ReplayAssessment } from '../../types/replay'

// The 8 regime labels from regime.py — must match backend exactly
const REGIME_OPTIONS = [
  '',
  'Goldilocks',
  'Overheating',
  'Stagflation Risk',
  'Deflation Risk',
  'Liquidity Transition',
  'Fed Pivot Incoming',
  'Crisis / Crash Watch',
  'Recovery Runway',
]

interface Props {
  assessment: ReplayAssessment
  onUpdate: (partial: Partial<Omit<ReplayAssessment, 'date' | 'saved_at'>>) => void
  onSave: () => void
  onReveal: () => void
  saved: boolean
}

export function AssessmentForm({ assessment, onUpdate, onSave, onReveal, saved }: Props) {
  const [justSaved, setJustSaved] = useState(false)

  function handleSave() {
    onSave()
    setJustSaved(true)
    setTimeout(() => setJustSaved(false), 2000)
  }

  function setWatchpoint(idx: number, value: string) {
    const wp = [...assessment.watchpoints] as [string, string, string]
    wp[idx] = value
    onUpdate({ watchpoints: wp })
  }

  return (
    <div style={s.wrapper}>
      <div style={s.header}>
        <h3 style={s.title}>Your Assessment</h3>
        <p style={s.subtitle}>
          Fill in your macro reading before revealing the app's computed playbook.
          Comparing your judgment to the engine is the learning loop.
        </p>
      </div>

      {/* Regime */}
      <Field label="Regime">
        <select
          value={assessment.regime}
          onChange={(e) => onUpdate({ regime: e.target.value })}
          style={s.select}
        >
          {REGIME_OPTIONS.map((o) => (
            <option key={o} value={o}>{o || '— Select regime —'}</option>
          ))}
        </select>
      </Field>

      {/* Posture */}
      <Field label="Posture">
        <textarea
          value={assessment.posture}
          onChange={(e) => onUpdate({ posture: e.target.value })}
          placeholder="E.g. Hold quality equities, reduce bond duration, watch credit spreads…"
          rows={2}
          style={s.textarea}
        />
      </Field>

      {/* Watchpoints */}
      <Field label="Top Watchpoints">
        {([0, 1, 2] as const).map((idx) => (
          <input
            key={idx}
            type="text"
            value={assessment.watchpoints[idx]}
            onChange={(e) => setWatchpoint(idx, e.target.value)}
            placeholder={`Watchpoint ${idx + 1}…`}
            style={{ ...s.input, marginBottom: idx < 2 ? '6px' : 0 }}
          />
        ))}
      </Field>

      {/* Confidence */}
      <Field label="Confidence">
        <div style={s.radioGroup}>
          {(['High', 'Medium', 'Low'] as const).map((level) => (
            <label key={level} style={s.radioLabel}>
              <input
                type="radio"
                name="confidence"
                value={level}
                checked={assessment.confidence === level}
                onChange={() => onUpdate({ confidence: level })}
                style={{ marginRight: '5px' }}
              />
              {level}
            </label>
          ))}
        </div>
      </Field>

      {/* Free notes */}
      <Field label="Notes (optional)">
        <textarea
          value={assessment.notes}
          onChange={(e) => onUpdate({ notes: e.target.value })}
          placeholder="Any additional thoughts or observations…"
          rows={3}
          style={s.textarea}
        />
      </Field>

      {/* Actions */}
      <div style={s.actions}>
        <button onClick={handleSave} style={s.saveBtn}>
          {justSaved ? '✓ Saved' : 'Save Assessment'}
        </button>
        <button onClick={onReveal} style={s.revealBtn}>
          Reveal Playbook →
        </button>
      </div>

      {saved && !justSaved && (
        <p style={s.savedNote}>
          Assessment saved. You can update it at any time.
        </p>
      )}
    </div>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: '14px' }}>
      <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '6px' }}>
        {label}
      </div>
      {children}
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
    marginBottom: '18px',
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
    lineHeight: 1.5,
  },
  select: {
    width: '100%',
    background: 'var(--bg-card-raised)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-sm)',
    color: 'var(--text)',
    fontSize: '13px',
    padding: '7px 10px',
    cursor: 'pointer',
    outline: 'none',
  },
  textarea: {
    width: '100%',
    background: 'var(--bg-card-raised)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-sm)',
    color: 'var(--text)',
    fontSize: '13px',
    padding: '7px 10px',
    resize: 'vertical' as const,
    outline: 'none',
    fontFamily: 'inherit',
    lineHeight: 1.5,
    boxSizing: 'border-box' as const,
  },
  input: {
    width: '100%',
    background: 'var(--bg-card-raised)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-sm)',
    color: 'var(--text)',
    fontSize: '13px',
    padding: '7px 10px',
    outline: 'none',
    boxSizing: 'border-box' as const,
  },
  radioGroup: {
    display: 'flex',
    gap: '20px',
  },
  radioLabel: {
    fontSize: '13px',
    color: 'var(--text)',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
  },
  actions: {
    display: 'flex',
    gap: '10px',
    marginTop: '18px',
  },
  saveBtn: {
    background: 'var(--bg-card-raised)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-sm)',
    color: 'var(--text)',
    cursor: 'pointer',
    fontSize: '13px',
    fontWeight: 600,
    padding: '8px 16px',
    transition: 'all 0.15s',
  },
  revealBtn: {
    background: 'var(--blue, #3b82f6)',
    border: 'none',
    borderRadius: 'var(--radius-sm)',
    color: '#fff',
    cursor: 'pointer',
    fontSize: '13px',
    fontWeight: 700,
    padding: '8px 20px',
    transition: 'all 0.15s',
  },
  savedNote: {
    margin: '8px 0 0',
    fontSize: '11px',
    color: 'var(--text-muted)',
  },
}
