// Curated historical context panel — labeled "Secondary" so it does not
// dominate the macro reading. Replaces ThingsToLookOutForCard in replay mode.

import type { ContextNote } from '../../types/replay'

interface Props {
  notes: ContextNote[]
  asOf: string
}

// Tag color mapping — speaker-relevant tags get colored chips
const TAG_COLORS: Record<string, string> = {
  fed:               'var(--blue, #3b82f6)',
  rates:             'var(--blue, #3b82f6)',
  inflation:         'var(--yellow, #eab308)',
  cpi:               'var(--yellow, #eab308)',
  stagflation:       'var(--yellow, #eab308)',
  'crash-risk':      'var(--red, #ef4444)',
  'systemic-stress': 'var(--red, #ef4444)',
  npl:               'var(--red, #ef4444)',
  valuation:         'var(--purple, #a855f7)',
  liquidity:         'var(--green, #22c55e)',
  'yield-curve':     'var(--blue, #3b82f6)',
  disinflation:      'var(--green, #22c55e)',
}

function tagColor(tag: string): string {
  return TAG_COLORS[tag] ?? 'var(--text-muted)'
}

export function ReplayContextPanel({ notes, asOf }: Props) {
  if (notes.length === 0) {
    return (
      <div style={s.wrapper}>
        <div style={s.header}>
          <span style={s.badge}>Historical Context — Secondary</span>
          <h3 style={s.title}>Relevant Context for {asOf}</h3>
        </div>
        <p style={s.empty}>
          No curated context notes available for this date. The macro engine reading stands on its own.
        </p>
      </div>
    )
  }

  return (
    <div style={s.wrapper}>
      <div style={s.header}>
        <span style={s.badge}>Historical Context — Secondary</span>
        <h3 style={s.title}>Relevant Context for {asOf}</h3>
        <p style={s.subtitle}>
          These notes explain the narrative of the period. They are supplementary — the
          macro engine reading is primary.
        </p>
      </div>

      <div style={s.notes}>
        {notes.map((note, i) => (
          <div key={i} style={s.note}>
            <div style={s.noteHeader}>
              <span style={s.noteTitle}>{note.title}</span>
              <div style={s.tags}>
                {note.tags.map((t) => (
                  <span key={t} style={{ ...s.tag, color: tagColor(t), borderColor: tagColor(t) }}>
                    {t}
                  </span>
                ))}
              </div>
            </div>
            <p style={s.noteBody}>{note.body}</p>
          </div>
        ))}
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
    gridColumn: '1 / -1',
  },
  header: {
    marginBottom: '16px',
  },
  badge: {
    display: 'inline-block',
    fontSize: '10px',
    fontWeight: 700,
    textTransform: 'uppercase' as const,
    letterSpacing: '0.08em',
    color: 'var(--text-muted)',
    border: '1px solid var(--border)',
    borderRadius: '4px',
    padding: '2px 7px',
    marginBottom: '8px',
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
  empty: {
    fontSize: '13px',
    color: 'var(--text-muted)',
    fontStyle: 'italic' as const,
    margin: 0,
  },
  notes: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '14px',
  },
  note: {
    background: 'var(--bg-card-raised)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-md)',
    padding: '14px 16px',
  },
  noteHeader: {
    display: 'flex',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    gap: '10px',
    marginBottom: '8px',
    flexWrap: 'wrap' as const,
  },
  noteTitle: {
    fontSize: '13px',
    fontWeight: 700,
    color: 'var(--text)',
  },
  tags: {
    display: 'flex',
    gap: '5px',
    flexWrap: 'wrap' as const,
  },
  tag: {
    fontSize: '10px',
    fontWeight: 600,
    padding: '1px 6px',
    borderRadius: '3px',
    border: '1px solid',
    opacity: 0.85,
  },
  noteBody: {
    margin: 0,
    fontSize: '12px',
    color: 'var(--text-muted)',
    lineHeight: 1.65,
  },
}
