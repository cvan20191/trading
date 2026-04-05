import type { PlaybookSummary } from '../../types/summary'

interface Props {
  summary: PlaybookSummary
}

interface Section {
  title: string
  accent: string
  items: readonly string[]
  numbered: boolean
}

export function WatchpointsList({ summary }: Props) {
  const sections: Section[] = [
    {
      title: 'Watch Now',
      accent: 'var(--blue)',
      items: summary.watch_now,
      numbered: true,
    },
    {
      title: 'What Changed',
      accent: 'var(--yellow)',
      items: summary.what_changed_bullets,
      numbered: false,
    },
    {
      title: 'What Would Change the Call',
      accent: 'var(--green)',
      items: summary.what_changes_call_bullets,
      numbered: false,
    },
  ]

  return (
    <div style={styles.grid}>
      {sections.map((section) => (
        <div key={section.title} style={styles.card}>
          <div style={{ ...styles.cardTitle, color: section.accent }}>
            {section.title}
          </div>
          <ol style={section.numbered ? styles.orderedList : styles.unorderedList}>
            {section.items.map((item, i) => (
              <li key={i} style={styles.listItem}>
                {!section.numbered && <span style={{ ...styles.bullet, color: section.accent }}>—</span>}
                <span>{item}</span>
              </li>
            ))}
          </ol>
        </div>
      ))}
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
    gap: '12px',
  },
  card: {
    background: 'var(--bg-card)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-md)',
    padding: '18px 20px',
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
  },
  cardTitle: {
    fontSize: '11px',
    fontWeight: 700,
    letterSpacing: '0.08em',
    textTransform: 'uppercase',
  },
  orderedList: {
    listStyle: 'none',
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
    counterReset: 'items',
  },
  unorderedList: {
    listStyle: 'none',
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  listItem: {
    fontSize: '13px',
    lineHeight: 1.55,
    color: 'var(--text-secondary)',
    display: 'flex',
    gap: '8px',
    alignItems: 'flex-start',
  },
  bullet: {
    flexShrink: 0,
    fontWeight: 700,
    marginTop: '1px',
  },
}
