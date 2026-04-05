import type { PlaybookSummary } from '../../types/summary'

interface Props {
  summary: PlaybookSummary
}

export function TeachingNote({ summary }: Props) {
  return (
    <div style={styles.container}>
      <span style={styles.icon} aria-hidden="true">
        ◈
      </span>
      <p style={styles.text}>{summary.teaching_note}</p>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    background: 'var(--blue-bg)',
    border: '1px solid var(--blue-dim)',
    borderRadius: 'var(--radius-md)',
    padding: '14px 18px',
    display: 'flex',
    gap: '12px',
    alignItems: 'flex-start',
  },
  icon: {
    color: 'var(--blue)',
    fontSize: '16px',
    flexShrink: 0,
    marginTop: '2px',
    opacity: 0.8,
  },
  text: {
    fontSize: '13px',
    lineHeight: 1.65,
    color: 'var(--text-secondary)',
    fontStyle: 'italic',
  },
}
