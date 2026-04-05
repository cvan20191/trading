import type { PlaybookSummary } from '../../types/summary'

interface Props {
  summary: PlaybookSummary
}

function formatTime(iso: string): string {
  try {
    return new Date(iso).toLocaleTimeString(undefined, {
      hour: '2-digit',
      minute: '2-digit',
      timeZoneName: 'short',
    })
  } catch {
    return iso
  }
}

export function SummaryMetaBadge({ summary }: Props) {
  const { meta } = summary
  const isFallback = meta.used_fallback

  return (
    <div style={styles.container}>
      {isFallback && (
        <span style={{ ...styles.chip, ...styles.fallbackChip }}>
          Deterministic fallback
        </span>
      )}
      <span style={styles.chip}>
        Data: {meta.data_status}
      </span>
      {meta.model && (
        <span style={styles.chip}>
          {meta.model}
        </span>
      )}
      <span style={styles.chip}>
        Generated {formatTime(meta.generated_at)}
      </span>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '6px',
    alignItems: 'center',
    paddingTop: '4px',
  },
  chip: {
    fontSize: '10px',
    fontWeight: 500,
    letterSpacing: '0.03em',
    color: 'var(--text-muted)',
    background: 'var(--bg-card)',
    border: '1px solid var(--border-subtle)',
    borderRadius: '4px',
    padding: '2px 7px',
  },
  fallbackChip: {
    color: 'var(--yellow)',
    background: 'var(--yellow-bg)',
    border: '1px solid var(--yellow-dim)',
  },
}
