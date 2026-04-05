import { useState } from 'react'
import type { SourceMeta } from '../../types/playbook'

interface Props {
  overall_status: string
  stale_series: string[]
  generated_at: string
  sources?: Record<string, SourceMeta>
  onRefresh?: () => void
  refreshing?: boolean
}

const STATUS_COLOR: Record<string, string> = {
  fresh: 'var(--green)',
  mixed: 'var(--yellow)',
  stale: 'var(--red)',
}

const SOURCE_STATUS_COLOR: Record<string, string> = {
  fresh: 'var(--green)',
  stale: 'var(--yellow)',
  missing: 'var(--text-muted)',
  error: 'var(--red)',
  fallback: 'var(--yellow)',
  unknown: 'var(--text-muted)',
}

export function FreshnessPanel({
  overall_status,
  stale_series,
  generated_at,
  sources,
  onRefresh,
  refreshing,
}: Props) {
  const [showSources, setShowSources] = useState(false)

  const statusColor = STATUS_COLOR[overall_status] ?? 'var(--text-muted)'
  const generatedDate = generated_at
    ? new Date(generated_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : '—'

  return (
    <div style={styles.container}>
      {/* Top row: status dot, text, generated time, refresh button */}
      <div style={styles.topRow}>
        <span style={{ ...styles.dot, background: statusColor }} />
        <span style={{ ...styles.statusLabel, color: statusColor }}>
          {overall_status.toUpperCase()}
        </span>
        <span style={styles.meta}>
          {stale_series.length > 0
            ? `${stale_series.length} stale series`
            : 'All series fresh'}
          {' · '}Updated {generatedDate}
        </span>

        <div style={styles.actions}>
          {sources && (
            <button style={styles.btn} onClick={() => setShowSources((v) => !v)}>
              {showSources ? 'Hide sources' : 'Show sources'}
            </button>
          )}
          {onRefresh && (
            <button
              style={{ ...styles.btn, ...styles.refreshBtn }}
              onClick={onRefresh}
              disabled={refreshing}
            >
              {refreshing ? 'Refreshing…' : '↺ Refresh'}
            </button>
          )}
        </div>
      </div>

      {/* Stale series tags */}
      {stale_series.length > 0 && (
        <div style={styles.staleRow}>
          {stale_series.map((s) => (
            <span key={s} style={styles.staleTag}>
              {s}
            </span>
          ))}
        </div>
      )}

      {/* Sources table */}
      {showSources && sources && (
        <div style={styles.sourcesGrid}>
          <div style={styles.sourceHeader}>
            <span>Series</span>
            <span>Provider</span>
            <span>Observed</span>
            <span>Status</span>
            <span>Basis</span>
          </div>
          {Object.entries(sources).map(([key, meta]) => (
            <div key={key} style={styles.sourceRow}>
              <span style={styles.sourceKey}>{key}</span>
              <span style={styles.sourceMeta}>{meta.provider}</span>
              <span style={styles.sourceMeta}>{meta.observed_at ?? '—'}</span>
              <span
                style={{
                  ...styles.sourceMeta,
                  color: SOURCE_STATUS_COLOR[meta.status] ?? 'var(--text-muted)',
                  fontWeight: 600,
                }}
              >
                {meta.status}
              </span>
              <span
                style={{
                  ...styles.sourceMeta,
                  color: meta.basis === 'forward'
                    ? 'var(--green)'
                    : meta.basis && meta.basis !== 'unavailable'
                    ? 'var(--yellow)'
                    : 'var(--text-muted)',
                  fontStyle: meta.basis && meta.basis !== 'forward' && meta.basis !== 'unavailable' ? 'italic' : 'normal',
                }}
              >
                {meta.basis ?? '—'}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    background: 'var(--bg-card)',
    border: '1px solid var(--border-subtle)',
    borderRadius: 'var(--radius-md)',
    padding: '10px 14px',
    fontSize: '11px',
  },
  topRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    flexWrap: 'wrap',
  },
  dot: {
    width: '7px',
    height: '7px',
    borderRadius: '50%',
    flexShrink: 0,
  },
  statusLabel: {
    fontWeight: 700,
    letterSpacing: '0.06em',
    fontSize: '10px',
  },
  meta: {
    color: 'var(--text-muted)',
    fontSize: '11px',
  },
  actions: {
    marginLeft: 'auto',
    display: 'flex',
    gap: '8px',
  },
  btn: {
    background: 'transparent',
    border: '1px solid var(--border)',
    borderRadius: '4px',
    color: 'var(--text-muted)',
    cursor: 'pointer',
    fontSize: '10px',
    padding: '2px 8px',
  },
  refreshBtn: {
    color: 'var(--blue)',
    borderColor: 'var(--blue)',
  },
  staleRow: {
    marginTop: '7px',
    display: 'flex',
    flexWrap: 'wrap',
    gap: '5px',
  },
  staleTag: {
    background: 'var(--yellow-bg, rgba(234,179,8,0.12))',
    color: 'var(--yellow)',
    border: '1px solid var(--yellow-dim, rgba(234,179,8,0.3))',
    borderRadius: '3px',
    padding: '1px 6px',
    fontSize: '10px',
    fontFamily: 'var(--font-mono)',
  },
  sourcesGrid: {
    marginTop: '10px',
    display: 'flex',
    flexDirection: 'column',
    gap: '2px',
  },
  sourceHeader: {
    display: 'grid',
    gridTemplateColumns: '2fr 1.2fr 1.2fr 0.8fr 1fr',
    gap: '8px',
    color: 'var(--text-muted)',
    fontWeight: 600,
    fontSize: '10px',
    letterSpacing: '0.04em',
    textTransform: 'uppercase',
    paddingBottom: '4px',
    borderBottom: '1px solid var(--border-subtle)',
    marginBottom: '4px',
  },
  sourceRow: {
    display: 'grid',
    gridTemplateColumns: '2fr 1.2fr 1.2fr 0.8fr 1fr',
    gap: '8px',
    alignItems: 'center',
  },
  sourceKey: {
    fontFamily: 'var(--font-mono)',
    color: 'var(--text-secondary)',
    fontSize: '11px',
  },
  sourceMeta: {
    color: 'var(--text-muted)',
    fontSize: '11px',
  },
}
