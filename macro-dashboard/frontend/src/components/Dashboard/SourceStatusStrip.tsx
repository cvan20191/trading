import type { CSSProperties } from 'react'
import type { AppMode } from '../../hooks/usePlaybookData'
import type { SourceMeta } from '../../types/playbook'
import { freshnessColor, colorVars } from '../../lib/colors'
import { fmtTime } from '../../lib/fmt'

function staleSeriesDisplayLabel(
  key: string,
  sources?: Record<string, SourceMeta>,
): string {
  const name = sources?.[key]?.series_name?.trim()
  return name && name.length > 0 ? name : key
}

interface Props {
  mode: AppMode
  onSetMode: (m: AppMode) => void
  overallStatus?: string
  staleCount: number
  /** Live API keys flagged stale/missing/error; used for tag row labels. */
  staleSeries?: string[]
  sources?: Record<string, SourceMeta>
  generatedAt?: string
  refreshing: boolean
  onRefresh: () => void
  onReset: () => void
  showDebug: boolean
  onToggleDebug: () => void
}

export function SourceStatusStrip({
  mode,
  onSetMode,
  overallStatus,
  staleCount,
  staleSeries = [],
  sources,
  generatedAt,
  refreshing,
  onRefresh,
  onReset,
  showDebug,
  onToggleDebug,
}: Props) {
  const color = colorVars(freshnessColor(mode === 'live' ? overallStatus : 'fresh'))
  const isDegraded = mode === 'live' && overallStatus === 'stale'
  const isMixed = mode === 'live' && overallStatus === 'mixed'
  const staleTagsTitle = staleSeries
    .map((k) => `${k}: ${staleSeriesDisplayLabel(k, sources)}`)
    .join(' · ')

  return (
    <div style={s.wrapper}>
      <div style={s.inner}>
        {/* Left: status + optional stale tag row */}
        <div style={s.leftColumn}>
          <div style={s.left}>
            <span style={{ ...s.dot, background: color.fg }} />
            <span style={{ fontSize: '11px', color: color.fg, fontWeight: 700, letterSpacing: '0.05em' }}>
              {mode === 'live' ? (overallStatus ?? 'live').toUpperCase() : 'MOCK'}
            </span>
            {mode === 'live' && generatedAt && (
              <span style={s.meta}>Updated {fmtTime(generatedAt)}</span>
            )}
            {mode === 'live' && staleCount > 0 && (
              <span style={s.staleBadge}>{staleCount} stale series</span>
            )}
            {(isDegraded || isMixed) && (
              <span style={s.degradedNote}>
                {isDegraded
                  ? 'Multiple live series unavailable — running in degraded mode.'
                  : 'Some live series are stale; data may lag recent events.'}
              </span>
            )}
          </div>
          {mode === 'live' && staleSeries.length > 0 && (
            <div style={s.staleTagsRow} title={staleTagsTitle}>
              {staleSeries.map((key) => (
                <span key={key} style={s.staleTag} title={key}>
                  {staleSeriesDisplayLabel(key, sources)}
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Right: controls */}
        <div style={s.controls}>
          {/* Live / Mock toggle */}
          <div style={s.toggleGroup}>
            <button
              style={{ ...s.toggleBtn, ...(mode === 'live' ? s.toggleActive : {}) }}
              onClick={() => onSetMode('live')}
            >Live</button>
            <button
              style={{ ...s.toggleBtn, ...(mode === 'mock' ? s.toggleActive : {}) }}
              onClick={() => onSetMode('mock')}
            >Mock</button>
          </div>

          {mode === 'live' && (
            <button style={s.refreshBtn} onClick={onRefresh} disabled={refreshing}>
              {refreshing ? '⟳ …' : '⟳ Refresh'}
            </button>
          )}
          <button style={s.resetBtn} onClick={onReset} disabled={refreshing} title="Clear data and reload from scratch">
            Reset
          </button>

          <button
            style={{ ...s.debugBtn, ...(showDebug ? s.debugActive : {}) }}
            onClick={onToggleDebug}
          >
            {showDebug ? 'Hide Debug' : 'Debug'}
          </button>
        </div>
      </div>
    </div>
  )
}

const s: Record<string, CSSProperties> = {
  wrapper: {
    background: 'var(--bg-card)',
    borderBottom: '1px solid var(--border)',
    position: 'sticky',
    top: 0,
    zIndex: 10,
  },
  inner: {
    maxWidth: '1200px',
    margin: '0 auto',
    padding: '8px 20px',
    display: 'flex',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    gap: '12px',
    flexWrap: 'wrap',
  },
  leftColumn: {
    flex: '1 1 200px',
    minWidth: 0,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'flex-start',
    gap: '6px',
  },
  left: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    flexWrap: 'wrap',
  },
  staleTagsRow: {
    display: 'flex',
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: '6px',
    width: '100%',
  },
  staleTag: {
    fontSize: '10px',
    lineHeight: 1.3,
    background: 'var(--bg-card-raised)',
    color: 'var(--text-secondary)',
    border: '1px solid var(--border-subtle)',
    borderRadius: '3px',
    padding: '2px 6px',
    maxWidth: '100%',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  dot: {
    width: '7px', height: '7px', borderRadius: '50%', flexShrink: 0,
  },
  meta: {
    fontSize: '11px', color: 'var(--text-muted)',
  },
  staleBadge: {
    fontSize: '10px',
    background: 'var(--yellow-bg)',
    color: 'var(--yellow)',
    border: '1px solid var(--yellow-dim)',
    borderRadius: '3px',
    padding: '1px 7px',
  },
  degradedNote: {
    fontSize: '11px',
    color: 'var(--text-muted)',
    fontStyle: 'italic',
  },
  controls: {
    display: 'flex', alignItems: 'center', gap: '8px',
  },
  toggleGroup: {
    display: 'flex', border: '1px solid var(--border)', borderRadius: '5px', overflow: 'hidden',
  },
  toggleBtn: {
    padding: '3px 12px',
    fontSize: '11px',
    fontWeight: 600,
    background: 'transparent',
    border: 'none',
    color: 'var(--text-muted)',
    cursor: 'pointer',
  },
  toggleActive: {
    background: 'var(--blue)',
    color: '#fff',
  },
  refreshBtn: {
    padding: '3px 10px',
    fontSize: '11px',
    fontWeight: 600,
    background: 'transparent',
    border: '1px solid var(--blue-dim)',
    borderRadius: '4px',
    color: 'var(--blue)',
    cursor: 'pointer',
  },
  resetBtn: {
    padding: '3px 10px',
    fontSize: '11px',
    fontWeight: 600,
    background: 'transparent',
    border: '1px solid var(--border)',
    borderRadius: '4px',
    color: 'var(--text-muted)',
    cursor: 'pointer',
  },
  debugBtn: {
    padding: '3px 10px',
    fontSize: '11px',
    background: 'transparent',
    border: '1px solid var(--border)',
    borderRadius: '4px',
    color: 'var(--text-muted)',
    cursor: 'pointer',
  },
  debugActive: {
    background: 'var(--purple-bg)',
    borderColor: 'var(--purple)',
    color: 'var(--purple)',
  },
}
