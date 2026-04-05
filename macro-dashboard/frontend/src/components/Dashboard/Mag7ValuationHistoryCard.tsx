import type { CSSProperties } from 'react'
import { Card } from '../ui/Card'
import { HistoricalLineChart } from '../ui/HistoricalLineChart'
import { fmtPE } from '../../lib/fmt'
import cacheData from '../../data/mag7ValuationHistory.json'

type HistoryPoint = { date: string; value: number }

type Mag7ValuationCache = {
  label: string
  metric: string
  basis: string
  source: string
  script_version?: string
  methodology_note: string
  last_refreshed: string | null
  history: HistoryPoint[]
}

const cache = cacheData as Mag7ValuationCache

const STROKE = '#4338ca'

export function Mag7ValuationHistoryCard() {
  const points = cache.history ?? []
  const hasData = points.length > 0
  const singlePointNote = points.length === 1

  return (
    <Card title="Mag 7 valuation — cached history" accent="var(--yellow)">
      <p style={s.copy}>
        {cache.methodology_note || 'Local JSON cache only — no API calls when you load the dashboard.'}{' '}
        This is <strong>context</strong>, not a substitute for the live Big Tech Valuation Trigger above.
      </p>
      <div style={s.meta}>
        <span>
          <strong>Last refreshed:</strong> {cache.last_refreshed ?? '—'}
        </span>
        <span style={s.metaSep}>·</span>
        <span>
          <strong>Points:</strong> {points.length}
        </span>
      </div>
      {!hasData ? (
        <p style={s.empty}>
          No history rows yet. From the repo root run:{' '}
          <code style={s.code}>./backend/.venv/bin/python scripts/refresh_mag7_valuation_history.py</code>
          {' '}(requires <code style={s.code}>FMP_API_KEY</code> in <code style={s.code}>backend/.env</code>).
        </p>
      ) : (
        <>
          {singlePointNote && (
            <p style={s.singlePointNote}>
              Only 1 saved observation so far — this historical chart fills in as you add more dated JSON entries.
            </p>
          )}
          <HistoricalLineChart
            seriesLabel={cache.label}
            points={points}
            stroke={STROKE}
            yTickFormatter={(v) => fmtPE(v)}
            tooltipValueFormatter={(v) => fmtPE(v)}
          />
        </>
      )}
    </Card>
  )
}

const s: Record<string, CSSProperties> = {
  copy: {
    fontSize: '12px',
    color: 'var(--text-muted)',
    lineHeight: 1.55,
    margin: 0,
  },
  meta: {
    marginTop: '8px',
    fontSize: '11px',
    color: 'var(--text-dim)',
    display: 'flex',
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: '6px',
  },
  metaSep: { opacity: 0.5 },
  singlePointNote: {
    margin: '10px 0 0',
    fontSize: '12px',
    color: 'var(--text-secondary)',
    lineHeight: 1.5,
  },
  empty: {
    margin: '12px 0 0',
    fontSize: '12px',
    color: 'var(--text-secondary)',
    lineHeight: 1.5,
  },
  code: {
    fontFamily: 'monospace',
    fontSize: '10px',
    background: 'var(--bg-card-raised)',
    padding: '1px 5px',
    borderRadius: '4px',
    border: '1px solid var(--border-subtle)',
  },
}
