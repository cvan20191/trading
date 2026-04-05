import { useEffect, useMemo, useRef, useState } from 'react'
import { Card } from '../ui/Card'
import {
  HistoricalLineChart,
  nearestHistoryPoint,
  parseDateMs,
} from '../ui/HistoricalLineChart'
import overviewData from '../../data/fedLiquidityOverview.json'

type RangeKey = '5Y' | '10Y' | 'All'

type Point = { date: string; value: number }

type LeverData = {
  label: string
  series_id: string
  latest_date: string
  latest_value: number
  unit: string
  next_release_date: string
  history: Point[]
}

type OverviewData = {
  fed_balance_sheet: LeverData
  fed_rate: LeverData
}

const data = overviewData as OverviewData

function formatUtcDate(ms: number): string {
  const d = new Date(ms)
  const y = d.getUTCFullYear()
  const mo = String(d.getUTCMonth() + 1).padStart(2, '0')
  const day = String(d.getUTCDate()).padStart(2, '0')
  return `${y}-${mo}-${day}`
}

function endOfHistoryMs(lever: LeverData): number {
  const h = lever.history
  if (h.length === 0) return parseDateMs(lever.latest_date)
  return parseDateMs(h[h.length - 1].date)
}

function filterByRange(points: Point[], range: RangeKey, anchorMs: number): Point[] {
  if (range === 'All' || points.length === 0) return points
  const years = range === '5Y' ? 5 : 10
  const a = new Date(anchorMs)
  const startMs = Date.UTC(a.getUTCFullYear() - years, a.getUTCMonth(), a.getUTCDate())
  return points.filter((p) => {
    const t = parseDateMs(p.date)
    return t >= startMs && t <= anchorMs
  })
}

function formatBalanceSheet(value: number): string {
  if (Math.abs(value) >= 1_000_000) return `$${(value / 1_000_000).toFixed(2)}T`
  if (Math.abs(value) >= 1_000) return `$${(value / 1_000).toFixed(1)}B`
  return `$${value.toFixed(0)}M`
}

function formatRate(value: number): string {
  return `${value.toFixed(2)}%`
}

const RANGE_KEYS = ['5Y', '10Y', 'All'] as const

function RangeToolbar({
  range,
  onRangeChange,
  fromDate,
  toDate,
}: {
  range: RangeKey
  onRangeChange: (r: RangeKey) => void
  fromDate: string
  toDate: string
}) {
  return (
    <div style={s.leverToolbar}>
      <div style={s.rangeRow}>
        <span style={s.zoomLabel}>Zoom</span>
        {RANGE_KEYS.map((r) => (
          <button
            key={r}
            type="button"
            onClick={() => onRangeChange(r)}
            style={{
              ...s.rangeBtn,
              ...(range === r ? s.rangeBtnActive : {}),
            }}
          >
            {r}
          </button>
        ))}
      </div>
      <div style={s.dateBoxes}>
        <div style={s.dateBox}>
          <span style={s.dateLabel}>From</span>
          <span>{fromDate}</span>
        </div>
        <div style={s.dateBox}>
          <span style={s.dateLabel}>To</span>
          <span>{toDate}</span>
        </div>
      </div>
    </div>
  )
}

function LeverBlock({
  lever,
  points,
  fromDate,
  toDate,
  showToolbar,
  range,
  onRangeChange,
  accent,
  latestFormatter,
  yTickFormatter,
  tooltipFormatter,
  xDomainMinMs,
  xDomainMaxMs,
  syncHoverTimeMs,
  onSyncHoverChange,
  suppressChartTooltip,
}: {
  lever: LeverData
  points: Point[]
  fromDate: string
  toDate: string
  showToolbar: boolean
  range: RangeKey
  onRangeChange: (r: RangeKey) => void
  accent: string
  latestFormatter: (value: number) => string
  yTickFormatter: (value: number) => string
  tooltipFormatter: (value: number) => string
  xDomainMinMs?: number
  xDomainMaxMs?: number
  syncHoverTimeMs?: number | null
  onSyncHoverChange?: (timeMs: number | null, chartXPct: number) => void
  suppressChartTooltip?: boolean
}) {
  return (
    <div style={s.leverBlock}>
      <div style={s.leverHeader}>
        <div>
          <div style={s.leverTitle}>{lever.label}</div>
          <div style={s.leverMeta}>
            Series: <span style={s.seriesId}>{lever.series_id}</span>
          </div>
        </div>
        <div style={s.rightMeta}>
          <div>
            <strong>Latest:</strong> {latestFormatter(lever.latest_value)} ({lever.latest_date})
          </div>
          <div>
            <strong>Next date to watch:</strong> {lever.next_release_date}
          </div>
        </div>
      </div>
      {showToolbar && (
        <RangeToolbar range={range} onRangeChange={onRangeChange} fromDate={fromDate} toDate={toDate} />
      )}
      <HistoricalLineChart
        seriesLabel={lever.label}
        points={points}
        stroke={accent}
        yTickFormatter={yTickFormatter}
        tooltipValueFormatter={tooltipFormatter}
        xDomainMinMs={xDomainMinMs}
        xDomainMaxMs={xDomainMaxMs}
        syncHoverTimeMs={syncHoverTimeMs}
        onSyncHoverChange={onSyncHoverChange}
        suppressLocalTooltip={suppressChartTooltip}
      />
    </div>
  )
}

export function FedLiquidityOverviewCard() {
  const [syncTimeline, setSyncTimeline] = useState(false)
  const [rangeShared, setRangeShared] = useState<RangeKey>('10Y')
  const [rangeBs, setRangeBs] = useState<RangeKey>('10Y')
  const [rangeRate, setRangeRate] = useState<RangeKey>('10Y')
  const [syncHover, setSyncHover] = useState<{ ms: number; xpct: number } | null>(null)
  const syncLeaveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (!syncTimeline) {
      setSyncHover(null)
      if (syncLeaveTimerRef.current) {
        clearTimeout(syncLeaveTimerRef.current)
        syncLeaveTimerRef.current = null
      }
    }
  }, [syncTimeline])

  useEffect(
    () => () => {
      if (syncLeaveTimerRef.current) clearTimeout(syncLeaveTimerRef.current)
    },
    [],
  )

  const anchorMs = useMemo(
    () => Math.max(endOfHistoryMs(data.fed_balance_sheet), endOfHistoryMs(data.fed_rate)),
    [],
  )

  const syncedView = useMemo(() => {
    const bsH = data.fed_balance_sheet.history
    const rtH = data.fed_rate.history
    if (rangeShared === 'All') {
      const firstBs = bsH.length ? parseDateMs(bsH[0].date) : parseDateMs(data.fed_balance_sheet.latest_date)
      const firstRt = rtH.length ? parseDateMs(rtH[0].date) : parseDateMs(data.fed_rate.latest_date)
      const xMin = Math.min(firstBs, firstRt)
      const xMax = anchorMs
      const bsPoints = bsH.filter((p) => parseDateMs(p.date) <= xMax)
      const rtPoints = rtH.filter((p) => parseDateMs(p.date) <= xMax)
      return {
        bsPoints,
        rtPoints,
        xDomainMinMs: xMin,
        xDomainMaxMs: xMax,
        fromDate: formatUtcDate(xMin),
        toDate: formatUtcDate(xMax),
      }
    }
    const years = rangeShared === '5Y' ? 5 : 10
    const a = new Date(anchorMs)
    const xMin = Date.UTC(a.getUTCFullYear() - years, a.getUTCMonth(), a.getUTCDate())
    const xMax = anchorMs
    const bsPoints = filterByRange(bsH, rangeShared, anchorMs)
    const rtPoints = filterByRange(rtH, rangeShared, anchorMs)
    return {
      bsPoints,
      rtPoints,
      xDomainMinMs: xMin,
      xDomainMaxMs: xMax,
      fromDate: formatUtcDate(xMin),
      toDate: formatUtcDate(xMax),
    }
  }, [rangeShared, anchorMs])

  const asyncBsView = useMemo(() => {
    const anchor = endOfHistoryMs(data.fed_balance_sheet)
    const pts = filterByRange(data.fed_balance_sheet.history, rangeBs, anchor)
    return {
      points: pts,
      fromDate: pts[0]?.date ?? data.fed_balance_sheet.latest_date,
      toDate: pts[pts.length - 1]?.date ?? data.fed_balance_sheet.latest_date,
    }
  }, [rangeBs])

  const asyncRateView = useMemo(() => {
    const anchor = endOfHistoryMs(data.fed_rate)
    const pts = filterByRange(data.fed_rate.history, rangeRate, anchor)
    return {
      points: pts,
      fromDate: pts[0]?.date ?? data.fed_rate.latest_date,
      toDate: pts[pts.length - 1]?.date ?? data.fed_rate.latest_date,
    }
  }, [rangeRate])

  const handleSyncHoverChange = (timeMs: number | null, chartXPct: number) => {
    if (syncLeaveTimerRef.current) {
      clearTimeout(syncLeaveTimerRef.current)
      syncLeaveTimerRef.current = null
    }
    if (timeMs === null) {
      syncLeaveTimerRef.current = setTimeout(() => setSyncHover(null), 120)
    } else {
      setSyncHover({ ms: timeMs, xpct: chartXPct })
    }
  }

  const bsAtHover =
    syncTimeline && syncHover !== null
      ? nearestHistoryPoint(syncedView.bsPoints, syncHover.ms)
      : null
  const rateAtHover =
    syncTimeline && syncHover !== null
      ? nearestHistoryPoint(syncedView.rtPoints, syncHover.ms)
      : null

  return (
    <Card title="Fed Liquidity Overview" accent="var(--blue)">
      <div style={s.copy}>
        The Fed balance sheet and policy rate are the two core liquidity levers. This overview is contextual
        and does not imply precise market timing.
      </div>

      <div style={s.syncRow}>
        <button
          type="button"
          onClick={() => setSyncTimeline((v) => !v)}
          style={{
            ...s.syncBtn,
            ...(syncTimeline ? s.syncBtnOn : {}),
          }}
          aria-pressed={syncTimeline}
        >
          Sync timeline
        </button>
        <span style={s.syncHint}>
          {syncTimeline
            ? 'Both charts share the same date window and From/To.'
            : 'Each chart uses its own zoom range.'}
        </span>
      </div>

      {syncTimeline && (
        <RangeToolbar
          range={rangeShared}
          onRangeChange={setRangeShared}
          fromDate={syncedView.fromDate}
          toDate={syncedView.toDate}
        />
      )}

      <div
        style={{
          ...s.stackWrap,
          ...(syncTimeline ? { paddingTop: '56px' } : {}),
        }}
      >
        {syncTimeline && syncHover !== null && bsAtHover && rateAtHover && (
          <div
            style={{
              ...s.syncTooltip,
              left: `${Math.min(92, Math.max(8, syncHover.xpct))}%`,
            }}
          >
            <div style={s.syncTooltipDate}>{formatUtcDate(syncHover.ms)}</div>
            <div style={s.syncTooltipLine}>
              <strong>{data.fed_balance_sheet.label}:</strong> {formatBalanceSheet(bsAtHover.value)}{' '}
              <span style={s.syncAsOf}>(as of {bsAtHover.date})</span>
            </div>
            <div style={s.syncTooltipLine}>
              <strong>{data.fed_rate.label}:</strong> {formatRate(rateAtHover.value)}{' '}
              <span style={s.syncAsOf}>(as of {rateAtHover.date})</span>
            </div>
          </div>
        )}

        <div style={s.stack}>
          <LeverBlock
            lever={data.fed_balance_sheet}
            points={syncTimeline ? syncedView.bsPoints : asyncBsView.points}
            fromDate={syncTimeline ? syncedView.fromDate : asyncBsView.fromDate}
            toDate={syncTimeline ? syncedView.toDate : asyncBsView.toDate}
            showToolbar={!syncTimeline}
            range={syncTimeline ? rangeShared : rangeBs}
            onRangeChange={syncTimeline ? setRangeShared : setRangeBs}
            accent="#274b8a"
            latestFormatter={formatBalanceSheet}
            yTickFormatter={formatBalanceSheet}
            tooltipFormatter={formatBalanceSheet}
            xDomainMinMs={syncTimeline ? syncedView.xDomainMinMs : undefined}
            xDomainMaxMs={syncTimeline ? syncedView.xDomainMaxMs : undefined}
            syncHoverTimeMs={syncTimeline ? syncHover?.ms ?? null : undefined}
            onSyncHoverChange={syncTimeline ? handleSyncHoverChange : undefined}
            suppressChartTooltip={syncTimeline}
          />
          <LeverBlock
            lever={data.fed_rate}
            points={syncTimeline ? syncedView.rtPoints : asyncRateView.points}
            fromDate={syncTimeline ? syncedView.fromDate : asyncRateView.fromDate}
            toDate={syncTimeline ? syncedView.toDate : asyncRateView.toDate}
            showToolbar={!syncTimeline}
            range={syncTimeline ? rangeShared : rangeRate}
            onRangeChange={syncTimeline ? setRangeShared : setRangeRate}
            accent="#274b8a"
            latestFormatter={formatRate}
            yTickFormatter={formatRate}
            tooltipFormatter={formatRate}
            xDomainMinMs={syncTimeline ? syncedView.xDomainMinMs : undefined}
            xDomainMaxMs={syncTimeline ? syncedView.xDomainMaxMs : undefined}
            syncHoverTimeMs={syncTimeline ? syncHover?.ms ?? null : undefined}
            onSyncHoverChange={syncTimeline ? handleSyncHoverChange : undefined}
            suppressChartTooltip={syncTimeline}
          />
        </div>
      </div>
    </Card>
  )
}

const s: Record<string, React.CSSProperties> = {
  copy: {
    fontSize: '12px',
    color: 'var(--text-muted)',
    lineHeight: 1.5,
  },
  syncRow: {
    display: 'flex',
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: '10px',
    marginTop: '10px',
    marginBottom: '8px',
  },
  syncBtn: {
    border: '1px solid var(--border)',
    background: 'var(--bg-card)',
    color: 'var(--text-muted)',
    borderRadius: '999px',
    padding: '5px 12px',
    fontSize: '11px',
    fontWeight: 700,
    letterSpacing: '0.04em',
    cursor: 'pointer',
  },
  syncBtnOn: {
    color: 'var(--text)',
    background: 'var(--bg-card-raised)',
    borderColor: 'var(--blue-dim)',
  },
  syncHint: {
    fontSize: '11px',
    color: 'var(--text-dim)',
    flex: '1 1 200px',
  },
  stackWrap: {
    position: 'relative',
    marginTop: '10px',
  },
  stack: {
    display: 'flex',
    flexDirection: 'column',
    gap: '14px',
  },
  syncTooltip: {
    position: 'absolute',
    top: '-4px',
    transform: 'translate(-50%, -100%)',
    zIndex: 5,
    minWidth: '220px',
    maxWidth: 'min(92vw, 360px)',
    padding: '8px 10px',
    borderRadius: '8px',
    border: '1px solid var(--border)',
    background: 'var(--bg-card)',
    boxShadow: '0 8px 24px rgba(23, 37, 84, 0.12)',
    pointerEvents: 'none',
  },
  syncTooltipDate: {
    fontSize: '11px',
    fontWeight: 700,
    color: 'var(--text)',
    marginBottom: '6px',
    borderBottom: '1px solid var(--border-subtle)',
    paddingBottom: '4px',
  },
  syncTooltipLine: {
    fontSize: '11px',
    color: 'var(--text-secondary)',
    lineHeight: 1.45,
    marginTop: '4px',
  },
  syncAsOf: {
    fontSize: '10px',
    color: 'var(--text-muted)',
  },
  leverToolbar: {
    display: 'flex',
    justifyContent: 'space-between',
    gap: '12px',
    alignItems: 'center',
    flexWrap: 'wrap',
    paddingBottom: '2px',
  },
  rangeRow: {
    display: 'flex',
    gap: '8px',
    alignItems: 'center',
  },
  zoomLabel: {
    fontSize: '12px',
    color: 'var(--text-secondary)',
    fontWeight: 600,
    marginRight: '4px',
  },
  dateBoxes: {
    display: 'flex',
    gap: '8px',
    alignItems: 'center',
    flexWrap: 'wrap',
  },
  dateBox: {
    display: 'flex',
    gap: '6px',
    alignItems: 'center',
    border: '1px solid var(--border)',
    borderRadius: '6px',
    padding: '4px 8px',
    fontSize: '11px',
    color: 'var(--text-secondary)',
    background: 'var(--bg)',
  },
  dateLabel: {
    color: 'var(--text-muted)',
    textTransform: 'uppercase',
    letterSpacing: '0.04em',
    fontSize: '10px',
  },
  rangeBtn: {
    border: '1px solid var(--border)',
    background: 'var(--bg-card)',
    color: 'var(--text-muted)',
    borderRadius: '999px',
    padding: '4px 10px',
    fontSize: '11px',
    fontWeight: 700,
    letterSpacing: '0.04em',
    cursor: 'pointer',
  },
  rangeBtnActive: {
    color: 'var(--text)',
    background: 'var(--bg-card-raised)',
    borderColor: 'var(--blue-dim)',
  },
  leverBlock: {
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-md)',
    padding: '12px',
    display: 'flex',
    flexDirection: 'column',
    gap: '10px',
  },
  leverHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    gap: '16px',
    alignItems: 'flex-start',
  },
  leverTitle: {
    fontSize: '13px',
    fontWeight: 700,
    color: 'var(--text)',
  },
  leverMeta: {
    fontSize: '11px',
    color: 'var(--text-dim)',
    marginTop: '3px',
  },
  seriesId: {
    fontFamily: 'monospace',
    color: 'var(--text-muted)',
  },
  rightMeta: {
    fontSize: '11px',
    color: 'var(--text-dim)',
    lineHeight: 1.4,
    textAlign: 'right',
  },
}
