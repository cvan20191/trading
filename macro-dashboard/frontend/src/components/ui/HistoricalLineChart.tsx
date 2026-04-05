import { useMemo, useState } from 'react'

export type HistoryPoint = { date: string; value: number }

interface HistoricalLineChartProps {
  seriesLabel: string
  points: HistoryPoint[]
  stroke: string
  yTickFormatter: (value: number) => string
  tooltipValueFormatter: (value: number) => string
  /** When set, x-axis uses this domain so stacked charts can share the same calendar window. */
  xDomainMinMs?: number
  xDomainMaxMs?: number
  /** Controlled hover time (ms) for synced charts; pair with onSyncHoverChange. */
  syncHoverTimeMs?: number | null
  /** Reports pointer-derived time + horizontal position (0–100, chart width) for shared tooltip. */
  onSyncHoverChange?: (timeMs: number | null, chartXPct: number) => void
  /** Hide local tooltip when parent shows a combined sync tooltip. */
  suppressLocalTooltip?: boolean
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value))
}

/** Parse YYYY-MM-DD as UTC midnight for stable ordering. */
export function parseDateMs(dateStr: string): number {
  const [y, m, d] = dateStr.split('-').map(Number)
  return Date.UTC(y, (m ?? 1) - 1, d ?? 1)
}

function sortedByDate(points: HistoryPoint[]): HistoryPoint[] {
  return [...points].sort((a, b) => parseDateMs(a.date) - parseDateMs(b.date))
}

/** Nearest index by time (binary search on sorted times). */
function nearestIndexByTime(sortedTimes: number[], targetMs: number): number {
  if (sortedTimes.length === 0) return 0
  if (sortedTimes.length === 1) return 0
  let lo = 0
  let hi = sortedTimes.length - 1
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2)
    if (sortedTimes[mid] < targetMs) lo = mid + 1
    else hi = mid
  }
  const i = lo
  if (i <= 0) return 0
  if (i >= sortedTimes.length) return sortedTimes.length - 1
  const prev = i - 1
  return Math.abs(sortedTimes[i] - targetMs) < Math.abs(sortedTimes[prev] - targetMs) ? i : prev
}

/** Nearest observation to a target time (for synced crosshair readouts). */
export function nearestHistoryPoint(points: HistoryPoint[], targetMs: number): HistoryPoint | null {
  if (points.length === 0) return null
  const sorted = sortedByDate(points)
  const times = sorted.map((p) => parseDateMs(p.date))
  const idx = nearestIndexByTime(times, targetMs)
  return sorted[idx] ?? null
}

export function HistoricalLineChart({
  seriesLabel,
  points,
  stroke,
  yTickFormatter,
  tooltipValueFormatter,
  xDomainMinMs,
  xDomainMaxMs,
  syncHoverTimeMs,
  onSyncHoverChange,
  suppressLocalTooltip,
}: HistoricalLineChartProps) {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null)

  const width = 760
  const height = 260
  const padTop = 18
  const padBottom = 34
  const padLeft = 14
  const padRight = 78
  const plotLeft = padLeft
  const plotRight = width - padRight
  const plotTop = padTop
  const plotBottom = height - padBottom
  const plotWidth = plotRight - plotLeft
  const plotHeight = plotBottom - plotTop

  const sortedPoints = useMemo(() => sortedByDate(points), [points])
  const times = useMemo(
    () => sortedPoints.map((p) => parseDateMs(p.date)),
    [sortedPoints],
  )

  const { minTime, maxTime, spanTime, degenerateTime } = useMemo(() => {
    const hasFixedDomain =
      xDomainMinMs !== undefined &&
      xDomainMaxMs !== undefined &&
      xDomainMaxMs > xDomainMinMs

    if (hasFixedDomain) {
      const minT = xDomainMinMs!
      const maxT = xDomainMaxMs!
      const span = maxT - minT
      return {
        minTime: minT,
        maxTime: maxT,
        spanTime: Math.max(span, 1),
        degenerateTime: span <= 0 || sortedPoints.length === 0,
      }
    }

    if (times.length === 0) {
      return { minTime: 0, maxTime: 1, spanTime: 1, degenerateTime: true }
    }
    const minT = times[0]
    const maxT = times[times.length - 1]
    const span = maxT - minT
    const degenerate = span <= 0 || sortedPoints.length <= 1
    return {
      minTime: minT,
      maxTime: maxT,
      spanTime: Math.max(span, 1),
      degenerateTime: degenerate,
    }
  }, [times, sortedPoints.length, xDomainMinMs, xDomainMaxMs])

  const values = useMemo(() => sortedPoints.map((p) => p.value), [sortedPoints])
  const minRaw = values.length > 0 ? Math.min(...values) : 0
  const maxRaw = values.length > 0 ? Math.max(...values) : 1
  const spanRaw = Math.max(maxRaw - minRaw, 1e-9)
  const pad = spanRaw * 0.08
  const minY = minRaw - pad
  const maxY = maxRaw + pad
  const spanY = Math.max(maxY - minY, 1e-9)

  const xForTimeMs = (t: number): number => {
    if (sortedPoints.length === 0) return plotLeft + plotWidth / 2
    if (degenerateTime) return plotLeft + plotWidth / 2
    const u = (t - minTime) / spanTime
    return plotLeft + clamp(u, 0, 1) * plotWidth
  }

  const yForValue = (value: number): number =>
    plotBottom - ((value - minY) / spanY) * plotHeight

  const linePoints = useMemo(() => {
    if (sortedPoints.length === 0) return ''
    return sortedPoints
      .map((p) => {
        const t = parseDateMs(p.date)
        return `${xForTimeMs(t)},${yForValue(p.value)}`
      })
      .join(' ')
  }, [sortedPoints, degenerateTime, minTime, spanTime, minY, spanY])

  const yTicks = useMemo(() => {
    const ticks: number[] = []
    const count = 5
    for (let i = 0; i < count; i += 1) {
      ticks.push(minY + (i / (count - 1)) * spanY)
    }
    return ticks
  }, [minY, spanY])

  /** X-axis ticks: evenly spaced in time; label year from tick instant. */
  const xTickMeta = useMemo(() => {
    if (sortedPoints.length === 0 || degenerateTime) return []
    const fractions = [0, 0.25, 0.5, 0.75, 1]
    return fractions.map((f) => {
      const t = minTime + f * (maxTime - minTime)
      const x = xForTimeMs(t)
      const y = new Date(t)
      const label = String(y.getUTCFullYear())
      return { x, label, key: `${f}-${label}` }
    })
  }, [minTime, maxTime, degenerateTime, sortedPoints.length])

  const syncMode = Boolean(onSyncHoverChange)
  const effectiveHoverTimeMs =
    syncMode && syncHoverTimeMs !== undefined && syncHoverTimeMs !== null
      ? syncHoverTimeMs
      : null

  const internalHoverPoint = !syncMode && hoverIndex !== null ? sortedPoints[hoverIndex] : null
  const syncHoverPoint =
    syncMode && effectiveHoverTimeMs !== null && sortedPoints.length > 0
      ? sortedPoints[nearestIndexByTime(times, effectiveHoverTimeMs)]
      : null

  const hoverPoint = syncMode ? syncHoverPoint : internalHoverPoint
  const hoverX =
    syncMode && effectiveHoverTimeMs !== null && sortedPoints.length > 0 && !degenerateTime
      ? xForTimeMs(effectiveHoverTimeMs)
      : hoverPoint
        ? xForTimeMs(parseDateMs(hoverPoint.date))
        : null
  const hoverY = hoverPoint ? yForValue(hoverPoint.value) : null

  const handlePointerMove = (event: React.MouseEvent<SVGSVGElement>) => {
    if (sortedPoints.length === 0) return
    const rect = event.currentTarget.getBoundingClientRect()
    const svgX = ((event.clientX - rect.left) / rect.width) * width
    const ratio = clamp((svgX - plotLeft) / plotWidth, 0, 1)
    const chartXPct = clamp(((event.clientX - rect.left) / rect.width) * 100, 0, 100)

    if (syncMode && onSyncHoverChange) {
      if (degenerateTime) {
        onSyncHoverChange(minTime, chartXPct)
        return
      }
      const tHover = minTime + ratio * (maxTime - minTime)
      onSyncHoverChange(tHover, chartXPct)
      return
    }

    if (degenerateTime) {
      const idx = nearestIndexByTime(times, minTime)
      setHoverIndex(idx)
      return
    }
    const tHover = minTime + ratio * (maxTime - minTime)
    const idx = nearestIndexByTime(times, tHover)
    setHoverIndex(idx)
  }

  const handlePointerLeave = () => {
    if (syncMode && onSyncHoverChange) {
      onSyncHoverChange(null, 0)
      return
    }
    setHoverIndex(null)
  }

  return (
    <div style={s.wrap}>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="xMidYMid meet"
        style={s.svg}
        onMouseMove={handlePointerMove}
        onMouseEnter={handlePointerMove}
        onMouseLeave={handlePointerLeave}
      >
        <rect x={0} y={0} width={width} height={height} fill="#ffffff" />

        {yTicks.map((tick) => (
          <line
            key={`grid-${tick}`}
            x1={plotLeft}
            x2={plotRight}
            y1={yForValue(tick)}
            y2={yForValue(tick)}
            stroke="#e8edf4"
            strokeWidth={1}
          />
        ))}

        <polyline
          fill="none"
          stroke={stroke}
          strokeWidth={2}
          points={linePoints}
          vectorEffect="non-scaling-stroke"
        />

        {xTickMeta.map(({ x, label, key }) => (
          <text
            key={key}
            x={x}
            y={height - 10}
            textAnchor="middle"
            fontSize={10}
            fill="#5b6573"
          >
            {label}
          </text>
        ))}

        {yTicks.map((tick) => (
          <text
            key={`y-${tick}`}
            x={width - 6}
            y={yForValue(tick) + 3}
            textAnchor="end"
            fontSize={10}
            fill="#5b6573"
          >
            {yTickFormatter(tick)}
          </text>
        ))}

        {hoverX !== null && hoverY !== null && hoverPoint && (
          <>
            <line
              x1={hoverX}
              x2={hoverX}
              y1={plotTop}
              y2={plotBottom}
              stroke="#64748b"
              strokeDasharray="4 3"
              strokeWidth={1.25}
              vectorEffect="non-scaling-stroke"
            />
            <circle cx={hoverX} cy={hoverY} r={7} fill={`${stroke}33`} />
            <circle cx={hoverX} cy={hoverY} r={4} fill={stroke} />
          </>
        )}
      </svg>

      {hoverPoint && hoverX !== null && hoverY !== null && !suppressLocalTooltip && (
        <div
          style={{
            ...s.tooltip,
            left: `${clamp((hoverX / width) * 100, 10, 90)}%`,
            top: `${clamp((hoverY / height) * 100 - 18, 8, 82)}%`,
          }}
        >
          <div style={s.tooltipTitle}>
            <span style={{ ...s.bullet, background: stroke }} />
            {seriesLabel}
          </div>
          <div style={s.tooltipValue}>{tooltipValueFormatter(hoverPoint.value)}</div>
          <div style={s.tooltipDate}>{hoverPoint.date}</div>
        </div>
      )}
    </div>
  )
}

const s: Record<string, React.CSSProperties> = {
  wrap: {
    position: 'relative',
    border: '1px solid #d8e0eb',
    borderRadius: '8px',
    overflow: 'hidden',
    background: '#ffffff',
    width: '100%',
    aspectRatio: '760 / 260',
  },
  svg: {
    width: '100%',
    height: '100%',
    display: 'block',
    background: '#ffffff',
  },
  tooltip: {
    position: 'absolute',
    transform: 'translate(-50%, -100%)',
    background: '#ffffff',
    border: '1px solid #d0d9e6',
    borderRadius: '8px',
    boxShadow: '0 6px 20px rgba(23, 37, 84, 0.15)',
    padding: '8px 10px',
    minWidth: '150px',
    pointerEvents: 'none',
  },
  tooltipTitle: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    fontSize: '11px',
    color: '#4f5b6a',
  },
  tooltipValue: {
    marginTop: '2px',
    fontSize: '12px',
    fontWeight: 700,
    color: '#1f2937',
  },
  tooltipDate: {
    marginTop: '2px',
    fontSize: '11px',
    color: '#6b7280',
  },
  bullet: {
    width: '8px',
    height: '8px',
    borderRadius: '999px',
    display: 'inline-block',
  },
}
