// Horizontal segmented gauge bar

interface Segment {
  label: string
  min: number
  max: number        // use Infinity for open-ended last segment
  color: string      // CSS color or var()
}

interface Props {
  segments: Segment[]
  value: number | null | undefined
}

const TOTAL_RANGE = (segments: Segment[]) => {
  const last = segments[segments.length - 1]
  return last.max === Infinity ? last.min * 2 : last.max
}

export function GaugeBar({ segments, value }: Props) {
  const total = TOTAL_RANGE(segments)

  return (
    <div>
      {/* Segmented track */}
      <div style={{ display: 'flex', height: '10px', borderRadius: '5px', overflow: 'hidden', gap: '2px' }}>
        {segments.map((seg, i) => {
          const segMax = seg.max === Infinity ? total : seg.max
          const width = ((segMax - seg.min) / total) * 100
          const isActive = value !== null && value !== undefined && value >= seg.min &&
            (seg.max === Infinity ? true : value < seg.max)
          return (
            <div key={i} style={{
              flex: `0 0 ${width}%`,
              background: isActive ? seg.color : `${seg.color}33`,
              borderRadius: i === 0 ? '5px 0 0 5px' : i === segments.length - 1 ? '0 5px 5px 0' : '0',
              transition: 'background 0.3s',
            }} />
          )
        })}
      </div>

      {/* Zone labels */}
      <div style={{ display: 'flex', marginTop: '5px' }}>
        {segments.map((seg, i) => {
          const segMax = seg.max === Infinity ? total : seg.max
          const width = ((segMax - seg.min) / total) * 100
          const isActive = value !== null && value !== undefined && value >= seg.min &&
            (seg.max === Infinity ? true : value < seg.max)
          return (
            <div key={i} style={{
              flex: `0 0 ${width}%`,
              fontSize: '10px',
              color: isActive ? seg.color : 'var(--text-muted)',
              fontWeight: isActive ? 700 : 400,
              textAlign: 'center',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
              paddingLeft: '2px',
              paddingRight: '2px',
            }}>
              {seg.label}
            </div>
          )
        })}
      </div>
    </div>
  )
}
