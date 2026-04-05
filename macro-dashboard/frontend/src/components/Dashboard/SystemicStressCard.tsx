import type { SystemicStress } from '../../types/summary'
import { Card } from '../ui/Card'
import { fmtNum } from '../../lib/fmt'

interface Props { stress: SystemicStress }

interface StressGaugeProps {
  label: string
  value: number | null | undefined
  displayValue: string
  zone: string
  zoneColor: string
  segments: { label: string; color: string; active: boolean }[]
  note?: string
}

function StressGauge({ label, value: _value, displayValue, zone, zoneColor, segments, note }: StressGaugeProps) {
  return (
    <div style={s.gauge}>
      <div style={s.gaugeTop}>
        <span style={s.gaugeLabel}>{label}</span>
        <span style={{ fontSize: '16px', fontWeight: 800, color: zoneColor }}>{displayValue}</span>
        <span style={{ fontSize: '10px', fontWeight: 700, color: zoneColor, letterSpacing: '0.04em' }}>{zone}</span>
      </div>
      {/* Segment bar */}
      <div style={{ display: 'flex', gap: '2px', height: '6px', borderRadius: '3px', overflow: 'hidden' }}>
        {segments.map((seg, i) => (
          <div key={i} style={{
            flex: 1,
            background: seg.active ? seg.color : `${seg.color}30`,
            borderRadius: i === 0 ? '3px 0 0 3px' : i === segments.length - 1 ? '0 3px 3px 0' : '0',
          }} />
        ))}
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
        {segments.map((seg, i) => (
          <span key={i} style={{ fontSize: '9px', color: seg.active ? seg.color : 'var(--text-muted)', fontWeight: seg.active ? 700 : 400 }}>
            {seg.label}
          </span>
        ))}
      </div>
      {note && (
        <p style={{ margin: 0, fontSize: '11px', color: 'var(--text-muted)', lineHeight: 1.5, marginTop: '2px' }}>{note}</p>
      )}
    </div>
  )
}

export function SystemicStressCard({ stress }: Props) {
  // Yield Curve
  const yc = stress.yield_curve_value ?? null
  const ycZone = stress.yield_curve_inverted ? 'Inverted' : 'Normal'
  const ycColor = stress.yield_curve_inverted ? 'var(--yellow)' : 'var(--green)'
  const ycSegments = [
    { label: 'Normal', color: 'var(--green)', active: !stress.yield_curve_inverted },
    { label: 'Inverted', color: 'var(--yellow)', active: !!stress.yield_curve_inverted },
  ]

  // NPL
  const npl = stress.npl_ratio ?? null
  const nplZone = stress.npl_zone ?? 'Normal'
  const nplColor = nplZone === 'Normal' ? 'var(--green)' : nplZone === 'Caution' ? 'var(--yellow)' : 'var(--red)'
  const nplSegments = [
    { label: 'Normal (<1%)', color: 'var(--green)', active: nplZone === 'Normal' },
    { label: 'Caution', color: 'var(--yellow)', active: nplZone === 'Caution' },
    { label: 'Warning (≥1.5%)', color: 'var(--red)', active: nplZone === 'Warning' },
  ]

  // Market Cap / M2
  const mcm2 = stress.market_cap_m2_ratio ?? null
  const mcm2Zone = stress.market_cap_m2_zone ?? 'Normal'
  const mcm2Color = mcm2Zone === 'Normal' ? 'var(--green)' : mcm2Zone === 'Warning' ? 'var(--yellow)' : 'var(--red)'
  const mcm2Segments = [
    { label: 'Normal (<2)', color: 'var(--green)', active: mcm2Zone === 'Normal' },
    { label: 'Warning', color: 'var(--yellow)', active: mcm2Zone === 'Warning' },
    { label: 'Extreme (≥3)', color: 'var(--red)', active: mcm2Zone === 'Extreme' },
  ]

  return (
    <Card title="Systemic Stress Gauges">
      <StressGauge
        label="10Y–2Y Yield Curve"
        value={yc}
        displayValue={yc !== null ? `${fmtNum(yc, 2)}%` : 'N/A'}
        zone={ycZone}
        zoneColor={ycColor}
        segments={ycSegments}
        note="An inverted yield curve has historically preceded recessions — it is a warning, not a timer."
      />

      <div style={{ height: '1px', background: 'var(--border-subtle)' }} />

      <StressGauge
        label="Bank NPL Ratio (Delinquency Proxy)"
        value={npl}
        displayValue={npl !== null ? `${fmtNum(npl, 2)}%` : 'N/A'}
        zone={nplZone ?? 'N/A'}
        zoneColor={nplColor}
        segments={nplSegments}
        note="NPL above 1.5% signals rising credit stress. This can be temporary — context matters."
      />

      <div style={{ height: '1px', background: 'var(--border-subtle)' }} />

      <StressGauge
        label="Market Cap / M2"
        value={mcm2}
        displayValue={mcm2 !== null ? fmtNum(mcm2, 2) : 'N/A'}
        zone={mcm2Zone ?? 'N/A'}
        zoneColor={mcm2Color}
        segments={mcm2Segments}
        note="Ratio above 2.0 suggests markets may be pricing in more liquidity than exists."
      />

      <div style={{
        padding: '8px 12px',
        background: 'var(--bg-card-raised)',
        border: '1px solid var(--border-subtle)',
        borderRadius: 'var(--radius-sm)',
        fontSize: '11px',
        color: 'var(--text-muted)',
        fontStyle: 'italic',
      }}>
        These are structural warning lights, not exact timing tools. Conditions can persist longer than expected.
      </div>
    </Card>
  )
}

const s: Record<string, React.CSSProperties> = {
  gauge: { display: 'flex', flexDirection: 'column', gap: '5px' },
  gaugeTop: { display: 'flex', alignItems: 'center', gap: '10px' },
  gaugeLabel: { fontSize: '11px', color: 'var(--text-muted)', flex: 1 },
}
