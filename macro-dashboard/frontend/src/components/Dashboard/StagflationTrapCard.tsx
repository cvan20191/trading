import { useEffect, useState } from 'react'
import type { StagflationTrap } from '../../types/summary'
import { Card } from '../ui/Card'
import { fmtNum, fmtPct, na } from '../../lib/fmt'

interface Props {
  trap: StagflationTrap
  pmiMfgOverride: number | null
  pmiSvcOverride: number | null
  onPmiOverrideChange: (mfg: number | null, svc: number | null) => void
}

const PMI_MIN = 30
const PMI_MAX = 70

function parsePmiInput(value: string): number | null {
  const trimmed = value.trim()
  if (!trimmed) return null
  const parsed = Number(trimmed)
  if (!Number.isFinite(parsed)) return null
  if (parsed < PMI_MIN || parsed > PMI_MAX) return null
  return parsed
}

function MetricRow({ label, value, warn }: { label: string; value: string; warn?: boolean }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingBottom: '8px', borderBottom: '1px solid var(--border-subtle)' }}>
      <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{label}</span>
      <span style={{ fontSize: '12px', fontWeight: 600, color: warn ? 'var(--yellow)' : 'var(--text-primary)' }}>{value}</span>
    </div>
  )
}

function StatusChip({ label, active, color }: { label: string; active?: boolean; color: string }) {
  return (
    <span style={{
      fontSize: '10px', fontWeight: 700, padding: '2px 8px', borderRadius: '3px',
      background: active ? `${color}22` : 'var(--bg-card-raised)',
      color: active ? color : 'var(--text-muted)',
      border: `1px solid ${active ? color : 'var(--border-subtle)'}`,
    }}>
      {label}
    </span>
  )
}

export function StagflationTrapCard({
  trap,
  pmiMfgOverride,
  pmiSvcOverride,
  onPmiOverrideChange,
}: Props) {
  const isActive = trap.active
  const [mfgDraft, setMfgDraft] = useState(pmiMfgOverride == null ? '' : String(pmiMfgOverride))
  const [svcDraft, setSvcDraft] = useState(pmiSvcOverride == null ? '' : String(pmiSvcOverride))

  useEffect(() => {
    setMfgDraft(pmiMfgOverride == null ? '' : String(pmiMfgOverride))
  }, [pmiMfgOverride])

  useEffect(() => {
    setSvcDraft(pmiSvcOverride == null ? '' : String(pmiSvcOverride))
  }, [pmiSvcOverride])

  const applyOverrides = () => {
    onPmiOverrideChange(parsePmiInput(mfgDraft), parsePmiInput(svcDraft))
  }

  const clearOverrides = () => {
    setMfgDraft('')
    setSvcDraft('')
    onPmiOverrideChange(null, null)
  }

  return (
    <Card title="Stagflation Trap Monitor">
      {/* Trap banner */}
      {isActive && (
        <div style={{
          padding: '10px 14px',
          background: 'var(--red-bg)',
          border: '1px solid var(--red)',
          borderRadius: 'var(--radius-sm)',
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
        }}>
          <span style={{ fontSize: '14px' }}>⚠</span>
          <div>
            <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--red)', letterSpacing: '0.05em' }}>
              STAGFLATION TRAP ACTIVE
            </div>
            <div style={{ fontSize: '11px', color: 'var(--red)', opacity: 0.85, marginTop: '2px' }}>
              Growth is weakening, but inflation is still sticky enough to constrain rate cuts.
            </div>
          </div>
        </div>
      )}
      {!isActive && (
        <div style={{ padding: '6px 10px', background: 'var(--green-bg)', border: '1px solid var(--green-dim)', borderRadius: 'var(--radius-sm)', fontSize: '12px', color: 'var(--green)' }}>
          Stagflation trap not currently active
        </div>
      )}

      {/* Status chips */}
      <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
        <StatusChip label="Growth Weakening" active={trap.growth_weakening} color="var(--yellow)" />
        <StatusChip label="Sticky Inflation" active={trap.sticky_inflation} color="var(--red)" />
        <StatusChip label="Oil Risk" active={trap.oil_risk_active ?? false} color="var(--red)" />
      </div>

      {/* Split columns: Growth vs Inflation */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>

        {/* Growth side */}
        <div style={s.side}>
          <div style={s.sideHeader}>
            <span style={s.sideTitle}>Growth</span>
            <span style={{ ...s.sideBadge, color: trap.growth_weakening ? 'var(--yellow)' : 'var(--green)' }}>
              {trap.growth_weakening ? 'Weakening' : 'Holding'}
            </span>
          </div>
          <MetricRow label="PMI Manufacturing" value={na(trap.pmi_manufacturing)} warn={(trap.pmi_manufacturing ?? 51) < 50} />
          <MetricRow label="PMI Services" value={na(trap.pmi_services)} warn={(trap.pmi_services ?? 51) < 50} />
          <div style={s.overrideWrap}>
            <div style={s.overrideTitle}>Manual monthly PMI override</div>
            <div style={s.overrideRow}>
              <input
                type="number"
                step="0.1"
                min={PMI_MIN}
                max={PMI_MAX}
                placeholder="PMI Mfg"
                value={mfgDraft}
                onChange={(e) => setMfgDraft(e.target.value)}
                onBlur={applyOverrides}
                style={s.overrideInput}
              />
              <input
                type="number"
                step="0.1"
                min={PMI_MIN}
                max={PMI_MAX}
                placeholder="PMI Svc"
                value={svcDraft}
                onChange={(e) => setSvcDraft(e.target.value)}
                onBlur={applyOverrides}
                style={s.overrideInput}
              />
              <button type="button" onClick={applyOverrides} style={s.overrideBtn}>Apply</button>
              <button type="button" onClick={clearOverrides} style={s.clearBtn}>Clear</button>
            </div>
            <div style={s.helperNote}>Stored in this browser only. Valid range: 30-70.</div>
          </div>
          <MetricRow label="Unemployment" value={fmtPct(trap.unemployment_rate)} warn={(trap.unemployment_rate ?? 0) > 4.0} />
        </div>

        {/* Inflation side */}
        <div style={s.side}>
          <div style={s.sideHeader}>
            <span style={s.sideTitle}>Inflation</span>
            <span style={{ ...s.sideBadge, color: trap.sticky_inflation ? 'var(--red)' : 'var(--green)' }}>
              {trap.sticky_inflation ? 'Sticky' : 'Not Sticky'}
            </span>
          </div>
          <MetricRow label="Core CPI YoY" value={fmtPct(trap.core_cpi_yoy)} warn={(trap.core_cpi_yoy ?? 0) > 3.0} />
          <MetricRow label="Shelter" value={trap.shelter_status ?? 'N/A'} warn={trap.shelter_status === 'sticky'} />
          <MetricRow label="Services ex Energy" value={trap.services_ex_energy_status ?? 'N/A'} warn={trap.services_ex_energy_status === 'sticky'} />
          <MetricRow label="WTI Oil" value={`$${fmtNum(trap.wti_oil, 1)}`} warn={(trap.wti_oil ?? 0) >= 100} />
        </div>
      </div>
    </Card>
  )
}

const s: Record<string, React.CSSProperties> = {
  side: {
    display: 'flex', flexDirection: 'column', gap: '6px',
    padding: '10px 12px', background: 'var(--bg-card-raised)',
    borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)',
  },
  sideHeader: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' },
  sideTitle: { fontSize: '11px', fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--text-muted)' },
  sideBadge: { fontSize: '10px', fontWeight: 700 },
  overrideWrap: {
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
    marginTop: '2px',
    marginBottom: '2px',
  },
  overrideTitle: {
    fontSize: '10px',
    color: 'var(--text-muted)',
    fontWeight: 600,
  },
  overrideRow: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr auto auto',
    gap: '6px',
    alignItems: 'center',
  },
  overrideInput: {
    fontSize: '11px',
    padding: '4px 6px',
    border: '1px solid var(--border-subtle)',
    borderRadius: '4px',
    background: 'var(--bg-card)',
    color: 'var(--text-primary)',
  },
  overrideBtn: {
    fontSize: '11px',
    padding: '4px 8px',
    borderRadius: '4px',
    border: '1px solid var(--blue-dim)',
    background: 'transparent',
    color: 'var(--blue)',
    cursor: 'pointer',
  },
  clearBtn: {
    fontSize: '11px',
    padding: '4px 8px',
    borderRadius: '4px',
    border: '1px solid var(--border-subtle)',
    background: 'transparent',
    color: 'var(--text-muted)',
    cursor: 'pointer',
  },
  helperNote: {
    fontSize: '10px',
    color: 'var(--text-muted)',
    lineHeight: 1.4,
    marginTop: '2px',
    marginBottom: '2px',
  },
}
