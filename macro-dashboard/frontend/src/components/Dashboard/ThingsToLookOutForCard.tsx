import type { CatalystState, MegaIPOItem, FedChairCandidate } from '../../types/catalysts'
import { Card } from '../ui/Card'
import { StatusPill } from '../ui/StatusPill'
import { fmtNum } from '../../lib/fmt'
import type { ColorKey } from '../../lib/colors'

interface Props {
  catalysts: CatalystState
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function ipoStatusColor(status: string): ColorKey {
  if (['delayed', 'weak_demand', 'missed_target'].includes(status)) return 'red'
  if (['strong_demand', 'completed', 'on_track'].includes(status)) return 'green'
  if (status === 'rumored') return 'muted'
  return 'muted'
}

function ipoStatusLabel(status: string): string {
  const map: Record<string, string> = {
    unknown: 'Unknown',
    rumored: 'Rumored',
    on_track: 'On Track',
    delayed: 'Delayed',
    weak_demand: 'Weak Demand',
    strong_demand: 'Strong Demand',
    missed_target: 'Missed Target',
    completed: 'Completed',
  }
  return map[status] ?? status
}

function signalColor(signal: string): ColorKey {
  if (signal === 'Weakening') return 'red'
  if (signal === 'Supportive') return 'green'
  return 'muted'
}

function biasColor(bias: string): ColorKey {
  if (bias.includes('Fast Cuts') || bias.includes('Bullish')) return 'green'
  if (bias.includes('Hawkish') || bias.includes('Inflation Wary')) return 'yellow'
  return 'muted'
}

function conditionDot({ met, label }: { met: boolean; label: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
      <span style={{
        width: '7px', height: '7px', borderRadius: '50%', flexShrink: 0,
        background: met ? 'var(--green)' : 'var(--red)',
      }} />
      <span style={{ fontSize: '11px', color: met ? 'var(--green)' : 'var(--text-muted)' }}>
        {label}
      </span>
    </div>
  )
}

function plumbingStatusColor(status: string): ColorKey {
  if (status === 'stress') return 'red'
  if (status === 'watch') return 'yellow'
  if (status === 'normal') return 'green'
  return 'muted'
}

function stressLabelColor(label: string): ColorKey {
  if (label === 'Stress') return 'red'
  if (label === 'Watch') return 'yellow'
  if (label === 'Normal') return 'green'
  return 'muted'
}

// ---------------------------------------------------------------------------
// Provenance badge
// ---------------------------------------------------------------------------

function ProvenanceBadge({ provenance }: { provenance: string }) {
  const label =
    provenance === 'live-linked' ? 'Live-linked'
    : provenance === 'mixed' ? 'Mixed'
    : 'Config-backed'
  const color: React.CSSProperties['color'] =
    provenance === 'live-linked' ? 'var(--green)'
    : provenance === 'mixed' ? 'var(--yellow)'
    : 'var(--text-muted)'
  return (
    <span style={{
      fontSize: '9px', fontWeight: 600, letterSpacing: '0.05em',
      textTransform: 'uppercase', color, opacity: 0.8,
      padding: '1px 5px', borderRadius: '3px',
      border: `1px solid ${color}`,
    }}>
      {label}
    </span>
  )
}

// ---------------------------------------------------------------------------
// Section wrapper
// ---------------------------------------------------------------------------

function Section({
  title, pill, pillColor, provenance, whyItMatters, playbookImpact, children,
}: {
  title: string
  pill?: string
  pillColor?: ColorKey
  provenance?: string
  whyItMatters?: string
  playbookImpact?: string
  children?: React.ReactNode
}) {
  return (
    <div style={s.section}>
      <div style={s.sectionHeader}>
        <span style={s.sectionTitle}>{title}</span>
        {pill && pillColor && <StatusPill label={pill} colorKey={pillColor} size="sm" />}
        {provenance && <ProvenanceBadge provenance={provenance} />}
      </div>
      {children}
      {whyItMatters && (
        <p style={s.whyText}>{whyItMatters}</p>
      )}
      {playbookImpact && (
        <div style={s.impactBox}>
          <span style={s.impactIcon}>→</span>
          <span style={s.impactText}>{playbookImpact}</span>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function ThingsToLookOutForCard({ catalysts }: Props) {
  const { mega_ipos, fed_chair, tariff_twin_deficit, clean_cut_watch, plumbing_watch, next_lookouts } = catalysts

  return (
    <Card title="Forward Catalysts">
      {/* Card subtitle — secondary-overlay reminder */}
      <p style={s.cardSubtitle}>
        Secondary forward-watch overlays — do not override core macro regime
      </p>

      {/* ---- 1. Mega-IPOs ---- */}
      <Section
        title="Mega-IPOs — Liquidity Litmus Test"
        pill={mega_ipos.overall_signal}
        pillColor={signalColor(mega_ipos.overall_signal)}
        provenance={mega_ipos.provenance}
        whyItMatters={mega_ipos.why_it_matters}
        playbookImpact={mega_ipos.playbook_impact}
      >
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginBottom: '4px' }}>
          {mega_ipos.items.length === 0
            ? <span style={s.naText}>No IPO items configured</span>
            : mega_ipos.items.map((item: MegaIPOItem) => (
              <div key={item.name} style={s.ipoChip}>
                <span style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-primary)' }}>
                  {item.name}
                </span>
                {item.target_valuation && (
                  <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
                    {item.target_valuation}
                  </span>
                )}
                <StatusPill label={ipoStatusLabel(item.status)} colorKey={ipoStatusColor(item.status)} size="sm" />
              </div>
            ))
          }
        </div>
      </Section>

      <div style={s.divider} />

      {/* ---- 2. Fed Chair ---- */}
      <Section
        title="New Fed Chair"
        pill={fed_chair.current_bias}
        pillColor={biasColor(fed_chair.current_bias)}
        provenance={fed_chair.provenance}
        whyItMatters={fed_chair.why_it_matters}
        playbookImpact={fed_chair.playbook_impact}
      >
        {fed_chair.candidates.length > 0 && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', marginBottom: '4px' }}>
            {fed_chair.candidates.map((c: FedChairCandidate) => (
              <div key={c.name} style={s.candidateRow}>
                <span style={{ fontSize: '11px', color: 'var(--text-primary)', fontWeight: c.status === 'front_runner' ? 700 : 400 }}>
                  {c.name}
                </span>
                <StatusPill
                  label={c.status.replace(/_/g, ' ')}
                  colorKey={c.status === 'front_runner' ? 'blue' : 'muted'}
                  size="sm"
                />
                <span style={{ fontSize: '10px', color: 'var(--text-muted)', marginLeft: 'auto' }}>
                  {c.tone.replace(/_/g, ' ')}
                </span>
              </div>
            ))}
          </div>
        )}
      </Section>

      <div style={s.divider} />

      {/* ---- 3. Tariffs / Twin Deficits / Dollar ---- */}
      <Section
        title="Tariffs / Twin Deficits / Dollar"
        pill={tariff_twin_deficit.dxy_pressure || tariff_twin_deficit.tariff_pressure_active ? 'Pressure Active' : 'Watching'}
        pillColor={tariff_twin_deficit.dxy_pressure || tariff_twin_deficit.tariff_pressure_active ? 'yellow' : 'muted'}
        provenance={tariff_twin_deficit.provenance}
        whyItMatters={tariff_twin_deficit.why_it_matters}
        playbookImpact={tariff_twin_deficit.playbook_impact}
      >
        <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', marginBottom: '4px' }}>
          <div style={s.metricMini}>
            <span style={s.metricMiniLabel}>DXY</span>
            <span style={{
              fontSize: '14px', fontWeight: 700,
              color: tariff_twin_deficit.dxy_pressure ? 'var(--yellow)' : 'var(--text-primary)',
            }}>
              {tariff_twin_deficit.dxy !== null ? fmtNum(tariff_twin_deficit.dxy, 1) : 'N/A'}
            </span>
            {tariff_twin_deficit.dxy_pressure && (
              <span style={{ fontSize: '9px', color: 'var(--yellow)' }}>PRESSURE</span>
            )}
          </div>
          <div style={s.metricMini}>
            <span style={s.metricMiniLabel}>Tariffs</span>
            <StatusPill
              label={tariff_twin_deficit.tariff_status.replace(/_/g, ' ')}
              colorKey={tariff_twin_deficit.tariff_pressure_active ? 'yellow' : 'muted'}
              size="sm"
            />
          </div>
        </div>
      </Section>

      <div style={s.divider} />

      {/* ---- 4. Labor vs Inflation / Clean Cut ---- */}
      <Section
        title="Labor vs Inflation — Clean Cut Window"
        pill={clean_cut_watch.clean_cut_window_open ? 'Window Open' : 'Window Closed'}
        pillColor={clean_cut_watch.clean_cut_window_open ? 'green' : 'red'}
        provenance={clean_cut_watch.provenance}
        whyItMatters={clean_cut_watch.why_it_matters}
        playbookImpact={clean_cut_watch.playbook_impact}
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: '5px', marginBottom: '4px' }}>
          {conditionDot({ met: clean_cut_watch.unemployment_condition_met, label: 'Unemployment ≥ 5% (labor slack sufficient)' })}
          {conditionDot({ met: clean_cut_watch.inflation_condition_met, label: 'Core CPI < 2.5% (inflation credibly beaten)' })}
        </div>
        {!clean_cut_watch.clean_cut_window_open && clean_cut_watch.why_not_open && (
          <p style={{ margin: 0, fontSize: '11px', color: 'var(--text-muted)', fontStyle: 'italic', marginBottom: '2px' }}>
            Missing: {clean_cut_watch.why_not_open}
          </p>
        )}
      </Section>

      <div style={s.divider} />

      {/* ---- 5. Plumbing Watch ---- */}
      <Section
        title="Repo / Reverse Repo / Reserve Plumbing"
        pill={plumbing_watch.stress_label}
        pillColor={stressLabelColor(plumbing_watch.stress_label)}
        provenance={plumbing_watch.provenance}
        whyItMatters={plumbing_watch.why_it_matters}
        playbookImpact={plumbing_watch.playbook_impact}
      >
        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', marginBottom: '4px' }}>
          {[
            { label: 'Repo', status: plumbing_watch.repo_status },
            { label: 'Rev. Repo', status: plumbing_watch.reverse_repo_status },
            { label: 'Reserves', status: plumbing_watch.reserve_status },
          ].map(({ label, status }) => (
            <div key={label} style={s.plumbingChip}>
              <span style={s.plumbingLabel}>{label}</span>
              <StatusPill label={status} colorKey={plumbingStatusColor(status)} size="sm" />
            </div>
          ))}
        </div>
      </Section>

      <div style={s.divider} />

      {/* ---- Next Lookouts ---- */}
      <div>
        <div style={{
          fontSize: '11px', fontWeight: 700, letterSpacing: '0.08em',
          textTransform: 'uppercase', color: 'var(--blue)', marginBottom: '10px',
        }}>
          Next Lookouts
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {next_lookouts.slice(0, 3).map((lookout, i) => (
            <div key={i} style={s.lookoutRow}>
              <span style={s.lookoutNum}>{i + 1}</span>
              <span style={s.lookoutText}>{lookout}</span>
            </div>
          ))}
        </div>
      </div>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const s: Record<string, React.CSSProperties> = {
  cardSubtitle: {
    margin: '0 0 12px 0', fontSize: '11px',
    color: 'var(--text-muted)', fontStyle: 'italic', lineHeight: 1.4,
    borderLeft: '2px solid var(--border-subtle)', paddingLeft: '8px',
  },
  section: { display: 'flex', flexDirection: 'column', gap: '6px' },
  sectionHeader: {
    display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap',
  },
  sectionTitle: {
    fontSize: '12px', fontWeight: 700, color: 'var(--text-secondary)',
    letterSpacing: '0.03em',
  },
  whyText: {
    margin: 0, fontSize: '11px', color: 'var(--text-muted)', lineHeight: 1.55,
  },
  impactBox: {
    display: 'flex', gap: '6px', alignItems: 'flex-start',
    padding: '7px 10px',
    background: 'var(--bg-card-raised)',
    borderRadius: 'var(--radius-sm)',
    border: '1px solid var(--border-subtle)',
  },
  impactIcon: { color: 'var(--blue)', fontSize: '11px', flexShrink: 0 },
  impactText: { fontSize: '11px', color: 'var(--text-secondary)', lineHeight: 1.55 },
  divider: { height: '1px', background: 'var(--border-subtle)' },
  ipoChip: {
    display: 'flex', flexDirection: 'column', gap: '3px',
    padding: '6px 10px',
    background: 'var(--bg-card-raised)',
    borderRadius: 'var(--radius-sm)',
    border: '1px solid var(--border-subtle)',
    minWidth: '90px',
  },
  candidateRow: {
    display: 'flex', alignItems: 'center', gap: '8px',
    padding: '5px 8px',
    background: 'var(--bg-card-raised)',
    borderRadius: '4px',
    border: '1px solid var(--border-subtle)',
  },
  metricMini: { display: 'flex', flexDirection: 'column', gap: '2px', alignItems: 'center', minWidth: '60px' },
  metricMiniLabel: { fontSize: '9px', color: 'var(--text-muted)', fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase' },
  plumbingChip: { display: 'flex', flexDirection: 'column', gap: '3px', alignItems: 'center', minWidth: '70px' },
  plumbingLabel: { fontSize: '9px', color: 'var(--text-muted)', fontWeight: 700, letterSpacing: '0.04em', textTransform: 'uppercase' },
  naText: { fontSize: '12px', color: 'var(--text-muted)' },
  lookoutRow: { display: 'flex', gap: '10px', alignItems: 'flex-start' },
  lookoutNum: {
    flexShrink: 0, width: '18px', height: '18px', borderRadius: '50%',
    background: 'var(--blue-bg)', border: '1px solid var(--blue-dim)',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    fontSize: '10px', fontWeight: 700, color: 'var(--blue)',
    marginTop: '1px',
  },
  lookoutText: {
    fontSize: '12px', color: 'var(--text-secondary)', lineHeight: 1.6,
  },
}
