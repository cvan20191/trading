import type { Valuation, ValuationConstituent } from '../../types/summary'
import { Card } from '../ui/Card'
import { fmtPE } from '../../lib/fmt'

interface Props { val: Valuation }

const SEGMENTS = [
  { label: 'Safe Margin\n20×–25×', min: 0, max: 25, color: 'var(--green)' },
  { label: 'Neutral\n26×–29×', min: 25, max: 30, color: 'var(--yellow)' },
  { label: 'Stretched\n30×+', min: 30, max: 40, color: 'var(--red)' },
]

// Action guidance for true forward P/E
const ACTION_FORWARD: Record<string, string> = {
  Green:       'Buy zone — forward valuations offer a margin of safety. Selective accumulation is appropriate.',
  Yellow:      'Neutral territory — hold existing positions, but avoid aggressive new buying.',
  Red:         'Halt new accumulation — be patient with fresh buying until the P/E compresses.',
  Neutral:     'Transitional zone — neither clearly stretched nor firmly in the buy range. Hold and monitor.',
  BelowBuyZone:'Below the historical buy zone — strong forward margin of safety. Quality accumulation is warranted, but verify fundamentals.',
}

// Softer action guidance for trailing / TTM-derived proxy
const ACTION_PROXY: Record<string, string> = {
  Green:       'Proxy valuation in the buy-zone range — directional signal only; verify with true forward P/E before acting.',
  Yellow:      'Proxy valuation in neutral territory — treat as a directional read; not a clean accumulation signal.',
  Red:         'Proxy valuation in stretched territory — directional caution warranted, but do not treat as a hard pause trigger without true forward P/E confirmation.',
  Neutral:     'Proxy valuation in transitional range — no clean signal; monitor for true forward P/E data.',
  BelowBuyZone:'Proxy valuation below historical buy zone — directional value signal; confirm with true forward P/E before adding.',
}

export function ValuationTriggerCard({ val }: Props) {
  const pe = val.forward_pe ?? null
  const zone = val.zone
  const zoneLabel = val.zone_label
  const isFallback = val.is_fallback ?? false

  const sourceNote = val.source_note ?? null
  const metricName = val.metric_name
    ? (isFallback ? val.metric_name : `${val.metric_name} (cap-weighted)`)
    : (isFallback ? 'QQQ P/E Proxy' : 'Forward P/E (cap-weighted)')
  const objectLabel = val.object_label ?? null
  const providerLabel = val.provider ? val.provider.toUpperCase() : null
  const coverageCount = val.coverage_count ?? null
  const coverageRatio = val.coverage_ratio ?? null

  const zoneColor =
    zone === 'Green' ? 'var(--green)' :
    zone === 'Yellow' ? 'var(--yellow)' : 'var(--red)'

  // Needle position (clamp to visible range)
  const DISPLAY_MAX = 40
  const peClamp = pe !== null ? Math.min(Math.max(pe as number, 0), DISPLAY_MAX) : null
  const pePct = peClamp !== null ? (peClamp / DISPLAY_MAX) * 100 : null

  const actionText = pe !== null
    ? (isFallback ? ACTION_PROXY : ACTION_FORWARD)[zone ?? 'Neutral'] ?? ''
    : null

  // Build the subtitle: "Mag 7 Forward P/E · Mag 7 Basket · FMP"
  const subtitleParts = [metricName, objectLabel, providerLabel].filter(Boolean)
  const subtitle = subtitleParts.join(' · ')

  return (
    <Card title="Big Tech Valuation Trigger">
      {pe === null ? (
        <p style={{ fontSize: '12px', color: 'var(--text-muted)', margin: 0, lineHeight: 1.55 }}>
          P/E data unavailable. In <strong>Live</strong> mode the API computes a Mag 7 Forward P/E basket via FMP,
          falling back to Yahoo QQQ proxy if coverage is insufficient. Click <strong>Refresh</strong> to
          bypass cache and re-fetch, or check the <strong>Debug</strong> panel for the source status.{' '}
          <strong>Mock</strong> mode uses a sample snapshot that includes a demo P/E.
        </p>
      ) : (
        <>
          {/* Current value + zone pill */}
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '12px' }}>
            <span style={{ fontSize: '36px', fontWeight: 800, color: zoneColor, letterSpacing: '-0.03em' }}>
              {fmtPE(pe)}
            </span>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
              <span style={{ fontSize: '11px', fontWeight: 700, color: zoneColor, letterSpacing: '0.05em', textTransform: 'uppercase' }}>
                {zoneLabel}
              </span>
              <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
                {subtitle}
              </span>
              {/* Coverage chip — shown for Mag 7 basket */}
              {coverageCount !== null && coverageRatio !== null && (
                <span style={{
                  fontSize: '10px', color: coverageRatio >= 0.8 ? 'var(--green)' : 'var(--yellow)',
                  background: coverageRatio >= 0.8 ? 'var(--green-bg)' : 'var(--yellow-bg)',
                  border: `1px solid ${coverageRatio >= 0.8 ? 'var(--green-dim)' : 'var(--yellow-dim)'}`,
                  borderRadius: '3px', padding: '1px 6px', display: 'inline-block', marginTop: '2px',
                }}>
                  {coverageCount}/7 constituents · {Math.round(coverageRatio * 100)}% mkt cap
                </span>
              )}
            </div>
          </div>

          {/* Proxy caveat banner — shown when basis is not true forward P/E */}
          {isFallback && (
            <div style={{
              padding: '7px 12px', borderRadius: 'var(--radius-sm)',
              background: 'var(--yellow-bg)',
              border: '1px solid var(--yellow-dim)',
              fontSize: '11px', color: 'var(--text-muted)', lineHeight: 1.5,
            }}>
              <strong style={{ color: 'var(--yellow)' }}>Proxy data</strong> — {sourceNote ?? 'Mag 7 basket unavailable; showing directional proxy. Apply softer interpretation vs the speaker\'s exact zones.'}
            </div>
          )}

          {/* Segmented gauge */}
          <div style={{ position: 'relative', paddingBottom: '4px' }}>
            <div style={{ display: 'flex', height: '12px', borderRadius: '6px', overflow: 'hidden', gap: '2px' }}>
              {SEGMENTS.map((seg, i) => {
                const width = ((seg.max - seg.min) / DISPLAY_MAX) * 100
                const peNum = pe as number
                const isActive = peNum >= seg.min && (i === SEGMENTS.length - 1 ? true : peNum < seg.max)
                return (
                  <div key={i} style={{
                    flex: `0 0 ${width}%`,
                    background: isActive ? seg.color : `${seg.color}30`,
                    borderRadius: i === 0 ? '6px 0 0 6px' : i === SEGMENTS.length - 1 ? '0 6px 6px 0' : '0',
                  }} />
                )
              })}
            </div>

            {/* Needle marker */}
            {pePct !== null && (
              <div style={{
                position: 'absolute', top: '-3px',
                left: `calc(${pePct}% - 1px)`,
                width: '3px', height: '18px',
                background: 'white', borderRadius: '2px',
                boxShadow: '0 1px 6px rgba(0,0,0,0.7)',
              }} />
            )}

            {/* Zone threshold labels */}
            <div style={{ display: 'flex', marginTop: '8px', fontSize: '10px', color: 'var(--text-muted)', position: 'relative' }}>
              <span style={{ position: 'absolute', left: `${(20/DISPLAY_MAX)*100}%`, transform: 'translateX(-50%)' }}>20×</span>
              <span style={{ position: 'absolute', left: `${(25/DISPLAY_MAX)*100}%`, transform: 'translateX(-50%)' }}>25×</span>
              <span style={{ position: 'absolute', left: `${(30/DISPLAY_MAX)*100}%`, transform: 'translateX(-50%)' }}>30×</span>
            </div>
          </div>

          {/* Zone labels row */}
          <div style={{ display: 'flex', gap: '2px', marginTop: '14px' }}>
            {SEGMENTS.map((seg, i) => {
              const width = ((seg.max - seg.min) / DISPLAY_MAX) * 100
              const peNum = pe as number
              const isActive = peNum >= seg.min && (i === SEGMENTS.length - 1 ? true : peNum < seg.max)
              return (
                <div key={i} style={{
                  flex: `0 0 ${width}%`,
                  fontSize: '10px',
                  color: isActive ? seg.color : 'var(--text-muted)',
                  fontWeight: isActive ? 700 : 400,
                  textAlign: 'center',
                  whiteSpace: 'pre-line',
                  lineHeight: 1.3,
                }}>
                  {seg.label}
                </div>
              )
            })}
          </div>

          {/* Individual Mag 7 constituent P/Es */}
          {val.constituents && val.constituents.length > 0 && !isFallback && (
            <Mag7ConstituentGrid constituents={val.constituents} />
          )}

          {/* Action guidance */}
          <div style={{
            padding: '10px 14px', borderRadius: 'var(--radius-sm)',
            background: zone === 'Red' ? 'var(--red-bg)' : zone === 'Yellow' ? 'var(--yellow-bg)' : 'var(--green-bg)',
            border: `1px solid ${zone === 'Red' ? 'var(--red-dim)' : zone === 'Yellow' ? 'var(--yellow-dim)' : 'var(--green-dim)'}`,
          }}>
            <p style={{ margin: 0, fontSize: '12px', color: zoneColor, lineHeight: 1.6 }}>
              {actionText}
            </p>
          </div>
        </>
      )}

      {/* Thresholds reminder */}
      <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
        {[
          { label: '20×–25× Buy Zone', c: 'var(--green)' },
          { label: '26×–29× Neutral', c: 'var(--yellow)' },
          { label: '30×+ Pause', c: 'var(--red)' },
        ].map(t => (
          <span key={t.label} style={{ fontSize: '10px', color: t.c, background: `${t.c}18`, border: `1px solid ${t.c}44`, padding: '2px 8px', borderRadius: '3px' }}>
            {t.label}
          </span>
        ))}
        {isFallback && (
          <span style={{ fontSize: '10px', color: 'var(--yellow)', background: 'var(--yellow-bg)', border: '1px solid var(--yellow-dim)', padding: '2px 8px', borderRadius: '3px' }}>
            Proxy — apply softer interpretation
          </span>
        )}
      </div>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Individual Mag 7 constituent forward P/E grid
// ---------------------------------------------------------------------------

function peZoneColor(pe: number | undefined): string {
  if (pe === undefined) return 'var(--text-muted)'
  if (pe >= 30) return 'var(--red)'
  if (pe >= 25) return 'var(--yellow)'
  return 'var(--green)'
}

function peZoneBg(pe: number | undefined): string {
  if (pe === undefined) return 'var(--bg-card-raised)'
  if (pe >= 30) return 'var(--red-bg)'
  if (pe >= 25) return 'var(--yellow-bg)'
  return 'var(--green-bg)'
}

function peZoneBorder(pe: number | undefined): string {
  if (pe === undefined) return 'var(--border-subtle)'
  if (pe >= 30) return 'var(--red-dim)'
  if (pe >= 25) return 'var(--yellow-dim)'
  return 'var(--green-dim)'
}

function Mag7ConstituentGrid({ constituents }: { constituents: ValuationConstituent[] }) {
  return (
    <div>
      <div style={{
        fontSize: '10px', fontWeight: 700, color: 'var(--text-muted)',
        letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: '6px',
      }}>
        Individual Forward P/E
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gap: '4px' }}>
        {constituents.map(c => {
          const pe = c.forward_pe ?? undefined
          const available = pe !== undefined
          return (
            <div key={c.ticker} style={{
              display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '3px',
              padding: '6px 4px',
              borderRadius: '6px',
              background: peZoneBg(pe),
              border: `1px solid ${peZoneBorder(pe)}`,
            }}>
              <span style={{ fontSize: '9px', fontWeight: 700, color: 'var(--text-muted)', letterSpacing: '0.04em' }}>
                {c.ticker}
              </span>
              <span style={{
                fontSize: '12px', fontWeight: 800, lineHeight: 1,
                color: available ? peZoneColor(pe) : 'var(--border)',
              }}>
                {available ? `${pe!.toFixed(1)}×` : '—'}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
