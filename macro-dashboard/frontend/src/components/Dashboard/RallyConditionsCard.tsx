import type { RallyConditions } from '../../types/summary'
import { Card } from '../ui/Card'
import { rallyLabel } from '../../lib/fmt'

interface Props { rally: RallyConditions }

function PutChip({ label, active }: { label: string; active: boolean }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: '7px',
      padding: '6px 12px', borderRadius: '5px',
      background: active ? 'var(--green-bg)' : 'var(--bg-card-raised)',
      border: `1px solid ${active ? 'var(--green)' : 'var(--border-subtle)'}`,
      flex: 1,
    }}>
      <span style={{
        width: '7px', height: '7px', borderRadius: '50%', flexShrink: 0,
        background: active ? 'var(--green)' : 'var(--border)',
      }} />
      <span style={{ fontSize: '12px', color: active ? 'var(--green)' : 'var(--text-muted)', fontWeight: active ? 600 : 400 }}>
        {label}
      </span>
    </div>
  )
}

export function RallyConditionsCard({ rally }: Props) {
  const score = rally.rally_fuel_score ?? null
  const label = rallyLabel(score)
  const scoreColor = label === 'High' ? 'var(--green)' : label === 'Medium' ? 'var(--yellow)' : 'var(--red)'

  return (
    <Card title='Rally Conditions — "Shrewd Animal" Monitor'>
      {/* Rally fuel score */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
        <div>
          <div style={{ fontSize: '36px', fontWeight: 800, color: scoreColor, letterSpacing: '-0.03em' }}>
            {score ?? '—'}
          </div>
          <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '1px' }}>/ 100 Rally Fuel</div>
        </div>
        <div style={{ flex: 1 }}>
          {/* Fuel bar */}
          <div style={{ height: '10px', background: 'var(--bg-card-raised)', borderRadius: '5px', overflow: 'hidden', border: '1px solid var(--border-subtle)', marginBottom: '6px' }}>
            <div style={{
              height: '100%', width: `${Math.min(score ?? 0, 100)}%`,
              background: scoreColor, borderRadius: '5px', transition: 'width 0.5s',
            }} />
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: 'var(--text-muted)' }}>
            <span>Low (0–39)</span><span>Medium (40–64)</span><span>High (65+)</span>
          </div>
          <div style={{ marginTop: '6px', fontSize: '13px', fontWeight: 700, color: scoreColor }}>
            {label} Rally Fuel
          </div>
        </div>
      </div>

      {/* Put indicators */}
      <div>
        <div style={{ fontSize: '11px', fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: '8px' }}>
          Policy Backstops (Puts)
        </div>
        <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
          <PutChip label="Fed Put" active={rally.fed_put ?? false} />
          <PutChip label="Treasury Put" active={rally.treasury_put ?? false} />
          <PutChip label="Political Put" active={rally.political_put ?? false} />
        </div>
      </div>

      {/* Shrewd animal note */}
      {rally.market_ignoring_bad_news && (
        <div style={{
          padding: '10px 14px',
          background: 'var(--blue-bg)',
          border: '1px solid var(--blue-dim)',
          borderRadius: 'var(--radius-sm)',
        }}>
          <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--blue)', marginBottom: '3px' }}>
            The Shrewd Animal
          </div>
          <p style={{ margin: 0, fontSize: '12px', color: 'var(--blue)', lineHeight: 1.6, opacity: 0.9 }}>
            Liquidity expectations are strong enough that bad news may currently be getting ignored by the market.
          </p>
        </div>
      )}

      {!rally.market_ignoring_bad_news && score !== null && (score as number) < 40 && (
        <div style={{ padding: '8px 12px', background: 'var(--red-bg)', border: '1px solid var(--red-dim)', borderRadius: 'var(--radius-sm)', fontSize: '12px', color: 'var(--red)' }}>
          Low rally fuel — bad macro data is more likely to be reflected in prices rather than looked through.
        </div>
      )}
    </Card>
  )
}
