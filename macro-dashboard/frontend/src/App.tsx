import { useState } from 'react'
import { DashboardPage } from './components/Dashboard/DashboardPage'
import { ReplayLabPage } from './components/ReplayLab/ReplayLabPage'

type Page = 'live' | 'replay'

export default function App() {
  const [page, setPage] = useState<Page>('live')

  return (
    <>
      {/* Global spinner keyframe (used by ReplayLabPage loading state) */}
      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>

      {/* Minimal tab bar */}
      <nav style={s.nav}>
        <div style={s.navInner}>
          <TabButton label="Live Dashboard" active={page === 'live'} onClick={() => setPage('live')} />
          <TabButton label="Replay Lab" active={page === 'replay'} onClick={() => setPage('replay')} beta />
        </div>
      </nav>

      {/* Both pages are always mounted so React preserves their state across tab switches.
          The inactive page is hidden via CSS rather than unmounted. */}
      <div style={{ display: page === 'live' ? 'block' : 'none' }}>
        <DashboardPage />
      </div>
      <div style={{ display: page === 'replay' ? 'block' : 'none' }}>
        <ReplayLabPage onGoLive={() => setPage('live')} />
      </div>
    </>
  )
}

interface TabButtonProps {
  label: string
  active: boolean
  onClick: () => void
  beta?: boolean
}

function TabButton({ label, active, onClick, beta }: TabButtonProps) {
  return (
    <button
      onClick={onClick}
      style={{
        ...s.tab,
        ...(active ? s.tabActive : {}),
      }}
    >
      {label}
      {beta && <span style={s.betaBadge}>BETA</span>}
    </button>
  )
}

const s: Record<string, React.CSSProperties> = {
  nav: {
    background: 'var(--bg-card)',
    borderBottom: '1px solid var(--border)',
    position: 'sticky' as const,
    top: 0,
    zIndex: 10,
  },
  navInner: {
    maxWidth: '1200px',
    margin: '0 auto',
    padding: '0 20px',
    display: 'flex',
    gap: '0',
  },
  tab: {
    background: 'none',
    border: 'none',
    borderBottom: '2px solid transparent',
    color: 'var(--text-muted)',
    cursor: 'pointer',
    fontSize: '13px',
    fontWeight: 600,
    padding: '12px 16px',
    transition: 'color 0.15s',
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
  },
  tabActive: {
    borderBottomColor: 'var(--blue, #3b82f6)',
    color: 'var(--text)',
  },
  betaBadge: {
    fontSize: '8px',
    fontWeight: 800,
    letterSpacing: '0.08em',
    color: 'var(--yellow, #eab308)',
    border: '1px solid var(--yellow, #eab308)',
    borderRadius: '3px',
    padding: '1px 4px',
  },
}
