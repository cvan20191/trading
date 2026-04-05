import type { PlaybookSummary as PlaybookSummaryType } from '../../types/summary'
import { HeadlineBanner } from './HeadlineBanner'
import { WatchpointsList } from './WatchpointsList'
import { RiskFlags } from './RiskFlags'
import { TeachingNote } from './TeachingNote'
import { SummaryMetaBadge } from './SummaryMetaBadge'

interface Props {
  summary: PlaybookSummaryType
}

export function PlaybookSummary({ summary }: Props) {
  return (
    <div style={styles.wrapper}>
      <div style={styles.header}>
        <h1 style={styles.title}>Daily Playbook</h1>
        <p style={styles.subtitle}>
          Macro regime summary · Speaker-faithful framework
        </p>
      </div>

      <div style={styles.sections}>
        <HeadlineBanner summary={summary} />
        <RiskFlags summary={summary} />
        <WatchpointsList summary={summary} />
        <TeachingNote summary={summary} />
        <SummaryMetaBadge summary={summary} />
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  wrapper: {
    display: 'flex',
    flexDirection: 'column',
    gap: '20px',
  },
  header: {
    borderBottom: '1px solid var(--border)',
    paddingBottom: '16px',
  },
  title: {
    fontSize: '22px',
    fontWeight: 700,
    color: 'var(--text-primary)',
    letterSpacing: '-0.02em',
  },
  subtitle: {
    fontSize: '13px',
    color: 'var(--text-muted)',
    marginTop: '4px',
  },
  sections: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
  },
}
