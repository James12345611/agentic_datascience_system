import type { SummaryCard } from '../types/platform'

export function StatCard({ title, value, delta, note, tone }: SummaryCard) {
  return (
    <article className={`stat-card ${tone}`}>
      <div className="stat-label">{title}</div>
      <div className="stat-value">{value}</div>
      <div className="stat-delta">{delta}</div>
      <div className="stat-note">{note}</div>
    </article>
  )
}
