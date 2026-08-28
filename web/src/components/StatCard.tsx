interface StatCardProps {
  label: string
  value: string | number
  type?: 'success' | 'warning' | 'danger' | 'info'
}

export default function StatCard({ label, value, type = 'info' }: StatCardProps) {
  const colorClasses = {
    success: 'text-harness-success',
    warning: 'text-harness-warning',
    danger: 'text-harness-danger',
    info: 'text-harness-accent',
  }

  return (
    <div className="stat-card">
      <span className={`stat-value ${colorClasses[type]}`}>{value}</span>
      <span className="stat-label">{label}</span>
    </div>
  )
}
