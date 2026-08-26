import { useState, useEffect } from 'react'

export default function ShadowTrading() {
  const [status, setStatus] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch('/api/shadow-trading/status')
      .then(r => r.json())
      .then(data => {
        setStatus(data)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  if (loading) {
    return <div className="text-gray-400">Loading shadow trading status...</div>
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="card">
          <div className="flex items-center justify-between">
            <span className="text-gray-400">Status</span>
            <span className={`badge ${
              status?.running ? 'badge-success' : 'badge-warning'
            }`}>
              {status?.running ? 'Running' : 'Stopped'}
            </span>
          </div>
        </div>
        <div className="card">
          <div className="text-gray-400 text-sm">Decisions Today</div>
          <div className="text-2xl font-bold">{status?.decisions_today || 0}</div>
        </div>
        <div className="card">
          <div className="text-gray-400 text-sm">Budget</div>
          <div className="text-2xl font-bold">{status?.budget_max || 0}</div>
        </div>
      </div>
      <div className="flex gap-4">
        <button className="px-4 py-2 bg-harness-success/20 text-harness-success rounded-lg hover:bg-harness-success/30 transition">
          Start Loop
        </button>
        <button className="px-4 py-2 bg-harness-danger/20 text-harness-danger rounded-lg hover:bg-harness-danger/30 transition">
          Stop Loop
        </button>
        <button className="px-4 py-2 bg-harness-surface border border-harness-border rounded-lg hover:bg-harness-border transition">
          Run Once
        </button>
      </div>
    </div>
  )
}
