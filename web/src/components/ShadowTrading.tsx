import { useState, useEffect } from 'react'
import { apiRequest } from '../api'

export default function ShadowTrading() {
  const [status, setStatus] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [action, setAction] = useState('')
  const [message, setMessage] = useState('')

  const loadStatus = () => apiRequest<any>('/shadow-trading/status').then(setStatus)

  useEffect(() => {
    loadStatus()
      .catch(error => setMessage(error instanceof Error ? error.message : 'Status unavailable.'))
      .finally(() => setLoading(false))
  }, [])

  const runAction = async (name: 'start' | 'stop' | 'run-once') => {
    setAction(name)
    setMessage('')
    try {
      await apiRequest(`/shadow-trading/${name}`, { method: 'POST' })
      await loadStatus()
      setMessage(name === 'run-once' ? 'Iteration completed.' : `Loop ${name} completed.`)
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Action failed.')
    } finally {
      setAction('')
    }
  }

  if (loading) {
    return <div className="text-gray-400">Loading shadow trading status...</div>
  }

  const running = status?.status === 'RUNNING'

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="card">
          <div className="flex items-center justify-between">
            <span className="text-gray-400">Status</span>
            <span className={`badge ${
              running ? 'badge-success' : 'badge-warning'
            }`}>
              {running ? 'Running' : 'Stopped'}
            </span>
          </div>
        </div>
        <div className="card">
          <div className="text-gray-400 text-sm">Decisions Today</div>
          <div className="text-2xl font-bold">{status?.decisions_today || 0}</div>
        </div>
        <div className="card">
          <div className="text-gray-400 text-sm">Iterations</div>
          <div className="text-2xl font-bold">{status?.iteration_count || 0}</div>
        </div>
      </div>
      {message && <div className="card text-sm text-gray-300">{message}</div>}
      <div className="flex gap-4">
        <button onClick={() => runAction('start')} disabled={Boolean(action)} className="px-4 py-2 bg-harness-success/20 text-harness-success rounded-lg hover:bg-harness-success/30 transition disabled:opacity-50">
          {action === 'start' ? 'Starting…' : 'Start Loop'}
        </button>
        <button onClick={() => runAction('stop')} disabled={Boolean(action)} className="px-4 py-2 bg-harness-danger/20 text-harness-danger rounded-lg hover:bg-harness-danger/30 transition disabled:opacity-50">
          {action === 'stop' ? 'Stopping…' : 'Stop Loop'}
        </button>
        <button onClick={() => runAction('run-once')} disabled={Boolean(action)} className="px-4 py-2 bg-harness-surface border border-harness-border rounded-lg hover:bg-harness-border transition disabled:opacity-50">
          {action === 'run-once' ? 'Running…' : 'Run Once'}
        </button>
      </div>
    </div>
  )
}
