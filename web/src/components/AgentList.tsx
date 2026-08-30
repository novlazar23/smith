import { useState, useEffect } from 'react'
import { apiRequest } from '../api'

interface Agent {
  id: string
  category: string
  status: string
  generation: number
}

export default function AgentList() {
  const [agents, setAgents] = useState<Agent[]>([])
  const [loading, setLoading] = useState(true)
  const [creating, setCreating] = useState(false)
  const [message, setMessage] = useState('')

  const loadAgents = () => apiRequest<Agent[]>('/agents')
    .then(data => setAgents(Array.isArray(data) ? data : []))

  useEffect(() => {
    loadAgents().catch(() => setMessage('Could not load agents.')).finally(() => setLoading(false))
  }, [])

  const createAgent = async () => {
    setCreating(true)
    setMessage('')
    try {
      const agent = await apiRequest<Agent>('/agents', {
        method: 'POST',
        body: JSON.stringify({
          category: 'general',
          indicators: ['rsi', 'macd'],
          timeframes: ['1h'],
        }),
      })
      await loadAgents()
      setMessage(`Created ${agent.id}`)
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Could not create agent.')
    } finally {
      setCreating(false)
    }
  }

  if (loading) {
    return <div className="text-gray-400">Loading agents...</div>
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <p className="text-gray-400">{agents.length} agents registered</p>
        <button
          onClick={createAgent}
          disabled={creating}
          className="px-4 py-2 bg-harness-accent/20 text-harness-accent rounded-lg hover:bg-harness-accent/30 transition disabled:opacity-50"
        >
          {creating ? 'Generating…' : 'Generate New Agent'}
        </button>
      </div>
      {message && <div className="card text-sm text-gray-300">{message}</div>}
      <div className="card">
        <table className="w-full">
          <thead>
            <tr className="text-left text-gray-400 border-b border-harness-border">
              <th className="pb-2">ID</th>
              <th className="pb-2">Category</th>
              <th className="pb-2">Status</th>
              <th className="pb-2">Generation</th>
            </tr>
          </thead>
          <tbody>
            {agents.map(agent => (
              <tr key={agent.id} className="border-b border-harness-border last:border-0">
                <td className="py-3 font-mono text-sm">{agent.id}</td>
                <td className="py-3">{agent.category}</td>
                <td className="py-3">
                  <span className={`badge ${
                    agent.status === 'ACTIVE' ? 'badge-success' :
                    agent.status === 'CHAMPION' ? 'badge-warning' :
                    'badge-info'
                  }`}>
                    {agent.status}
                  </span>
                </td>
                <td className="py-3 text-gray-400">{agent.generation}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
