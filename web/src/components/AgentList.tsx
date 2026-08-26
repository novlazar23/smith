import { useState, useEffect } from 'react'

interface Agent {
  id: string
  category: string
  status: string
  generation: number
}

export default function AgentList() {
  const [agents, setAgents] = useState<Agent[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch('/api/agents')
      .then(r => r.json())
      .then(data => {
        setAgents(Array.isArray(data) ? data : [])
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  if (loading) {
    return <div className="text-gray-400">Loading agents...</div>
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <p className="text-gray-400">{agents.length} agents registered</p>
        <button className="px-4 py-2 bg-harness-accent/20 text-harness-accent rounded-lg hover:bg-harness-accent/30 transition">
          Generate New Agent
        </button>
      </div>
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
