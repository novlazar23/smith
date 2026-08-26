import { useState, useEffect } from 'react'
import StatCard from './StatCard'
import AgentList from './AgentList'
import ShadowTrading from './ShadowTrading'
import QuantAnalytics from './QuantAnalytics'
import BacktestView from './BacktestView'
import SystemStatus from './SystemStatus'

interface DashboardProps {
  activeTab: string
  health: any
}

export default function Dashboard({ activeTab, health }: DashboardProps) {
  const [stats, setStats] = useState({
    agents: 0,
    activeAgents: 0,
    shadowTrading: false,
    quantModules: 13,
  })

  useEffect(() => {
    fetch('/api/agents')
      .then(r => r.json())
      .then(data => {
        setStats(prev => ({
          ...prev,
          agents: data.length || 0,
          activeAgents: data.filter((a: any) => a.status === 'ACTIVE').length || 0,
        }))
      })
      .catch(() => {})
  }, [])

  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <StatCard
                label="System Status"
                value={health?.status === 'ok' ? 'Healthy' : 'Degraded'}
                type={health?.status === 'ok' ? 'success' : 'warning'}
              />
              <StatCard label="Total Agents" value={stats.agents} type="info" />
              <StatCard label="Active Agents" value={stats.activeAgents} type="info" />
              <StatCard label="Quant Modules" value={stats.quantModules} type="info" />
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="card">
                <h3 className="text-lg font-semibold mb-4">Recent Activity</h3>
                <div className="space-y-2 text-sm text-gray-400">
                  <p>System initialized</p>
                  <p>Shadow trading loop ready</p>
                  <p>Quant platform online</p>
                </div>
              </div>
              <div className="card">
                <h3 className="text-lg font-semibold mb-4">Quick Actions</h3>
                <div className="space-y-2">
                  <button className="w-full px-4 py-2 bg-harness-accent/20 text-harness-accent rounded-lg hover:bg-harness-accent/30 transition">
                    Start Shadow Trading
                  </button>
                  <button className="w-full px-4 py-2 bg-harness-surface border border-harness-border rounded-lg hover:bg-harness-border transition">
                    Run Backtest
                  </button>
                </div>
              </div>
            </div>
          </div>
        )
      case 'agents':
        return <AgentList />
      case 'shadow':
        return <ShadowTrading />
      case 'quant':
        return <QuantAnalytics />
      case 'backtest':
        return <BacktestView />
      case 'system':
        return <SystemStatus health={health} />
      default:
        return null
    }
  }

  return (
    <div>
      <h2 className="text-2xl font-bold mb-6 capitalize">{activeTab}</h2>
      {renderContent()}
    </div>
  )
}
