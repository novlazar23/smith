interface SidebarProps {
  activeTab: string
  onTabChange: (tab: string) => void
}

const tabs = [
  { id: 'dashboard', label: 'Dashboard', icon: '📊' },
  { id: 'agents', label: 'Agents', icon: '🤖' },
  { id: 'shadow', label: 'Shadow Trading', icon: '📈' },
  { id: 'quant', label: 'Quant Analytics', icon: '🔬' },
  { id: 'backtest', label: 'Backtesting', icon: '⏱️' },
  { id: 'system', label: 'System', icon: '⚙️' },
]

export default function Sidebar({ activeTab, onTabChange }: SidebarProps) {
  return (
    <aside className="w-64 bg-harness-surface border-r border-harness-border flex flex-col">
      <div className="p-4 border-b border-harness-border">
        <h1 className="text-xl font-bold text-harness-accent">Trading Harness</h1>
        <p className="text-xs text-gray-500 mt-1">Evolutionary Research Platform</p>
      </div>
      <nav className="flex-1 p-4 space-y-1">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className={`w-full text-left px-3 py-2 rounded-lg transition-colors ${
              activeTab === tab.id
                ? 'bg-harness-accent/20 text-harness-accent'
                : 'text-gray-400 hover:bg-harness-border hover:text-white'
            }`}
          >
            <span className="mr-2">{tab.icon}</span>
            {tab.label}
          </button>
        ))}
      </nav>
      <div className="p-4 border-t border-harness-border">
        <p className="text-xs text-gray-500">v1.0.0</p>
      </div>
    </aside>
  )
}
