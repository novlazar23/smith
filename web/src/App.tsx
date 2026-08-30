import { useState, useEffect } from 'react'
import Dashboard from './components/Dashboard'
import Sidebar from './components/Sidebar'

function App() {
  const [activeTab, setActiveTab] = useState('dashboard')
  const [health, setHealth] = useState<any>(null)

  useEffect(() => {
    fetch('/health')
      .then(r => r.json())
      .then(setHealth)
      .catch(() => setHealth({ status: 'error' }))
  }, [])

  return (
    <div className="flex h-screen bg-harness-bg">
      <Sidebar activeTab={activeTab} onTabChange={setActiveTab} />
      <main className="flex-1 overflow-auto p-6">
        <Dashboard activeTab={activeTab} health={health} onTabChange={setActiveTab} />
      </main>
    </div>
  )
}

export default App
