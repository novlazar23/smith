import { useEffect, useState } from 'react'
import { apiRequest } from '../api'

interface QuantStatus {
  influxdb_connected: boolean
  quant_enabled: boolean
  buffered_points: number
  feature_version: string
}

export default function QuantAnalytics() {
  const [selectedModule, setSelectedModule] = useState('features')
  const [status, setStatus] = useState<QuantStatus | null>(null)
  const [message, setMessage] = useState('Loading quant status…')
  const modules = [
    { id: 'features', label: 'Features', description: 'RSI, MACD, Bollinger, ATR, VWAP' },
    { id: 'anomalies', label: 'Anomalies', description: 'Z-Score, IQR, Price Shock' },
    { id: 'regime', label: 'Regime', description: 'SMA, ADX, Volatility' },
    { id: 'similarity', label: 'Similarity', description: 'Euclidean, Pearson' },
    { id: 'outcomes', label: 'Forward Outcomes', description: 'Hit Rate, Profit Factor' },
    { id: 'backtest', label: 'Backtesting', description: 'SMA, RSI Strategies' },
  ]

  useEffect(() => {
    apiRequest<QuantStatus>('/quant/status')
      .then(data => {
        setStatus(data)
        setMessage(data.quant_enabled ? 'Quant API online.' : 'Quant API is disabled.')
      })
      .catch(error => setMessage(error instanceof Error ? error.message : 'Quant API unavailable.'))
  }, [])

  return (
    <div className="space-y-6">
      <div className="card flex flex-wrap gap-6 text-sm">
        <div><span className="text-gray-400">Quant:</span> {status?.quant_enabled ? 'Enabled' : 'Disabled'}</div>
        <div><span className="text-gray-400">InfluxDB:</span> {status?.influxdb_connected ? 'Connected' : 'Unavailable'}</div>
        <div><span className="text-gray-400">Feature version:</span> {status?.feature_version || '—'}</div>
        <div><span className="text-gray-400">Buffer:</span> {status?.buffered_points ?? '—'}</div>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        {modules.map(mod => (
          <button key={mod.id} onClick={() => setSelectedModule(mod.id)} className={`card text-left transition ${selectedModule === mod.id ? 'border-harness-accent bg-harness-accent/10' : 'hover:border-gray-600'}`}>
            <div className="font-medium">{mod.label}</div>
            <div className="text-xs text-gray-500 mt-1">{mod.description}</div>
          </button>
        ))}
      </div>
      <div className="card">
        <h3 className="text-lg font-semibold mb-4 capitalize">{selectedModule} Analytics</h3>
        <div className="text-gray-400">{message} Use Backtesting for an executable deterministic smoke test; data-driven analytics become available after OHLCV ingestion.</div>
      </div>
    </div>
  )
}
