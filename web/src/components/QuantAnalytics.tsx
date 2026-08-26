import { useState } from 'react'

export default function QuantAnalytics() {
  const [selectedModule, setSelectedModule] = useState('features')

  const modules = [
    { id: 'features', label: 'Features', description: 'RSI, MACD, Bollinger, ATR, VWAP' },
    { id: 'anomalies', label: 'Anomalies', description: 'Z-Score, IQR, Price Shock' },
    { id: 'regime', label: 'Regime', description: 'SMA, ADX, Volatility' },
    { id: 'similarity', label: 'Similarity', description: 'Euclidean, Pearson' },
    { id: 'outcomes', label: 'Forward Outcomes', description: 'Hit Rate, Profit Factor' },
    { id: 'backtest', label: 'Backtesting', description: 'SMA, RSI Strategies' },
  ]

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        {modules.map(mod => (
          <button
            key={mod.id}
            onClick={() => setSelectedModule(mod.id)}
            className={`card text-left transition ${
              selectedModule === mod.id
                ? 'border-harness-accent bg-harness-accent/10'
                : 'hover:border-gray-600'
            }`}
          >
            <div className="font-medium">{mod.label}</div>
            <div className="text-xs text-gray-500 mt-1">{mod.description}</div>
          </button>
        ))}
      </div>
      <div className="card">
        <h3 className="text-lg font-semibold mb-4 capitalize">{selectedModule} Analytics</h3>
        <div className="text-gray-400">
          Select a module above to view analytics and compute features.
        </div>
      </div>
    </div>
  )
}
