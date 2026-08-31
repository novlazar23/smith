import { useState } from 'react'
import { apiRequest } from '../api'

interface BacktestResponse {
  result: {
    total_trades: number
    win_rate: number
    total_pnl: number
    max_drawdown: number
    sharpe_ratio: number
  }
}

function demoCandles() {
  const start = Date.now() - 80 * 60_000
  return Array.from({ length: 80 }, (_, index) => {
    const base = 50_000 + index * 12 + Math.sin(index / 3) * 450
    const close = base + Math.sin(index / 2) * 120
    return {
      time: new Date(start + index * 60_000).toISOString(),
      open: base,
      high: Math.max(base, close) + 80,
      low: Math.min(base, close) - 80,
      close,
      volume: 10 + index / 10,
      trade_count: 100 + index,
    }
  })
}

export default function BacktestView() {
  const [symbol, setSymbol] = useState('BTCUSDT')
  const [strategy, setStrategy] = useState('sma')
  const [timeframe, setTimeframe] = useState('1m')
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<BacktestResponse['result'] | null>(null)
  const [message, setMessage] = useState('')

  const runBacktest = async () => {
    setRunning(true)
    setMessage('')
    try {
      const response = await apiRequest<BacktestResponse>('/quant/backtest/run', {
        method: 'POST',
        body: JSON.stringify({ symbol, timeframe, exchange: 'demo', strategy, candles: demoCandles() }),
      })
      setResult(response.result)
      setMessage('Backtest completed with deterministic demo candles.')
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Backtest failed.')
    } finally {
      setRunning(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="card">
        <h3 className="text-lg font-semibold mb-4">Run Backtest</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm text-gray-400 mb-1">Symbol</label>
            <input value={symbol} onChange={event => setSymbol(event.target.value)} className="w-full px-3 py-2 bg-harness-bg border border-harness-border rounded-lg focus:outline-none focus:border-harness-accent" />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-1">Strategy</label>
            <select value={strategy} onChange={event => setStrategy(event.target.value)} className="w-full px-3 py-2 bg-harness-bg border border-harness-border rounded-lg focus:outline-none focus:border-harness-accent">
              <option value="sma">SMA Crossover</option>
              <option value="rsi">RSI</option>
            </select>
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-1">Timeframe</label>
            <select value={timeframe} onChange={event => setTimeframe(event.target.value)} className="w-full px-3 py-2 bg-harness-bg border border-harness-border rounded-lg focus:outline-none focus:border-harness-accent">
              <option value="1m">1m</option><option value="5m">5m</option><option value="1h">1h</option><option value="1d">1d</option>
            </select>
          </div>
        </div>
        <p className="mt-3 text-xs text-gray-500">Smoke-test mode: uses 80 deterministic synthetic candles and never submits orders.</p>
        <button onClick={runBacktest} disabled={running || !symbol.trim()} className="mt-4 px-4 py-2 bg-harness-accent/20 text-harness-accent rounded-lg hover:bg-harness-accent/30 transition disabled:opacity-50">
          {running ? 'Running…' : 'Run Backtest'}
        </button>
      </div>
      {message && <div className="card text-sm text-gray-300">{message}</div>}
      <div className="card">
        <h3 className="text-lg font-semibold mb-4">Results</h3>
        {result ? (
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-sm">
            <div><div className="text-gray-400">Trades</div><div className="text-xl font-bold">{result.total_trades}</div></div>
            <div><div className="text-gray-400">Win rate</div><div className="text-xl font-bold">{(result.win_rate * 100).toFixed(1)}%</div></div>
            <div><div className="text-gray-400">PnL</div><div className="text-xl font-bold">{result.total_pnl.toFixed(2)}</div></div>
            <div><div className="text-gray-400">Max drawdown</div><div className="text-xl font-bold">{(result.max_drawdown * 100).toFixed(2)}%</div></div>
            <div><div className="text-gray-400">Sharpe</div><div className="text-xl font-bold">{result.sharpe_ratio.toFixed(2)}</div></div>
          </div>
        ) : <div className="text-gray-400">No backtest results yet.</div>}
      </div>
    </div>
  )
}
