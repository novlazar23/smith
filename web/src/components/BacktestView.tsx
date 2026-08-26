export default function BacktestView() {
  return (
    <div className="space-y-6">
      <div className="card">
        <h3 className="text-lg font-semibold mb-4">Run Backtest</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm text-gray-400 mb-1">Symbol</label>
            <input
              type="text"
              defaultValue="BTCUSDT"
              className="w-full px-3 py-2 bg-harness-bg border border-harness-border rounded-lg focus:outline-none focus:border-harness-accent"
            />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-1">Strategy</label>
            <select className="w-full px-3 py-2 bg-harness-bg border border-harness-border rounded-lg focus:outline-none focus:border-harness-accent">
              <option>SMA Crossover</option>
              <option>RSI</option>
            </select>
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-1">Timeframe</label>
            <select className="w-full px-3 py-2 bg-harness-bg border border-harness-border rounded-lg focus:outline-none focus:border-harness-accent">
              <option>1m</option>
              <option>5m</option>
              <option>1h</option>
              <option>1d</option>
            </select>
          </div>
        </div>
        <button className="mt-4 px-4 py-2 bg-harness-accent/20 text-harness-accent rounded-lg hover:bg-harness-accent/30 transition">
          Run Backtest
        </button>
      </div>
      <div className="card">
        <h3 className="text-lg font-semibold mb-4">Results</h3>
        <div className="text-gray-400">No backtest results yet.</div>
      </div>
    </div>
  )
}
