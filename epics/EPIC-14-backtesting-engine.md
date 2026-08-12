# EPIC-14: Backtesting Engine — Historische Simulations-Plattform

## Problem
EPIC-09 bis EPIC-13 haben das komplette Echtzeit-Trading-System gebaut, aber es fehlt eine Backtesting-Engine, die historische Daten nutzt, um Strategien vor dem Live-Einsatz zu validieren. Ohne Backtesting gibt es keine quantitative Basis für:
- Strategy-Vergleiche (Sharpe, Max Drawdown, Win Rate)
- Parameter-Optimierung (Grid Search, Bayesian)
- Walk-Forward- und Purged Cross-Validation
- Performance Attribution (welcher Agent/Signal trägt bei?)
- Slippage- und Fee-Modell-Validierung

## Ziel
Eine backtesting-Engine, die:
- Historische Candles aus ClickHouse replayed
- Das **gleiche** Strategy/Portfolio/Risk-Stack wie EPIC-09 nutzt
- Realistische Kosten (Spread, Slippage, Fees) simuliert
- Walk-Forward-Validierung und Purged K-Fold unterstützt
- Ergebnisse als Report mit Key Metrics exportiert

## Abhängigkeiten
- EPIC-03 (Historische Candles aus ClickHouse)
- EPIC-09 (Strategy, Portfolio, Risk Gates — müssen wiederverwendbar sein)
- EPIC-04 (Agent-Simulation optional)

## Arbeitspakete

### WP01: Core Backtesting Engine
- `packages/backtesting/` — core engine module
- `BacktestEngine` — Candle-by-Candle oder Bar-by-Bar Execution
- Event-Loop für Backtesting (on_bar, on_tick, on_order)
- DataFeed-Abstraktion: ClickHouse, CSV, Parquet
- State: Portfolio Snapshot, OrderBook (simuliert), Fill History
- Config: initial_capital, commission_model, slippage_model, date_range
- File: `packages/backtesting/core.py`, `datafeed.py`, `engine.py`

### WP02: Execution & Cost Models
- `ExecutionModel` — SimulatedFill (market, limit, stop_limit)
- Slippage: Fixed, Percentage, Volume-Based (VWAP)
- Commission: Tiered (maker/taker, volume-dependent)
- Partial fills mit Limit-Price-Probability
- Rejection model (margin call, risk gate failure)
- File: `packages/backtesting/execution.py`, `costs.py`

### WP03: Strategy Integration
- Backtest-compatible Strategy-Interface: `on_bar(data, portfolio, state)` → Signal
- Wiederverwendung von `packages/strategy/engine.py`
- Multi-Strategy-Backtesting (Vergleich, Ensemble)
- Agent-Simulation (random agent, heuristic agent, ML agent)
- File: `packages/backtesting/strategies.py`, `agents.py`

### WP04: Performance Metrics & Reporting
- Key Metrics: Sharpe, Sortino, Calmar, Max Drawdown, Win Rate, Profit Factor
- Trade Log: Entry/Exit, PnL, holding period, signal confidence
- Equity Curve: Daily returns, rolling metrics
- Benchmark Comparison: vs Buy&Hold, vs Index
- Report Export: JSON, CSV, HTML-Report
- File: `packages/backtesting/metrics.py`, `report.py`

### WP05: Walk-Forward & Cross-Validation
- Walk-Forward-Analysis: expanding train/test windows
- Purged K-Fold Cross-Validation (Le et al. 2022)
- Gap-Between-Portfolios für Overfitting-Schutz
- Parameter Sensitivity Analysis (Grid Search, Monte Carlo)
- Optimization via Bayesian Optimization (optional)
- File: `packages/backtesting/validation.py`, `optimization.py`

### WP06: CLI & Notebook Interface
- `scripts/backtest.py` — CLI: `python -m backtest --config backtest.yaml`
- Notebook Template: Strategy Development & Evaluation
- Example: MACD Crossover Backtest mit Ergebnissen
- Config: YAML-basiert (strategies, dates, capital, costs)
- File: `scripts/backtest.py`, `notebooks/backtesting_template.ipynb`, `configs/backtest.yaml`

## DoD
- Backtest läuft gegen historische BTC/USDT 1-year Daten
- Alle Key Metrics korrekt berechnet (manuell verifiziert)
- Walk-Forward-Validierung produces out-of-sample results
- HTML-Report mit Equity Curve und Trade Log
- ruff check: clean
- Tests: 50+ Tests (core, execution, metrics, validation)

## Risiken
- Backtesting vs. Real Trading Discrepanz (Slippage unmodelliert)
- Look-Ahead Bias vermeiden (Data Feed muss korrekt zeitlich sein)
- Performance bei großen Datendaten (>5 Jahre, tick data)

## Rollback
- Backtesting Engine ist separater Package (packages/backtesting/)
- Keine Änderung an Production-Code