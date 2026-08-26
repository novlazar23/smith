# Arbeitsplan: Quant-Plattform Phase 8 — Backtesting Engine

> Phase 7 abgeschlossen (2026-08-25, `edc3509`)

---

## Zusammenfassung

Phase 8 implementiert eine einfache Backtesting-Engine — testet Handelsstrategien
gegen historische Daten mit Tracking von PnL, Drawdown, Win Rate.

---

## Tasks

### P8-1: Backtesting Engine
**Was:** `quant/backtesting.py` — Klasse `BacktestEngine`:
- Führt strategische Trades auf historischen Daten aus
- Trackt PnL, Drawdown, Win Rate, Sharpe Ratio
- Konfigurierbare Strategie-Parameter

**Tests:** 12+ Unit-Tests

### P8-2: Backtest Store
**Was:** `quant/backtest_store.py`
**Tests:** 5+ Unit-Tests

### P8-3: Backtest API Endpoints
**Was:** /quant/backtest/run, /quant/backtest/{symbol}
**Tests:** 4+ neue Tests

### P8-4: Integration + Gate
**Tests:** 4+ Integration-Tests

### P8-5: Documentation + Handoff + Commit
