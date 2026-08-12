# EPIC-10: Paper Execution

## Problem
EPIC-09 hat Strategie, Portfolio und Risk gebaut. Jetzt fehlt die realistische Simulation genehmigter Handelsentscheidungen ohne reale Orders.

## Ziel
Realistische Simulation genehmigter Handelsentscheidungen mit:
- Paper Account Management (Cash, Positions, PnL)
- Order Model (Market/Limit/Stop, Partial Fills, Slippage)
- Fee Engine (Maker/Taker, Funding, Latenz)
- Outcome Evaluation (Resolution abgelaufener Prognosen, Scoring, Agentenbewertung)

## Abhängigkeiten
- EPIC-09 (Strategy, Portfolio, Risk, Decision)
- EPIC-02 (Paper Trading Executor, Data Ingestion)
- EPIC-07 (Historical Validation, Baselines)

## Arbeitspakete

### WP01: Paper Executor Package (Base, Executor, Fill Simulator)
- packages/paper/ — __init__.py, base.py, executor.py
- Simuliert: Market/Limit/Stop, Partial Fills, Slippage, variable Spreads
- Latenz, Gebühren, Funding, Orderablauf, Cancel-Replace
- MVP Fill Price: observed price + spread + size-dependent slippage + latency
- Queue-Position, Orderbook-Replay, stochastische Fill-Wahrscheinlichkeit
- File: packages/paper/*.py

### WP02: Paper Executor App (Account, Order Model, Fee Engine)
- apps/paper_executor/ — __init__.py, cli.py
- Account Management: Cash, Margin, Equity, Unrealized/Realized PnL
- Order Model: Order types, fill simulation, price impact
- Fee Engine: Trading fees, funding rates, slippage costs
- File: apps/paper_executor/*.py

### WP03: Position Lifecycle & PnL Tracking
- packages/paper/position_lifecycle.py
- Position lifecycle: Open → Partial Fill → Full Fill → Close → PnL
- PnL calculation: Realized, Unrealized, Cumulative
- Drawdown tracking, exposure tracking
- File: packages/paper/*.py

### WP04: Outcome Evaluation & Scoring
- apps/evaluation_worker/ — resolution, scoring, agent evaluation
- Auflösung abgelaufener Prognosen
- Scoring: Brier, Log Loss, Precision/Recall, Confusion Matrix
- Agentenbewertung, Kalibrierung, Modellvergleich
- Champion-Challenger evaluation
- File: apps/evaluation_worker/*.py

## DoD
- Keine echten API-Schlüssel im Paper-Executor
- Fills berücksichtigen Spread + Slippage + Latenz
- Partial Fills unterstützt
- Gebühren korrekt verbucht
- Positionen/PnL reproduzierbar
- Signal- und Ausführungserfolg getrennt bewertet
- Abgelaufene Signale nicht ausgeführt
- Alle Tests bestanden, ruff clean

## Risiken
- Reale API-Kredentials in Paper-Code, Slippage-Unterschätzung
- Fee-Modell unrealistisch
- PnL-Berechnung fehlerhaft

## Rollback
- Paper-Account zurücksetzen, Fee-Model anpassen