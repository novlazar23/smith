# EPIC-09: Strategy, Portfolio and Risk

## Problem
EPIC-01 bis EPIC-08 haben die Infrastruktur, Datenpipeline, Agenten, Konsens und Orchestrator gebaut.
Jetzt fehlt die entscheidende Ebene: Die Umwandlung von Marktprognosen in kosten-, portfolio- und risikobereinigte Entscheidungen.

## Ziel
Überführung einer Marktprognose in eine kosten-, portfolio- und risikobereinigte Entscheidung mit:
- Strategie-Engine (Entry/Stop/Target, EV-Berechnung, Kostenmodell)
- Portfolio-Management (Exposure, Constraints, Rebalancing)
- Risk-Gates (Hard/Soft Limits, Veto, Drawdown, Liquidität)
- Final Decision Engine (NO_TRADE Logic, immutable risk veto)

## Abhängigkeiten
- EPIC-08 (Konsens, Contrarian, Multi-Timeframe)
- EPIC-03 (Features, Indikatoren)
- EPIC-02 (Config, Domain Models)

## Arbeitspakete

### WP01: Strategy Package (Proposal, Cost Model, Path Probability)
- packages/strategy/ — models.py, entry.py, targets.py, evaluation.py, engine.py
- StrategyProposal mit direction, entry_type/price/condition, stop, targets
- EV nach Kosten (spread, slippage, fees) berechnen
- expected_return_gross/net, expected_mae/mfe, risk_reward_ratio
- Kostenmodell: size-dependent slippage, taker/maker fees
- File: packages/strategy/*.py

### WP02: Portfolio Package (Snapshot, Exposure, Constraints)
- packages/portfolio/ — __init__.py, base.py, exposure.py, rebalancer.py
- PortfolioSnapshot mit cash, positions, gross/net_exposure, drawdown
- Prüfungen: Instrumentenlimit, Asset-Klasse, Richtung, Korrelationscluster
- Krypto-Exposure, Einzelposition, Drawdown-Abhängigkeit, Liquidität
- File: packages/portfolio/*.py

### WP03: Risk Package (Gates, Veto, Hard Limits, Soft Warnings)
- packages/risk/ — __init__.py, base.py, drawdown.py, position_sizing.py, risk_adjusted.py
- Hard Gates: Datenqualität, Orderbook-Sequenz, nicht kalibriert, Drawdown/Exposure-Limit
- Soft Warnings: Vol, Dissens, Regimeunsicherheit, News-Risiko, Cross-Venue-Divergenz
- Decision: RiskDecision (approved, max_position_size, reduction_factor, blocking_reasons)
- Risk-Veto technisch nicht überschreibbar (immutable)
- File: packages/risk/*.py

### WP04: Decision Engine (Final Decision, NO_TRADE Logic)
- apps/orchestrator/decision.py — integrate strategy + portfolio + risk
- FinalDecisionType: LONG_BIAS, SHORT_BIAS, RANGE, NO_TRADE variants
- Priority chain: risk_veto > risk_not_approved > insufficient_agents > low_confidence > high_uncertainty > insufficient_edge > approve
- Jede Entscheidung mit Begründung + Sperrgründen
- File: apps/orchestrator/decision.py, apps/orchestrator/stages.py

## DoD
- Positiver Forecast ≠ automatischer Trade
- Neg. netto EV → NO_TRADE_INSUFFICIENT_EDGE
- Portfolioexposure vor Risk geprüft
- Risk-Service blockiert Strategien
- Risk-Veto technisch nicht überschreibbar (immutable)
- Schwellwerte versioniert/konfigurierbar
- Jede Entscheidung mit Begründung + Sperrgründen
- Alle Tests bestanden, ruff clean

## Risiken
- Risk-Veto umgangen, Kosten unterschätzt
- Portfolio-Constraints zu lax
- NO_TRADE-Logik unvollständig

## Rollback
- Risk-Konfiguration zurücksetzen, Schwellwerte anpassen

## Definition of Done (Specification Section 33)
16 Kriterien erfüllt: P0/P1 implementiert, alle AT bestehen, Graph reproduzierbar, PIT korrekt, Evidenz validiert, Kalibrierung aktiv, Dependencies berücksichtigt, Schichten getrennt, Risk Veto unveränderbar, Paper Kosten realistisch, Shadow möglich, Degradation/Quarantine funktional, Baseline-Vergleich, 30-Tage-Paper dokumentiert, Live deaktiviert, unabhängiger Review ohne krit. Mängel.