# EPIC-18: Continuous Optimization — ML, Auto-Tuning, Market Expansion

## Problem
EPIC-01 bis EPIC-17 haben das komplette Trading-System gebaut: Infrastruktur, Agenten, Konsens, Governance, Backtesting, Paper/Live-Execution, Observability. Das System funktioniert, ist aber noch starr konfiguriert und auf eine Asset-Klasse (Krypto) beschränkt. Es fehlen:
- ML-basierte Feature-Importance-Discovery (SHAP für Agenten-Features)
- Adaptive Gewichtungsanpassung im Konsens (gradient descent auf OOS-Performance)
- Anomalie-Erkennung für Market-Regime-Shifts (ungesupervised Learning)
- Optimal Execution Timing (RL-basierte Limit-Order-Platzierung)
- Automatisches Hyperparameter-Tuning für alle Agenten
- Backtest-to-Live Pipeline mit statistischer Signifikanzprüfung
- Multi-Asset-Support (Equities, FX, Commodities)
- Multi-Venue-Integration (Coinbase, Kraken, Bitstamp, traditionelle Exchanges)
- Sentiment-Analyse-Pipeline (Twitter/X, Reddit, News-Wires → strukturierte Features)
- Whale-Alert-Integration (On-Chain + Exchange-Flow-Anomalien)
- Macro-Event-Calendar-Integration (FOMC, CPI, Options-Expiry)
- Automatische Strategie-Dokumentation und Pattern Library
- Agent Performance Leaderboard mit Trendanalyse

## Ziel
Ein lernendes, sich selbst optimierendes Trading-System mit:
- ML-Modellen für Feature-Importance, adaptive Gewichte, Anomalie-Erkennung und Execution-Timing
- Auto-Tuning-Engine für Hyperparameter-Optimierung, automatische Backtest-to-Live-Pipeline, Shadow-to-Paper-Promotion mit statistischer Signifikanz, Rollback bei Performance-Degradation
- Multi-Asset-Support (bis 100 Instruments mit Korrelations-Clustering), Multi-Venue-Integration, Cross-Asset-Regime-Detection
- Intelligence-Layer: Sentiment-Analyse, Whale-Alerts, Macro-Events, Social-Sentiment-Consensus-Agent
- Knowledge-Base: automatische Strategie-Dokumentation, Pattern Library, Agent Leaderboard, quarterly Strategy Review Automation

## Priorität
P2 (long-term)

## Abhängigkeiten
- EPIC-01 bis EPIC-17 (alle vorangehenden Epics)
  - EPIC-02 (Config, Domain Models — Basis für neue Agenten/Modelle)
  - EPIC-03 (Features, Indikatoren — ML-Feature-Engineering baut auf)
  - EPIC-05 (ML Agents — neue ML-Modelle erweitern ML-Agenten-Katalog)
  - EPIC-07 (Historical Validation — SHAP/Backtest-Pipeline nutzt Validation-Stack)
  - EPIC-08 (Consensus — adaptive Gewichte und Anomalie-Erkennung verbessern Konsens)
  - EPIC-09 (Strategy, Portfolio, Risk — neue Assets/Korrelationen erweitern Portfolio/Risk)
  - EPIC-10 (Paper Execution — Backtest-to-Live nutzt Paper-Account)
  - EPIC-11 (Governance — ML-Agenten und neue Patterns durch Governance-Lifecycle)
  - EPIC-14 (Backtesting Engine — Auto-Tuning und statistical significance testing)

## Arbeitspakete

### WP01: ML Model Integration
- packages/ml/ — __init__.py, shap_analyzer.py, adaptive_weights.py, anomaly_detector.py, rl_execution.py
- Feature Importance Auto-Discovery: SHAP-Werte für alle Agenten-Features, ranking, feature-drift detection
- Adaptive Weighting: gradient descent auf Konsens-Gewichten, rolling OOS-performance als Loss-Funktion, weekly retraining
- Anomaly Detection: Isolation Forest / Autoencoder für Market Regime Shifts, regime-change alerting, confidence adjustment
- Optimal Execution Timing: RL-based limit order placement (PPO/DQN), reward = fill price improvement - cost, simulated environment aus historical orderbook
- File: packages/ml/*.py

### WP02: Auto-Tuning Engine
- packages/autotuning/ — __init__.py, hyperparameter_optimizer.py, backtest_pipeline.py, promotion_engine.py, rollback_manager.py
- Hyperparameter Optimization: Regime-Detection-Schwellwerte, Indicator-Parameter, Strategy-Regeln (Optuna/Bayesian Optimization)
- Backtest-to-Live Pipeline: config validation → backtest → OOS check → shadow → paper → live, automated gate checks
- Shadow-to-Paper Promotion: statistical significance testing (Diebold-Mariano, Clark-West), minimum performance delta, confidence intervals
- Rollback Automation: live performance degradation detection (EWMA chart), automatic rollback to last known good config, alert on rollback
- File: packages/autotuning/*.py

### WP03: Market Expansion
- packages/market_expansion/ — __init__.py, asset_universe.py, venue_manager.py, correlation_cluster.py, regime_detector.py
- Multi-Asset Support: Equities, FX, Commodities alongside Crypto, asset-specific features and indicators, normalization across assets
- Additional Venues: Coinbase, Kraken, Bitstamp, traditionelle Exchanges (interactive brokers API), unified exchange abstraction
- Multi-Instrument Portfolio: up to 100 instruments, correlation clustering (agglomerative/hierarchical), cluster-level exposure limits
- Cross-Asset Regime Detection: regime detection across asset classes, macro-driven regime classification, cross-asset momentum/mean-reversion
- File: packages/market_expansion/*.py

### WP04: Intelligence Layer
- packages/intelligence/ — __init__.py, sentiment_pipeline.py, whale_alert.py, macro_calendar.py, social_consensus.py
- Sentiment Analysis Pipeline: Twitter/X API, Reddit API, News Wires (RSS/API) → NLP features (VADER, FinBERT, custom transformer), structured sentiment features
- Whale Alert Integration: On-chain whale movement detection (Etherscan, etc.), exchange flow anomalies (net inflow/outflow), whale activity scoring
- Macro Event Calendar: FOMC, CPI, NFP, Options Expiry, Dividend Dates, earnings seasons, event risk scoring and position sizing adjustment
- Social Sentiment Consensus Agent: aggregates sentiment from all sources, calculates consensus confidence, generates structured signal with confidence bands
- File: packages/intelligence/*.py

### WP05: Knowledge Base
- packages/knowledge_base/ — __init__.py, auto_documenter.py, pattern_library.py, performance_leaderboard.py, quarterly_review.py
- Auto-Generated Strategy Documentation: from analysis runs, auto-generates strategy reports with performance metrics, key decisions, agent contributions
- Pattern Library: learned successful patterns from backtest + live performance, pattern classification (trend-following, mean-reversion, breakout, arbitrage), pattern effectiveness tracking
- Agent Performance Leaderboard: trend analysis (rolling Sharpe, win rate, calibration), agent comparison across assets/timeframes, marginal contribution ranking
- Quarterly Strategy Review: automated quarterly review package, performance attribution, strategy health check, recommendations for deprecation/promotion
- File: packages/knowledge_base/*.py

## DoD
- SHAP-Feature-Importance für alle Agenten berechnet und dokumentiert
- Adaptive Konsens-Gewichte weekly retrainiert, OOS-Performance verbessert sich im Vergleich zu statischen Gewichten
- Anomalie-Erkennung detektiert mindestens 1 historischen Regime-Shift (z.B. March 2020, FTX Crash)
- RL-Execution mindestens 5% price improvement gegenüber fixed-limit placement im Backtest
- Hyperparameter-Tuning mindestens einen Agenten verbessert (OOS Sharpe +0.1)
- Backtest-to-Live Pipeline funktioniert für mindestens einen neuen Agenten (shadow → paper → live)
- Shadow-to-Paper Promotion mit statistischem Test (p < 0.05)
- Rollback-System detektiert und rollt bei simulierter Degradation korrekt zurück
- Multi-Asset: mindestens Equities + FX parallel gemanagt (3 Asset-Klassen)
- Multi-Venue: mindestens 2 zusätzliche Exchanges integriert
- Correlation Clustering für 50+ Instruments berechnet
- Sentiment-Pipeline liefert strukturierte Features (mindestens Twitter/X + Reddit)
- Whale Alerts integiert und als Feature im Agenten-Input verfügbar
- Macro Calendar integriert, Event-Risk-Scoring aktiv
- Social Sentiment Consensus Agent generiert Signal mit Confidence Bands
- Pattern Library enthält mindestens 10 gelernte Patterns
- Agent Leaderboard mit Trendanalyse funktioniert
- Quarterly Review Automation generiert Review-Report
- Alle Tests bestanden, ruff clean

## Risiken
- ML-Modelle überfit on historical data (OOS-Performance schlechter als in-sample)
- Data Quality bei neuen Assets/Venues unzureichend
- Sentiment-Pipeline liefert Rauschen statt Signal (False Positives dominieren)
- Multi-Asset-Korrelationen instabil (breakdown during stress)
- RL-Training instabil, non-convergence
- Auto-Tuning zu aggressiv, Overfit auf spezifische Market Conditions

## Rollback
- ML-Modelle: manual revert auf letzte validierte Version
- Adaptive Gewichte: zurücksetzen auf statische Gewichte (aus EPIC-08)
- Auto-Tuning: Promotion stoppen, Rollback zu bekannt gutem Config
- Sentiment: Pipeline pausieren, nur traditionelle Features verwenden
- Multi-Asset: auf einzelne Asset-Klasse zurückfallen
