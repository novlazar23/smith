"""Backtest — historisches Backtesting der PRODUKTIONS-Agenten-Entscheidungslogik.

Führt dieselbe Entscheidungslogik wie der ``demo-trader`` (ACTIVE-Ensemble →
``OrchestratorPipeline`` → Konsens → Konfidenz-Gate → long-only Signal) auf
historischen ClickHouse-Kerzen ab und liefert Gate-Kalibrierungs-Analytics
(Szenarien-Vergleich, Konfidenz-Buckets, Gate-Sweep) plus MLflow-Run-Logging
(opt-in).
"""
