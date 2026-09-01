"""Orchestrator-Service — periodische Shadow-Pipeline (Analyse, Konsens, Audit).

Führt die getrackte ``OrchestratorPipeline`` zyklisch auf ClickHouse-Candles
aus und persistiert jede Entscheidung in PostgreSQL (``shadow_decisions``).
Die Ausführung ist strikt Shadow: es werden **nie** Orders ausgeführt,
unabhängig vom Feature-Flag ``live_trading_enabled``.
"""
