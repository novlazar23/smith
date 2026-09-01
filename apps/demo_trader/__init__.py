"""Demo-Trader — virtuelles Trading mit imaginärem Geld.

Führt die getrackte ``OrchestratorPipeline`` zyklisch mit einem ACTIVEn
Agenten-Ensemble auf ClickHouse-Candles aus und mappt die Konsens-
Entscheidung auf Paper-Trades (long-only) über den ``PaperExecutor``.
Ausgeführt wird ausschließlich auf imaginäres Geld — es werden nie
reale Orders platziert. Persistenz: PostgreSQL (``demo_trades``,
``demo_account``).
"""
