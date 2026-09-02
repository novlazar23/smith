"""Candle-Backfill: historische Binance-Futures-1m-Kerzen idempotent nachladen.

One-shot-CLI, die fehlende 1m-Kerzen-Historie von der Binance-Futures-
REST-API lädt und in die ClickHouse-Tabelle ``candles_history`` (Venue
``BINANCE_FUTURES``) schreibt. Bereits vorhandene Zeiträume werden
übersprungen — ein erneuter Lauf lädt nur die fehlenden Lücken.

Aufruf: ``python -m apps.backfill --months 12 --instruments BTC/USDT,ETH/USDT``
"""
