"""Performance-Integrationstests (Phase 10, P10-4).

Verifizieren die Zusammensetzung ``FeatureCache`` + ``BatchProcessor`` mit
mocktem ``FeatureEngine`` und mocktem ``InfluxDBStore``: kein Netzwerk, kein
Docker, keine echte InfluxDB.

Kontrakt:

- ``FeatureCache``: TTL-basiertes Ablauf + LRU-Eviction, ``stats()`` mit
  Hit/Miss-Zählung, ``cleanup_expired()`` für aktive Aufräumzyklen.
- ``BatchProcessor``: ``create_job``/``process`` über eine pro-Symbol-
  Callback; Status-Tracking (``completed``/``processed``/``errors``).
- Cache vor dem teuren Pfad (InfluxDB-Query + Engine-Compute) verbessert
  die Wiederverarbeitung desselben Symbol-Universums messbar (weniger
  Store-/Engine-Aufrufe), während TTL-Ablauf stale Daten verhindert.
"""
from __future__ import annotations

import asyncio
import math
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from trading_harness.quant.batch_processor import BatchJob, BatchProcessor
from trading_harness.quant.feature_cache import FeatureCache
from trading_harness.quant.features import FeatureEngine
from trading_harness.quant.influxdb_client import InfluxDBStore

CANDLE_COUNT = 50


# ----------------------------------------------------------------------
# Hilfsfunktionen (vollständig gemockt)
# ----------------------------------------------------------------------


def _candle(index: int) -> dict[str, Any]:
    """Synthetische Kerze mit Trend + Sinus-Overlay (deterministisch)."""
    base = 100.0 + 0.5 * index
    close = base + 4.0 * math.sin(index / 3.0)
    return {
        "open": base,
        "high": max(base, close) + 1.5,
        "low": min(base, close) - 1.5,
        "close": close,
        "volume": 1000.0 + 10.0 * (index % 7),
    }


def _candle_series(count: int) -> list[dict[str, Any]]:
    """Erzeugt ``count`` deterministische Kerzen."""
    return [_candle(i) for i in range(count)]


def make_engine_mock() -> MagicMock:
    """Gemockter FeatureEngine: deterministischer compute()-Output."""
    engine = MagicMock(spec=FeatureEngine)
    engine.compute.return_value = {
        "rsi": 61.5,
        "macd": {"macd": 1.2, "signal": 1.0, "histogram": 0.2},
        "bollinger": {"upper": 105.0, "middle": 100.0, "lower": 95.0, "bandwidth": 0.1},
        "atr": 2.5,
        "volatility": 1.1,
        "vwap": 100.5,
        "feature_count": 6,
        "computation_time_ms": 0.4,
    }
    return engine


def make_store_mock(row_count: int = 30) -> MagicMock:
    """Gemockter InfluxDBStore: async API, keine echte Verbindung."""
    store = MagicMock(spec=InfluxDBStore)
    store.is_available = True
    store.query.return_value = [{"close": 100.0 + i} for i in range(row_count)]
    return store


def make_symbol_processor(store: MagicMock) -> Any:
    """Pro-Symbol-Processor: holt Kerzen über den gemockten InfluxDB-Store."""

    def process_symbol(symbol: str) -> dict[str, Any]:
        rows = asyncio.run(store.query(f'symbol == "{symbol}"'))
        return {"symbol": symbol, "candles": len(rows)}

    return process_symbol


class TestPerformanceIntegration:
    def test_cache_with_feature_engine(self):
        """Cache vor dem Engine-Compute: Wiederholzugriff = Cache-Hit."""
        engine = make_engine_mock()
        cache = FeatureCache(max_size=100)
        candles = _candle_series(CANDLE_COUNT)
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

        def cached_compute(symbol: str) -> dict[str, Any]:
            cached = cache.get(symbol)
            if cached is not None:
                return cached
            features = engine.compute(candles)
            cache.put(symbol, features)
            return features

        first = [cached_compute(symbol) for symbol in symbols]
        second = [cached_compute(symbol) for symbol in symbols]

        # Engine wird nur bei Miss aufgerufen: genau einmal pro Symbol.
        assert engine.compute.call_count == 3
        for original, cached_value in zip(first, second, strict=True):
            assert cached_value == original
            assert cached_value["feature_count"] == 6

        stats = cache.stats()
        assert stats.hits == 3
        assert stats.misses == 3
        assert stats.hit_rate == pytest.approx(0.5)
        assert stats.size == 3
        assert cache.size == 3

    def test_batch_process_multiple_symbols(self):
        """Batch-Job über 10 Symbole: alle verarbeitet, Store je 1x abgefragt."""
        store = make_store_mock()
        processor = BatchProcessor()
        symbols = [f"SYM{i:02d}USDT" for i in range(10)]

        job_id = processor.create_job(symbols)
        job = processor.process(job_id, make_symbol_processor(store))

        assert isinstance(job, BatchJob)
        assert job.status == "completed"
        assert job.processed == 10
        assert job.total == 10
        assert job.errors == []
        assert set(job.results) == set(symbols)
        for symbol, result in job.results.items():
            assert result["symbol"] == symbol
            assert result["candles"] == 30

        # Der gemockte InfluxDB-Store wurde exakt einmal pro Symbol befragt.
        assert store.query.call_count == 10

        status = processor.get_status()
        assert status.total_jobs == 1
        assert status.completed_jobs == 1
        assert status.total_symbols == 10
        assert status.processed_symbols == 10

    def test_cache_cleanup_during_batch(self):
        """Abgelaufene Cache-Einträge werden während der Batch-Verarbeitung entfernt."""
        cache = FeatureCache(max_size=100)
        now = time.time()
        for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
            cache.put(symbol, {"rsi": 55.0})
        # Deterministisches Altern ohne Sleep: TTL determiniert Ablauf.
        for symbol in ("BTCUSDT", "ETHUSDT"):
            cache._cache[symbol].created_at = now - (cache.default_ttl + 10.0)

        def process_with_cleanup(symbol: str) -> dict[str, Any]:
            # Cleanup-Zyklus während der Batch-Verarbeitung.
            cache.cleanup_expired()
            if symbol == "XRPUSDT":
                cache.put(symbol, {"rsi": 40.0})
            return {"symbol": symbol}

        processor = BatchProcessor()
        job_id = processor.create_job(["XRPUSDT", "DOGEUSDT", "LTCUSDT"])
        job = processor.process(job_id, process_with_cleanup)

        assert job.status == "completed"
        assert job.processed == 3
        assert job.errors == []

        # Abgelaufene Einträge weg; gültiger + neu hinzugekommener Eintrag bleibt.
        assert "BTCUSDT" not in cache
        assert "ETHUSDT" not in cache
        assert "SOLUSDT" in cache
        assert "XRPUSDT" in cache
        assert cache.size == 2
        assert cache.get("SOLUSDT") == {"rsi": 55.0}

    def test_batch_with_cache(self):
        """Zweiter Batch-Durchlauf desselben Universums läuft komplett aus dem Cache."""
        store = make_store_mock()
        engine = make_engine_mock()
        cache = FeatureCache(max_size=100)
        symbols = [f"SYM{i:02d}USDT" for i in range(10)]

        def cached_symbol_processor(symbol: str) -> dict[str, Any]:
            cached = cache.get(symbol)
            if cached is not None:
                return {"symbol": symbol, "source": "cache"}
            rows = asyncio.run(store.query(f'symbol == "{symbol}"'))
            engine.compute(rows)
            cache.put(symbol, rows)
            return {"symbol": symbol, "source": "influxdb"}

        processor = BatchProcessor()

        first_id = processor.create_job(list(symbols))
        first = processor.process(first_id, cached_symbol_processor)
        second_id = processor.create_job(list(symbols))
        second = processor.process(second_id, cached_symbol_processor)

        assert first.status == "completed"
        assert second.status == "completed"
        assert first.processed == 10
        assert second.processed == 10

        # Erster Durchlauf: 10 Store-Queries + 10 Engine-Computes.
        assert store.query.call_count == 10
        assert engine.compute.call_count == 10

        # Zweiter Durchlauf: Cache eliminiert Store- und Engine-Aufrufe.
        assert store.query.call_count == 10
        assert engine.compute.call_count == 10
        assert all(result["source"] == "cache" for result in second.results.values())
        assert all(result["source"] == "influxdb" for result in first.results.values())

        stats = cache.stats()
        assert stats.hits == 10
        assert stats.misses == 10
        assert stats.total_accesses == 20
        assert stats.hit_rate == pytest.approx(0.5)

    def test_batch_with_cache_ttl_expiry(self):
        """TTL-Ablauf zwingt den zweiten Durchlauf zur Neuerechnung (keine Stale-Daten)."""
        store = make_store_mock()
        engine = make_engine_mock()
        cache = FeatureCache(max_size=100)
        symbols = ["BTCUSDT", "ETHUSDT"]

        def cached_symbol_processor(symbol: str) -> dict[str, Any]:
            cached = cache.get(symbol)
            if cached is not None:
                return {"symbol": symbol, "source": "cache"}
            rows = asyncio.run(store.query(f'symbol == "{symbol}"'))
            engine.compute(rows)
            cache.put(symbol, rows)
            return {"symbol": symbol, "source": "influxdb"}

        processor = BatchProcessor()
        job_id = processor.create_job(list(symbols))
        first = processor.process(job_id, cached_symbol_processor)
        assert first.status == "completed"
        assert store.query.call_count == 2
        assert engine.compute.call_count == 2

        # Alle Cache-Einträge deterministisch ablaufen lassen.
        now = time.time()
        for entry in cache._cache.values():
            entry.created_at = now - (cache.default_ttl + 10.0)

        retry_id = processor.create_job(list(symbols))
        second = processor.process(retry_id, cached_symbol_processor)

        assert second.status == "completed"
        # Stale Einträge verworfen: Store + Engine werden erneut aufgerufen.
        assert store.query.call_count == 4
        assert engine.compute.call_count == 4
        assert all(result["source"] == "influxdb" for result in second.results.values())
        stats = cache.stats()
        assert stats.hits == 0
        assert stats.misses == 4
