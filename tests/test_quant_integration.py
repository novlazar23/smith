"""Integration-Tests für das Quant-Modul (Quant-Plattform, P1-10).

Verifizieren die Zusammensetzung der Module (InfluxDBStore → OHLCVIngestion →
QuantMetrics/QuantHealthChecker → Schema-Kontrakt) ausschließlich mit Mocks:
kein Netzwerk, kein Docker, keine echte InfluxDB.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest

from trading_harness.quant import schema as quant_schema
from trading_harness.quant.influxdb_client import InfluxDBStore
from trading_harness.quant.observability import QuantHealthChecker, QuantMetrics
from trading_harness.quant.ohlcv_ingestion import (
    _TIMEFRAME_SECONDS,
    IngestResult,
    OHLCVIngestion,
)

BUCKET = "market_data"
NS = 1_000_000_000
BASE = datetime(2026, 1, 1, tzinfo=UTC)


def make_store() -> MagicMock:
    """Gemockter Store: async API wird automatisch zu AsyncMock, keine echte Verbindung."""
    return MagicMock(spec=InfluxDBStore)


def make_ingestion(store: MagicMock) -> OHLCVIngestion:
    return OHLCVIngestion(store, default_exchange="binance", bucket=BUCKET)


def candle(time: str, index: int = 0) -> dict[str, Any]:
    return {
        "time": time,
        "open": 100.0 + index,
        "high": 105.0 + index,
        "low": 99.0 + index,
        "close": 103.0 + index,
        "volume": 1000.0 * (index + 1),
    }


def candle_series(step: timedelta, count: int) -> list[dict[str, Any]]:
    """Erzeugt ``count`` aufeinanderfolgende Kerzen ab BASE (UTC)."""
    return [candle((BASE + step * i).isoformat().replace("+00:00", "Z"), i) for i in range(count)]


@pytest.mark.asyncio
async def test_store_to_ingestion_roundtrip() -> None:
    """Ingestion schreibt genau 5 Kerzen in das ``ohlcv``-Measurement (Timeframe als Tag)."""
    store = make_store()
    ing = make_ingestion(store)
    candles = candle_series(timedelta(minutes=1), 5)

    result = await ing.ingest_candles(["BTCUSDT"], "1m", candles)

    assert result == IngestResult(written=5, skipped=0, measurement=quant_schema.OHLCV_MEASUREMENT)
    assert store.write_batch.await_count == 1
    kwargs = store.write_batch.await_args.kwargs
    assert kwargs["measurement"] == quant_schema.OHLCV_MEASUREMENT
    assert kwargs["tags"] == {"symbol": "BTCUSDT", "exchange": "binance", "timeframe": "1m"}
    assert len(kwargs["points"]) == 5
    assert [set(p) for p in kwargs["points"]] == [set(quant_schema.OHLCV_FIELDS)] * 5
    expected_ts = [int((BASE + timedelta(minutes=i)).timestamp()) * NS for i in range(5)]
    assert kwargs["timestamps"] == expected_ts
    # Die Ingestion nutzt ausschließlich den Batch-Pfad — keine Einzel-Punkt-Write.
    store.write_points.assert_not_awaited()


@pytest.mark.asyncio
async def test_downsample_then_ingest() -> None:
    """60 × 1m → genau 1 × 1h mit korrekter OHLCV-Aggregation, beides wird geschrieben."""
    store = make_store()
    ing = make_ingestion(store)
    candles = candle_series(timedelta(minutes=1), 60)

    downsampled = ing.downsample_candles(candles, "1m", "1h")

    assert len(downsampled) == 1
    assert downsampled[0] == {
        "time": "2026-01-01T00:00:00Z",
        "open": candles[0]["open"],
        "high": max(c["high"] for c in candles),
        "low": min(c["low"] for c in candles),
        "close": candles[-1]["close"],
        "volume": sum(c["volume"] for c in candles),
    }

    result = await ing.ingest_and_downsample(["BTCUSDT"], "1m", "1h", candles)

    assert result == IngestResult(written=61, skipped=0, measurement=quant_schema.OHLCV_MEASUREMENT)
    assert store.write_batch.await_count == 2
    source_kwargs, target_kwargs = (call.kwargs for call in store.write_batch.await_args_list)
    assert source_kwargs["tags"]["timeframe"] == "1m"
    assert len(source_kwargs["points"]) == 60
    assert target_kwargs["measurement"] == quant_schema.OHLCV_MEASUREMENT
    assert target_kwargs["tags"] == {"symbol": "BTCUSDT", "exchange": "binance", "timeframe": "1h"}
    assert target_kwargs["points"][0] == {
        name: downsampled[0][name] for name in quant_schema.OHLCV_FIELDS
    }


def test_observability_tracks_writes() -> None:
    """QuantMetrics addiert Writes/Downsamples/Queries/Fehler korrekt in der Summary."""
    metrics = QuantMetrics()
    metrics.record_candles_written(quant_schema.OHLCV_MEASUREMENT, 5)
    metrics.record_candles_written(quant_schema.OHLCV_MEASUREMENT, 7)
    metrics.record_candles_downsampled("1m", "1h", 1)
    metrics.record_query(quant_schema.OHLCV_MEASUREMENT, 12.5)
    metrics.record_error("write_failed", "ingestion")

    assert metrics.total_writes == 12
    summary = metrics.get_summary()

    assert summary["total_writes"] == 12
    assert summary["candles_written"] == {quant_schema.OHLCV_MEASUREMENT: 12}
    assert summary["total_downsampled"] == 1
    assert summary["candles_downsampled"] == {"1m->1h": 1}
    assert summary["total_queries"] == 1
    assert summary["query_duration_total_ms"] == 12.5
    assert summary["query_duration_max_ms"] == 12.5
    assert summary["total_errors"] == 1
    assert summary["errors"] == {"write_failed": 1}
    assert summary["errors_by_component"] == {"ingestion": 1}


def test_health_checker_with_store() -> None:
    """Health-Checker meldet connected=True, wenn der Store verfügbar ist."""
    store = make_store()
    store.is_available = True
    store.buffer_size.return_value = 0
    metrics = QuantMetrics()
    metrics.record_candles_written(quant_schema.OHLCV_MEASUREMENT, 3)
    checker = QuantHealthChecker(store, metrics=metrics, url="http://influxdb:8086", enabled=True)

    health = checker.check_all()

    assert health["influxdb"]["connected"] is True
    assert health["influxdb"]["url"] == "http://influxdb:8086"
    assert health["influxdb"]["buffer_size"] == 0
    assert health["metrics"]["total_writes"] == 3
    assert health["metrics"]["total_queries"] == 0
    assert health["enabled"] is True


def test_health_checker_with_unavailable_store() -> None:
    """Health-Checker meldet connected=False, wenn der Store nicht verfügbar ist."""
    store = make_store()
    store.is_available = False
    store.buffer_size.return_value = 4
    checker = QuantHealthChecker(store, url="http://influxdb:8086", enabled=False)

    health = checker.check_all()

    assert health["influxdb"]["connected"] is False
    assert health["influxdb"]["buffer_size"] == 4
    assert health["metrics"]["total_writes"] == 0
    assert health["enabled"] is False


def test_schema_constants_consistent() -> None:
    """Schema-Konstanten und Timeframe-Tabelle der Ingestion sind konsistent."""
    assert quant_schema.OHLCV_MEASUREMENT == "ohlcv"
    assert quant_schema.OHLCV_TAGS == ("symbol", "exchange", "timeframe")
    assert quant_schema.OHLCV_FIELDS == ("open", "high", "low", "close", "volume")

    pattern = re.compile(r"^(\d+)([mhd])$")
    unit_seconds = {"m": 60, "h": 3600, "d": 86400}
    # Jede unterstützte Timeframe ist in der Ingestion-Tabelle und umgekehrt.
    assert set(quant_schema.SUPPORTED_TIMEFRAMES) == set(_TIMEFRAME_SECONDS)
    for timeframe in quant_schema.SUPPORTED_TIMEFRAMES:
        match = pattern.match(timeframe)
        assert match is not None, timeframe
        assert _TIMEFRAME_SECONDS[timeframe] == int(match[1]) * unit_seconds[match[2]]

    # Jedes ganzzahlige Vielfachen-Paar (Quelle, Ziel) wird vom Downsampling akzeptiert.
    ingestion = make_ingestion(make_store())
    valid_pairs = 0
    for source in quant_schema.SUPPORTED_TIMEFRAMES:
        for target in quant_schema.SUPPORTED_TIMEFRAMES:
            if (
                _TIMEFRAME_SECONDS[target] > _TIMEFRAME_SECONDS[source]
                and _TIMEFRAME_SECONDS[target] % _TIMEFRAME_SECONDS[source] == 0
            ):
                valid_pairs += 1
                assert ingestion.downsample_candles([], source, target) == []
    assert valid_pairs >= 3  # u. a. 1m→5m, 1m→1h, 1h→1d


@pytest.mark.asyncio
async def test_ingestion_validates_timeframes() -> None:
    """Ungültige Timeframes werden deterministisch abgelehnt (ValueError, kein Store-Zugriff)."""
    store = make_store()
    ing = make_ingestion(store)
    candles = candle_series(timedelta(minutes=1), 1)

    with pytest.raises(ValueError, match="unsupported timeframe"):
        await ing.ingest_candles(["BTCUSDT"], "7m", candles)
    with pytest.raises(ValueError, match="unsupported timeframe"):
        await ing.ingest_and_downsample(["BTCUSDT"], "1m", "7m", candles)
    with pytest.raises(ValueError, match="unsupported timeframe"):
        ing.downsample_candles(candles, "1m", "7m")
    with pytest.raises(ValueError, match="must be larger"):
        ing.downsample_candles(candles, "1h", "1m")

    store.write_batch.assert_not_awaited()
