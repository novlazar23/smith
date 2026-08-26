"""Unit-Tests für das Anomaly-Storage (Phase 3, P3-2).

Alle Tests nutzen einen gemockten ``InfluxDBStore`` (AsyncMock für die
async Store-API) — es werden keine echten InfluxDB-Verbindungen
aufgebaut. Der ``AnomalyDetector`` selbst läuft (deterministische
Stdlib-Berechnung); gemockt ist ausschließlich der Storage.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from trading_harness.quant import schema as quant_schema
from trading_harness.quant.anomaly_store import AnomalyResult, AnomalyStore

pytestmark = pytest.mark.asyncio

SYMBOL = "BTCUSDT"
TIMEFRAME = "1m"
EXCHANGE = "binance"
BASE_TS = 1_700_000_000  # Epoch-Sekunden der ersten Kerze (1-Minuten-Abstand)


def _candle_time(index: int) -> str:
    """UTC-ISO-Zeichenkette für die Kerze an Position *index*."""
    moment = datetime.fromtimestamp(BASE_TS + index * 60, tz=UTC)
    return moment.isoformat().replace("+00:00", "Z")


def make_shock_candles(count: int = 26) -> list[dict]:
    """Synthetische OHLCV-Reihe mit Preis-Schock auf der letzten Kerze."""
    candles: list[dict] = []
    for i in range(count):
        price = 100.0 if i < count - 1 else 200.0
        candles.append(
            {
                "time": _candle_time(i),
                "open": price,
                "high": price * 1.01,
                "low": price * 0.99,
                "close": price,
                "volume": 1000.0,
            }
        )
    return candles


def make_stable_candles(count: int = 30) -> list[dict]:
    """Synthetische OHLCV-Reihe ohne Anomalien (stabil, konstantes Volumen)."""
    return [
        {
            "time": _candle_time(i),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1000.0,
        }
        for i in range(count)
    ]


def make_mock_store(available: bool = True) -> MagicMock:
    """InfluxDBStore-Mock: Verfügbarkeits-Flag + async write/query + Bucket-Config."""
    store = MagicMock()
    store._bucket = "quant"
    store.is_available = available
    store.write_points = AsyncMock()
    store.query = AsyncMock(return_value=[])
    return store


# ----------------------------------------------------------------------
# detect_and_store — Schreibpfad
# ----------------------------------------------------------------------


async def test_detect_and_store_writes_points_with_measurement_and_tags() -> None:
    """Pro Anomalie ein write_points-Aufruf: anomalies-Measurement, Schema-Tags, ns-Timestamp."""
    store = make_mock_store()
    anomaly_store = AnomalyStore(store)

    result = await anomaly_store.detect_and_store(SYMBOL, TIMEFRAME, make_shock_candles())

    assert store.write_points.await_count == 1
    first = store.write_points.await_args_list[0]
    assert first.kwargs["measurement"] == quant_schema.ANOMALY_MEASUREMENT
    assert first.kwargs["tags"] == {
        "symbol": SYMBOL,
        "exchange": EXCHANGE,
        "anomaly_type": "price_shock",
    }
    # Schock-Kerze ist die letzte (Index 25) → ns-Timestamp aus ihrem ``time``.
    assert first.kwargs["timestamp"] == (BASE_TS + 25 * 60) * 1_000_000_000
    fields = first.kwargs["fields"]
    assert set(quant_schema.ANOMALY_FIELDS) <= set(fields)
    assert fields["anomaly_score"] == pytest.approx(6.0)  # flache Baseline → 2× Threshold
    assert fields["severity"] == pytest.approx(1.0)
    assert fields["feature"] == "close"
    assert fields["timeframe"] == TIMEFRAME
    assert fields["value"] == pytest.approx(math.log(2.0))
    assert fields["threshold"] == pytest.approx(3.0)
    assert isinstance(result, AnomalyResult)
    assert result.stored is True
    assert result.anomalies_found == 1
    assert result.symbol == SYMBOL
    assert result.timeframe == TIMEFRAME
    assert result.exchange == EXCHANGE


async def test_detect_and_store_unavailable_store_returns_stored_false() -> None:
    """is_available=False → kein write_points-Aufruf, stored=False, Erkennung läuft."""
    store = make_mock_store(available=False)
    anomaly_store = AnomalyStore(store)

    result = await anomaly_store.detect_and_store(SYMBOL, TIMEFRAME, make_shock_candles())

    assert result.stored is False
    store.write_points.assert_not_awaited()
    assert result.anomalies_found == 1
    assert result.symbol == SYMBOL
    assert result.timeframe == TIMEFRAME
    assert result.exchange == EXCHANGE


async def test_detect_and_store_no_anomalies_returns_zero() -> None:
    """Stabile Daten → keine Anomalien: anomalies_found=0, stored=False, kein Write."""
    store = make_mock_store()
    anomaly_store = AnomalyStore(store)

    result = await anomaly_store.detect_and_store(SYMBOL, TIMEFRAME, make_stable_candles())

    assert result.anomalies_found == 0
    assert result.stored is False
    store.write_points.assert_not_awaited()


async def test_detect_and_store_sets_symbol_on_anomalies() -> None:
    """Der symbol-Parameter wird auf jede Anomalie gesetzt (Tag + Feld-Kopplung)."""
    store = make_mock_store()
    anomaly_store = AnomalyStore(store)

    await anomaly_store.detect_and_store(SYMBOL, TIMEFRAME, make_shock_candles(),
                                         exchange="bybit")

    tags = store.write_points.await_args_list[0].kwargs["tags"]
    assert tags["symbol"] == SYMBOL
    assert tags["exchange"] == "bybit"


# ----------------------------------------------------------------------
# get_anomalies — Lesezugriff
# ----------------------------------------------------------------------


async def test_get_anomalies_builds_correct_flux_query() -> None:
    """Flux: Bucket, Range, Measurement- und Symbol-Filter, Pivot, Sort."""
    store = make_mock_store()
    anomaly_store = AnomalyStore(store)

    await anomaly_store.get_anomalies(
        SYMBOL,
        TIMEFRAME,
        start="2026-01-01T00:00:00Z",
        end="2026-01-02T00:00:00Z",
    )

    store.query.assert_awaited_once()
    flux = store.query.await_args.args[0]
    assert 'from(bucket: "quant")' in flux
    assert 'range(start: "2026-01-01T00:00:00Z", stop: "2026-01-02T00:00:00Z")' in flux
    assert 'r._measurement == "anomalies"' in flux
    assert 'r["symbol"] == "BTCUSDT"' in flux
    assert "anomaly_type" not in flux
    assert 'pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")' in flux
    assert 'sort(columns: ["_time"])' in flux


async def test_get_anomalies_with_anomaly_type_filter() -> None:
    """anomaly_type → zusätzlicher Tag-Filter, Default-Rückblick ohne start."""
    store = make_mock_store()
    anomaly_store = AnomalyStore(store)

    await anomaly_store.get_anomalies(SYMBOL, TIMEFRAME, anomaly_type="price_shock")

    store.query.assert_awaited_once()
    flux = store.query.await_args.args[0]
    assert 'r["anomaly_type"] == "price_shock"' in flux
    assert "now() - 30d" in flux
    assert 'r["symbol"] == "BTCUSDT"' in flux


async def test_get_anomalies_returns_empty_when_store_unavailable() -> None:
    """is_available=False → leere Liste, keine Query wird gestellt."""
    store = make_mock_store(available=False)
    anomaly_store = AnomalyStore(store)

    assert await anomaly_store.get_anomalies(SYMBOL, TIMEFRAME) == []
    store.query.assert_not_awaited()


async def test_get_anomalies_maps_records_and_filters_timeframe() -> None:
    """Pivotierte Records → Entry-Dicts; fremde Timeframes werden gefiltert."""
    record_1m = {
        "_time": datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        "symbol": SYMBOL,
        "exchange": EXCHANGE,
        "anomaly_type": "price_shock",
        "timeframe": "1m",
        "anomaly_score": 3.5,
        "severity": 0.7,
        "feature": "close",
        "value": 0.15,
        "threshold": 3.0,
    }
    record_5m = {
        "_time": datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        "symbol": SYMBOL,
        "exchange": EXCHANGE,
        "anomaly_type": "volume_spike",
        "timeframe": "5m",
        "anomaly_score": 4.0,
        "severity": 0.9,
        "feature": "volume",
    }
    store = make_mock_store()
    store.query = AsyncMock(return_value=[record_1m, record_5m])
    anomaly_store = AnomalyStore(store)

    result = await anomaly_store.get_anomalies(SYMBOL, TIMEFRAME)

    assert len(result) == 1
    entry = result[0]
    assert entry["time"] == "2026-01-01T00:00:00Z"
    assert entry["symbol"] == SYMBOL
    assert entry["exchange"] == EXCHANGE
    assert entry["anomaly_type"] == "price_shock"
    assert entry["timeframe"] == TIMEFRAME
    assert entry["anomaly_score"] == 3.5
    assert entry["severity"] == 0.7
    assert entry["feature"] == "close"
    assert entry["value"] == 0.15
    assert entry["threshold"] == 3.0
