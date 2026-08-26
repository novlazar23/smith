"""Unit-Tests für das Regime-Storage (Phase 4, P4-2).

Alle Tests nutzen einen gemockten ``InfluxDBStore`` (AsyncMock für die
async Store-API) — es werden keine echten InfluxDB-Verbindungen
aufgebaut. Der ``RegimeDetector`` selbst läuft (deterministische
Stdlib-Berechnung); gemockt ist ausschließlich der Storage.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from trading_harness.quant import schema as quant_schema
from trading_harness.quant.regime_detection import REGIME_NAMES, RegimeDetector
from trading_harness.quant.regime_store import RegimeStore, RegimeStoreResult

pytestmark = pytest.mark.asyncio

SYMBOL = "BTCUSDT"
TIMEFRAME = "1m"
EXCHANGE = "binance"
BASE_TS = 1_700_000_000  # Epoch-Sekunden der ersten Kerze (1-Minuten-Abstand)


def _candle_time(index: int) -> str:
    """UTC-ISO-Zeichenkette für die Kerze an Position *index*."""
    moment = datetime.fromtimestamp(BASE_TS + index * 60, tz=UTC)
    return moment.isoformat().replace("+00:00", "Z")


def make_uptrend_candles(count: int = 60) -> list[dict]:
    """Synthetische OHLCV-Steigung (deterministischer Aufwärtstrend)."""
    candles: list[dict] = []
    for i in range(count):
        price = 100.0 + 0.5 * i
        candles.append(
            {
                "time": _candle_time(i),
                "open": price - 0.1,
                "high": price + 0.2,
                "low": price - 0.2,
                "close": price,
                "volume": 1000.0,
            }
        )
    return candles


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


async def test_detect_and_store_writes_points_correctly() -> None:
    """Ein write_points-Aufruf: regime-Measurement, Schema-Tags, Schema-Fields, ns-Timestamp."""
    store = make_mock_store()
    regime_store = RegimeStore(store)
    candles = make_uptrend_candles()
    expected = RegimeDetector().detect(candles)

    result = await regime_store.detect_and_store(SYMBOL, TIMEFRAME, candles)

    assert store.write_points.await_count == 1
    first = store.write_points.await_args_list[0]
    assert first.kwargs["measurement"] == quant_schema.REGIME_MEASUREMENT
    assert first.kwargs["tags"] == {"symbol": SYMBOL, "exchange": EXCHANGE}
    # Zeitstempel = Zeit der letzten Kerze (1-Minuten-Abstand) in ns.
    assert first.kwargs["timestamp"] == (BASE_TS + (len(candles) - 1) * 60) * 1_000_000_000
    fields = first.kwargs["fields"]
    assert set(quant_schema.REGIME_FIELDS) <= set(fields)
    assert fields["regime_name"] == expected.regime
    assert fields["regime_confidence"] == pytest.approx(expected.confidence)
    assert fields["regime_duration"] == expected.duration
    assert fields["timeframe"] == TIMEFRAME
    assert isinstance(result, RegimeStoreResult)
    assert result.stored is True
    assert result.regime == expected.regime
    assert result.confidence == pytest.approx(expected.confidence)
    assert result.symbol == SYMBOL
    assert result.timeframe == TIMEFRAME
    assert result.exchange == EXCHANGE


async def test_detect_and_store_unavailable_store_returns_stored_false() -> None:
    """is_available=False → kein write_points-Aufruf, stored=False, Erkennung läuft."""
    store = make_mock_store(available=False)
    regime_store = RegimeStore(store)
    candles = make_uptrend_candles()

    result = await regime_store.detect_and_store(SYMBOL, TIMEFRAME, candles)

    assert result.stored is False
    store.write_points.assert_not_awaited()
    # Die Erkennung läuft trotzdem (Determinismus): Regime + Confidence gesetzt.
    assert result.regime in REGIME_NAMES
    assert 0.0 <= result.confidence <= 1.0
    assert result.regime == RegimeDetector().detect(candles).regime
    assert result.symbol == SYMBOL
    assert result.timeframe == TIMEFRAME
    assert result.exchange == EXCHANGE


async def test_detect_and_store_no_candles_returns_range_regime() -> None:
    """Leere Kerzenreihe → range-Regime (Detector-Fallback), Punkt wird geschrieben."""
    store = make_mock_store()
    regime_store = RegimeStore(store)

    result = await regime_store.detect_and_store(SYMBOL, TIMEFRAME, [])

    assert result.regime == "range"
    assert result.confidence == 0.5
    assert result.stored is True
    first = store.write_points.await_args_list[0]
    assert first.kwargs["measurement"] == quant_schema.REGIME_MEASUREMENT
    fields = first.kwargs["fields"]
    assert fields["regime_name"] == "range"
    assert fields["regime_confidence"] == 0.5
    assert fields["regime_duration"] == 0
    # Ohne Kerzen: Zeitstempel = aktueller UTC-Zeitpunkt (ns, plausibel positiv).
    assert first.kwargs["timestamp"] > 0


# ----------------------------------------------------------------------
# get_regime — Lesezugriff
# ----------------------------------------------------------------------


async def test_get_regime_builds_correct_flux_query() -> None:
    """Flux: Bucket, Range, Measurement- und Symbol-Filter, Pivot, Sort."""
    store = make_mock_store()
    regime_store = RegimeStore(store)

    await regime_store.get_regime(
        SYMBOL,
        TIMEFRAME,
        start="2026-01-01T00:00:00Z",
        end="2026-01-02T00:00:00Z",
    )

    store.query.assert_awaited_once()
    flux = store.query.await_args.args[0]
    assert 'from(bucket: "quant")' in flux
    assert 'range(start: "2026-01-01T00:00:00Z", stop: "2026-01-02T00:00:00Z")' in flux
    assert 'r._measurement == "regime"' in flux
    assert 'r["symbol"] == "BTCUSDT"' in flux
    assert 'pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")' in flux
    assert 'sort(columns: ["_time"])' in flux


async def test_get_regime_without_start_uses_default_lookback() -> None:
    """Ohne start: Default-Rückblick 30 Tage, kein stop im Range."""
    store = make_mock_store()
    regime_store = RegimeStore(store)

    await regime_store.get_regime(SYMBOL, TIMEFRAME)

    flux = store.query.await_args.args[0]
    assert "now() - 30d" in flux
    assert "stop:" not in flux


async def test_get_regime_returns_empty_when_store_unavailable() -> None:
    """is_available=False → leere Liste, keine Query wird gestellt."""
    store = make_mock_store(available=False)
    regime_store = RegimeStore(store)

    assert await regime_store.get_regime(SYMBOL, TIMEFRAME) == []
    store.query.assert_not_awaited()


async def test_get_regime_maps_records_and_filters_timeframe() -> None:
    """Pivotierte Records → Entry-Dicts; fremde Timeframes werden gefiltert."""
    record_1m = {
        "_time": datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        "symbol": SYMBOL,
        "exchange": EXCHANGE,
        "timeframe": "1m",
        "regime_name": "strong_bull",
        "regime_confidence": 0.85,
        "regime_duration": 5,
    }
    record_5m = {
        "_time": datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        "symbol": SYMBOL,
        "exchange": EXCHANGE,
        "timeframe": "5m",
        "regime_name": "range",
        "regime_confidence": 0.6,
        "regime_duration": 2,
    }
    store = make_mock_store()
    store.query = AsyncMock(return_value=[record_1m, record_5m])
    regime_store = RegimeStore(store)

    result = await regime_store.get_regime(SYMBOL, TIMEFRAME)

    assert len(result) == 1
    entry = result[0]
    assert entry["time"] == "2026-01-01T00:00:00Z"
    assert entry["symbol"] == SYMBOL
    assert entry["exchange"] == EXCHANGE
    assert entry["timeframe"] == TIMEFRAME
    assert entry["regime"] == "strong_bull"
    assert entry["confidence"] == 0.85
    assert entry["duration"] == 5


async def test_get_regime_invalid_start_raises_value_error() -> None:
    """Ungültiger start-Zeitstempel → ValueError, keine Query wird gestellt."""
    store = make_mock_store()
    regime_store = RegimeStore(store)

    with pytest.raises(ValueError):
        await regime_store.get_regime(SYMBOL, TIMEFRAME, start="not-a-date")
    store.query.assert_not_awaited()
