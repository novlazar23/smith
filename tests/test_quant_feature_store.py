"""Unit-Tests für das Feature-Storage (Phase 2, P2-2).

Alle Tests nutzen einen gemockten ``InfluxDBStore`` (AsyncMock für die
async Store-API) — es werden keine echten InfluxDB-Verbindungen
aufgebaut. Die ``FeatureEngine`` selbst läuft (deterministische
Stdlib-Berechnung); gemockt ist ausschließlich der Storage.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from trading_harness.quant import schema as quant_schema
from trading_harness.quant.feature_store import FeatureResult, FeatureStore
from trading_harness.quant.features import FeatureEngine

pytestmark = pytest.mark.asyncio

SYMBOL = "BTCUSDT"
TIMEFRAME = "1m"
EXCHANGE = "binance"
BASE_TS = 1_700_000_000  # Epoch-Sekunden der ersten Kerze (1-Minuten-Abstand)

# Erwartete Gesamtzahl der Feature-Werte über 40 Kerzen mit Default-Engine:
# vwap 40 + rsi 26 (ab Kerze 15) + atr 26 (ab Kerze 15) + volatility 20 (ab 21)
# + macd×3 je 6 (ab Kerze 35 = slow+signal = 34+1) + bollinger×4 je 21 (ab 20) = 214
EXPECTED_TOTAL_FEATURES_40 = 214


def _candle_time(index: int) -> str:
    """UTC-ISO-Zeichenkette für die Kerze an Position *index*."""
    moment = datetime.fromtimestamp(BASE_TS + index * 60, tz=UTC)
    return moment.isoformat().replace("+00:00", "Z")


def make_candles(count: int = 40) -> list[dict]:
    """Synthetische OHLCV-Reihe mit ``time`` (alle Features ab Kerze 34 verfügbar)."""
    candles: list[dict] = []
    for i in range(count):
        base = 100.0 + i * 0.5
        candles.append(
            {
                "time": _candle_time(i),
                "open": base,
                "high": base + 2.0,
                "low": base - 2.0,
                "close": base + 1.0,
                "volume": 100.0 + i,
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
# compute_and_store — Schreibpfad
# ----------------------------------------------------------------------


async def test_compute_and_store_writes_points_with_measurement_and_tags() -> None:
    """Pro Kerze ein write_points-Aufruf: features-Measurement, Schema-Tags, ns-Timestamp."""
    store = make_mock_store()
    fs = FeatureStore(store)
    candles = make_candles()

    result = await fs.compute_and_store(SYMBOL, TIMEFRAME, candles)

    assert store.write_points.await_count == len(candles)
    first = store.write_points.await_args_list[0]
    assert first.kwargs["measurement"] == quant_schema.FEATURE_MEASUREMENT
    assert first.kwargs["tags"] == {
        "symbol": SYMBOL,
        "exchange": EXCHANGE,
        "feature_version": quant_schema.FEATURE_VERSION,
    }
    assert first.kwargs["timestamp"] == BASE_TS * 1_000_000_000
    assert isinstance(result, FeatureResult)
    assert result.stored is True
    assert result.symbol == SYMBOL
    assert result.timeframe == TIMEFRAME
    assert result.exchange == EXCHANGE
    assert result.feature_count == EXPECTED_TOTAL_FEATURES_40
    assert result.computation_time_ms >= 0


async def test_compute_and_store_unavailable_store_returns_stored_false() -> None:
    """is_available=False → kein write_points-Aufruf, stored=False, Berechnung läuft."""
    store = make_mock_store(available=False)
    fs = FeatureStore(store)

    result = await fs.compute_and_store(SYMBOL, TIMEFRAME, make_candles())

    assert result.stored is False
    store.write_points.assert_not_awaited()
    assert result.feature_count == EXPECTED_TOTAL_FEATURES_40
    assert result.computation_time_ms >= 0


async def test_compute_and_store_skips_candles_without_time() -> None:
    """Kerze ohne parsebares ``time`` → kein Punkt, alle übrigen werden geschrieben."""
    store = make_mock_store()
    fs = FeatureStore(store)
    candles = make_candles()
    del candles[5]["time"]

    result = await fs.compute_and_store(SYMBOL, TIMEFRAME, candles)

    assert store.write_points.await_count == len(candles) - 1
    assert result.stored is True


async def test_feature_values_extracted_from_compute_output() -> None:
    """Punkt-Fields tragen exakt die compute()-Werte (inkl. Guard-Klauseln)."""
    store = make_mock_store()
    engine = FeatureEngine()
    fs = FeatureStore(store, engine=engine)
    candles = make_candles()

    await fs.compute_and_store(SYMBOL, TIMEFRAME, candles)

    reference = engine.compute(candles)
    last_fields = store.write_points.await_args_list[-1].kwargs["fields"]
    assert last_fields["rsi"] == reference["rsi"]
    assert last_fields["macd_line"] == reference["macd"]["macd"]
    assert last_fields["signal_line"] == reference["macd"]["signal"]
    assert last_fields["histogram"] == reference["macd"]["histogram"]
    assert last_fields["bollinger_upper"] == reference["bollinger"]["upper"]
    assert last_fields["bollinger_middle"] == reference["bollinger"]["middle"]
    assert last_fields["bollinger_lower"] == reference["bollinger"]["lower"]
    assert last_fields["bollinger_bandwidth"] == reference["bollinger"]["bandwidth"]
    assert last_fields["atr"] == reference["atr"]
    assert last_fields["volatility"] == reference["volatility"]
    assert last_fields["vwap"] == reference["vwap"]
    assert last_fields["timeframe"] == TIMEFRAME
    assert last_fields["feature_count"] == 11
    assert set(quant_schema.FEATURE_META_FIELDS) <= set(last_fields)

    # Erste Kerze: nur VWAP berechenbar → None-Werte bleiben aus dem Punkt heraus.
    first_ref = engine.compute(candles[:1])
    first_fields = store.write_points.await_args_list[0].kwargs["fields"]
    assert "rsi" not in first_fields
    assert "macd_line" not in first_fields
    assert "bollinger_upper" not in first_fields
    assert "atr" not in first_fields
    assert "volatility" not in first_fields
    assert first_fields["vwap"] == first_ref["vwap"]
    assert first_fields["feature_count"] == 1


# ----------------------------------------------------------------------
# get_features — Lesezugriff
# ----------------------------------------------------------------------


async def test_get_features_builds_correct_flux_query() -> None:
    """Flux: Bucket, Range, Measurement-, Tag- und Field-Filter, Pivot, Sort."""
    store = make_mock_store()
    fs = FeatureStore(store)

    await fs.get_features(
        SYMBOL,
        TIMEFRAME,
        feature_names=["rsi", "macd_line"],
        start="2026-01-01T00:00:00Z",
        end="2026-01-02T00:00:00Z",
    )

    store.query.assert_awaited_once()
    flux = store.query.await_args.args[0]
    assert 'from(bucket: "quant")' in flux
    assert 'range(start: "2026-01-01T00:00:00Z", stop: "2026-01-02T00:00:00Z")' in flux
    assert 'r._measurement == "features"' in flux
    assert 'r["symbol"] == "BTCUSDT"' in flux
    assert 'r["exchange"] == "binance"' in flux
    assert f'r["feature_version"] == "{quant_schema.FEATURE_VERSION}"' in flux
    assert 'r._field in ["rsi", "macd_line", "timeframe"]' in flux
    assert 'pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")' in flux
    assert 'sort(columns: ["_time"])' in flux


async def test_get_features_returns_empty_when_store_unavailable() -> None:
    """is_available=False → leere Liste, keine Query wird gestellt."""
    store = make_mock_store(available=False)
    fs = FeatureStore(store)

    assert await fs.get_features(SYMBOL, TIMEFRAME) == []
    store.query.assert_not_awaited()


async def test_get_features_maps_records_and_filters_timeframe() -> None:
    """Pivotierte Records → Entry-Dicts; fremde Timeframes werden gefiltert."""
    record_1m = {
        "_time": datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        "symbol": SYMBOL,
        "exchange": EXCHANGE,
        "feature_version": quant_schema.FEATURE_VERSION,
        "timeframe": "1m",
        "rsi": 65.4,
        "feature_count": 11,
        "computation_time_ms": 0.5,
    }
    record_5m = {
        "_time": datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        "symbol": SYMBOL,
        "exchange": EXCHANGE,
        "feature_version": quant_schema.FEATURE_VERSION,
        "timeframe": "5m",
        "rsi": 40.0,
    }
    store = make_mock_store()
    store.query = AsyncMock(return_value=[record_1m, record_5m])
    fs = FeatureStore(store)

    result = await fs.get_features(SYMBOL, TIMEFRAME)

    assert len(result) == 1
    entry = result[0]
    assert entry["time"] == "2026-01-01T00:00:00Z"
    assert entry["symbol"] == SYMBOL
    assert entry["exchange"] == EXCHANGE
    assert entry["timeframe"] == TIMEFRAME
    assert entry["rsi"] == 65.4
    assert entry["feature_count"] == 11
    assert entry["computation_time_ms"] == 0.5


async def test_get_features_invalid_start_raises_value_error() -> None:
    """Ungültiger Start-Zeitstempel → ValueError, keine Query wird gestellt."""
    store = make_mock_store()
    fs = FeatureStore(store)

    with pytest.raises(ValueError, match="invalid start timestamp"):
        await fs.get_features(SYMBOL, TIMEFRAME, start="not-a-date")
    store.query.assert_not_awaited()
