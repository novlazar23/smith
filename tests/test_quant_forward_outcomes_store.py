"""Unit-Tests für das Forward-Outcome-Storage (Phase 6, P6-2).

Alle Tests nutzen einen gemockten ``InfluxDBStore`` (AsyncMock für die
async Store-API) — es werden keine echten InfluxDB-Verbindungen
aufgebaut. Die ``ForwardOutcomeEngine`` selbst läuft (deterministische
Stdlib-Berechnung); gemockt ist ausschließlich der Storage.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from trading_harness.quant.forward_outcomes import ForwardOutcomeEngine
from trading_harness.quant.forward_outcomes_store import (
    FORWARD_OUTCOMES_MEASUREMENT,
    ForwardOutcomeStore,
    ForwardOutcomeStoreResult,
)

pytestmark = pytest.mark.asyncio

SYMBOL = "BTCUSDT"
TIMEFRAME = "1m"
EXCHANGE = "binance"
BASE_TS = 1_767_225_600  # 2026-01-01T00:00:00Z in Epoch-Sekunden


def _candle_time(index: int) -> str:
    """UTC-ISO-Zeichenkette für die Kerze an Position *index* (1-Minuten-Abstand)."""
    moment = datetime.fromtimestamp(BASE_TS + index * 60, tz=UTC)
    return moment.isoformat().replace("+00:00", "Z")


def make_uptrend_candles(count: int = 100) -> list[dict]:
    """Synthetische OHLCV-Reihe im stetigen Aufwärtstrend."""
    return [
        {
            "time": _candle_time(i),
            "open": 100.0 + i * 0.5,
            "high": (100.0 + i * 0.5) * 1.01,
            "low": (100.0 + i * 0.5) * 0.99,
            "close": 100.0 + i * 0.5,
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
# compute_and_store — Schreibpfad
# ----------------------------------------------------------------------


async def test_compute_and_store_calls_engine_and_stores_per_horizon() -> None:
    """Engine-Lauf: pro Horizont mit Daten ein Punkt (Tags/Fields/ns-Timestamp)."""
    store = make_mock_store()
    engine = ForwardOutcomeEngine(horizons=[5, 10])
    outcome_store = ForwardOutcomeStore(store, engine=engine)

    result = await outcome_store.compute_and_store(
        SYMBOL, TIMEFRAME, make_uptrend_candles(), pattern_length=10, exchange=EXCHANGE
    )

    assert isinstance(result, ForwardOutcomeStoreResult)
    assert result.symbol == SYMBOL
    assert result.timeframe == TIMEFRAME
    assert result.pattern_length == 10
    assert result.horizons_computed == 2
    assert result.stored is True
    assert store.write_points.await_count == 2
    first = store.write_points.await_args_list[0]
    assert first.kwargs["measurement"] == FORWARD_OUTCOMES_MEASUREMENT
    assert first.kwargs["tags"] == {"symbol": SYMBOL, "exchange": EXCHANGE, "horizon": "5"}
    fields = first.kwargs["fields"]
    assert set(fields) == {
        "mean_return", "median_return", "hit_rate", "profit_factor", "expectancy",
        "std_return", "sample_size", "max_gain", "max_loss",
        "timeframe", "pattern_length",
    }
    assert fields["timeframe"] == TIMEFRAME
    assert fields["pattern_length"] == 10
    assert fields["mean_return"] > 0.0  # Aufwärtstrend → positive Forward Returns
    assert fields["sample_size"] > 0
    # ns-Timestamp aus der Zeit der letzten Kerze (Index 99).
    assert first.kwargs["timestamp"] == (BASE_TS + 99 * 60) * 1_000_000_000
    assert store.write_points.await_args_list[1].kwargs["tags"]["horizon"] == "10"


async def test_compute_and_store_insufficient_data() -> None:
    """Zu wenige Kerzen: keine Horizonte mit Daten → nichts geschrieben."""
    store = make_mock_store()
    engine = ForwardOutcomeEngine(horizons=[5, 10])
    outcome_store = ForwardOutcomeStore(store, engine=engine)

    result = await outcome_store.compute_and_store(
        SYMBOL, TIMEFRAME, make_uptrend_candles(count=3), pattern_length=10
    )

    assert result.horizons_computed == 0
    assert result.stored is False
    assert result.symbol == SYMBOL
    assert result.timeframe == TIMEFRAME
    store.write_points.assert_not_awaited()


async def test_compute_and_store_unavailable_store_returns_stored_false() -> None:
    """is_available=False: Berechnung läuft, aber kein Schreibversuch."""
    store = make_mock_store(available=False)
    engine = ForwardOutcomeEngine(horizons=[5, 10])
    outcome_store = ForwardOutcomeStore(store, engine=engine)

    result = await outcome_store.compute_and_store(
        SYMBOL, TIMEFRAME, make_uptrend_candles(), pattern_length=10
    )

    assert result.horizons_computed == 2
    assert result.stored is False
    store.write_points.assert_not_awaited()


async def test_compute_and_store_handles_engine_exception_gracefully() -> None:
    """Engine-Fehler (fehlendes close-Feld → KeyError): abgefangen, leeres Ergebnis."""
    store = make_mock_store()
    outcome_store = ForwardOutcomeStore(store, engine=ForwardOutcomeEngine(horizons=[5, 10]))

    candles = [{"time": _candle_time(i)} for i in range(20)]  # strukturell: kein "close"
    result = await outcome_store.compute_and_store(SYMBOL, TIMEFRAME, candles, pattern_length=10)

    assert isinstance(result, ForwardOutcomeStoreResult)
    assert result.horizons_computed == 0
    assert result.stored is False
    assert result.symbol == SYMBOL
    assert result.timeframe == TIMEFRAME
    assert result.pattern_length == 10
    store.write_points.assert_not_awaited()


# ----------------------------------------------------------------------
# get_outcomes — Lesezugriff
# ----------------------------------------------------------------------


async def test_get_outcomes_builds_flux_query_and_maps_records() -> None:
    """Flux: Bucket, Range, Measurement- und Symbol-Filter; Records → Entry-Dicts."""
    record_1m = {
        "_time": datetime(2026, 1, 1, 0, 5, tzinfo=UTC),
        "symbol": SYMBOL,
        "exchange": EXCHANGE,
        "horizon": "5",  # Tag → Zeichenkette im gepivotierten Record
        "timeframe": TIMEFRAME,
        "pattern_length": 10,
        "mean_return": 0.012,
        "median_return": 0.011,
        "hit_rate": 0.75,
        "profit_factor": 3.5,
        "expectancy": 0.012,
        "std_return": 0.02,
        "sample_size": 90,
        "max_gain": 0.05,
        "max_loss": -0.03,
    }
    store = make_mock_store()
    store.query = AsyncMock(return_value=[record_1m])
    outcome_store = ForwardOutcomeStore(store)

    result = await outcome_store.get_outcomes(
        SYMBOL,
        TIMEFRAME,
        start="2026-01-01T00:00:00Z",
        end="2026-01-02T00:00:00Z",
    )

    store.query.assert_awaited_once()
    flux = store.query.await_args.args[0]
    assert 'from(bucket: "quant")' in flux
    assert 'range(start: "2026-01-01T00:00:00Z", stop: "2026-01-02T00:00:00Z")' in flux
    assert f'r._measurement == "{FORWARD_OUTCOMES_MEASUREMENT}"' in flux
    assert 'r["symbol"] == "BTCUSDT"' in flux
    assert 'pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")' in flux
    assert 'sort(columns: ["_time"])' in flux

    assert len(result) == 1
    entry = result[0]
    assert entry["time"] == "2026-01-01T00:05:00Z"
    assert entry["symbol"] == SYMBOL
    assert entry["exchange"] == EXCHANGE
    assert entry["timeframe"] == TIMEFRAME
    assert entry["horizon"] == 5
    assert entry["pattern_length"] == 10
    assert entry["mean_return"] == pytest.approx(0.012)
    assert entry["hit_rate"] == pytest.approx(0.75)
    assert entry["profit_factor"] == pytest.approx(3.5)
    assert entry["sample_size"] == 90


async def test_get_outcomes_returns_empty_when_store_unavailable() -> None:
    """is_available=False → leere Liste, keine Query wird gestellt."""
    store = make_mock_store(available=False)
    outcome_store = ForwardOutcomeStore(store)

    assert await outcome_store.get_outcomes(SYMBOL, TIMEFRAME) == []
    store.query.assert_not_awaited()


async def test_get_outcomes_invalid_start_raises_value_error() -> None:
    """Ungültiger Start-Zeitstempel → ValueError (keine Query)."""
    store = make_mock_store()
    outcome_store = ForwardOutcomeStore(store)

    with pytest.raises(ValueError, match="invalid start timestamp"):
        await outcome_store.get_outcomes(SYMBOL, TIMEFRAME, start="not-a-date")
    store.query.assert_not_awaited()


async def test_get_outcomes_filters_foreign_timeframe() -> None:
    """Fremde Timeframes (Field-Filterung nach dem Pivot) werden ausgesortiert."""
    record_5m = {
        "_time": datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        "symbol": SYMBOL,
        "exchange": EXCHANGE,
        "horizon": "10",
        "timeframe": "5m",
        "mean_return": 0.01,
    }
    store = make_mock_store()
    store.query = AsyncMock(return_value=[record_5m])
    outcome_store = ForwardOutcomeStore(store)

    result = await outcome_store.get_outcomes(SYMBOL, TIMEFRAME)

    assert result == []
