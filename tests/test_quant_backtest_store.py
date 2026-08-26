"""Unit-Tests für ``BacktestStore`` (Phase 8, P8-2).

Mockt ``InfluxDBStore`` (keine echte DB-Verbindung) und prüft die beiden
``run_and_store``-Aufrufkontrakte (P8-4-Integration / P8-2-Skizze:
Engine-Ausführung; P8-3-API: fremdberechnetes ``BacktestResult``), die
Punkt-Schreibsemantik sowie ``get_results`` und den Alias
``get_backtests``.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from trading_harness.quant.backtest_store import (
    BACKTEST_STAT_FIELDS,
    BACKTESTS_MEASUREMENT,
    BacktestStore,
    BacktestStoreResult,
)
from trading_harness.quant.backtesting import BacktestEngine, Signal

pytestmark = pytest.mark.asyncio

SYMBOL = "BTCUSDT"
TIMEFRAME = "1m"
EXCHANGE = "binance"
START = datetime(2024, 1, 1, tzinfo=UTC)
CANDLE_COUNT = 60


def make_candles(count: int = CANDLE_COUNT, start_price: float = 100.0, step: float = 1.0) -> list[dict]:
    """Monoton aufsteigende OHLCV-Kerzen (SMA-Langs-Setup) mit ISO-Zeiten."""
    candles: list[dict] = []
    for i in range(count):
        price = start_price + step * i
        candles.append(
            {
                "time": (START + timedelta(minutes=i)).isoformat(),
                "open": price - 0.2,
                "high": price + 0.5,
                "low": price - 0.5,
                "close": price,
                "volume": 10.0,
            }
        )
    return candles


def make_store(available: bool = True) -> MagicMock:
    store = MagicMock()
    store.is_available = available
    store._bucket = "quant_test"
    store.write_points = AsyncMock()
    store.query = AsyncMock(return_value=[])
    return store


def make_backtest_store(available: bool = True) -> tuple[BacktestStore, MagicMock]:
    store = make_store(available)
    return BacktestStore(store), store


def sma_strategy(fast: int = 3, slow: int = 8) -> Callable[[list[dict], int], Signal]:
    return BacktestEngine().simple_moving_average_strategy(fast, slow)


def hold_strategy(candles: list[dict], index: int) -> Signal:
    return Signal.HOLD


def exploding_strategy(candles: list[dict], index: int) -> Signal:
    raise RuntimeError("boom")


async def test_run_and_store_writes_backtest_point() -> None:
    """(symbol, timeframe, candles, strategy, exchange) → Punkt in ``backtests``."""
    backtest_store, store = make_backtest_store()
    candles = make_candles()

    store_result = await backtest_store.run_and_store(SYMBOL, TIMEFRAME, candles, sma_strategy(), EXCHANGE)

    assert store_result.stored is True
    assert store_result.trades_stored >= 1
    assert store_result.result is not None
    assert store_result.result.total_trades == store_result.trades_stored
    assert store_result.result.total_pnl > 0.0
    store.write_points.assert_awaited_once()
    kwargs = store.write_points.await_args.kwargs
    assert kwargs["measurement"] == BACKTESTS_MEASUREMENT
    assert kwargs["tags"] == {"symbol": SYMBOL, "exchange": EXCHANGE}
    fields = kwargs["fields"]
    assert fields["total_trades"] == store_result.result.total_trades
    assert fields["total_pnl"] == pytest.approx(store_result.result.total_pnl)
    for name in BACKTEST_STAT_FIELDS:
        assert name in fields
    assert fields["timeframe"] == TIMEFRAME
    last_time = START + timedelta(minutes=CANDLE_COUNT - 1)
    assert kwargs["timestamp"] == int(last_time.timestamp()) * 1_000_000_000


async def test_run_and_store_result_fields_with_default_exchange() -> None:
    """Der Rückgabewert trägt symbol/timeframe/trades_stored/stored/result; Exchange-Default."""
    backtest_store, store = make_backtest_store()

    store_result = await backtest_store.run_and_store(SYMBOL, TIMEFRAME, make_candles(), sma_strategy())

    assert isinstance(store_result, BacktestStoreResult)
    assert store_result.symbol == SYMBOL
    assert store_result.timeframe == TIMEFRAME
    assert store_result.stored is True
    assert store_result.trades_stored == store_result.result.total_trades
    assert store.write_points.await_args.kwargs["tags"] == {"symbol": SYMBOL, "exchange": "binance"}


async def test_run_and_store_insufficient_data_writes_nothing() -> None:
    """Eine Kerze → leeres Engine-Ergebnis, kein Punkt."""
    backtest_store, store = make_backtest_store()

    store_result = await backtest_store.run_and_store(SYMBOL, TIMEFRAME, make_candles(1), sma_strategy())

    assert store_result.stored is False
    assert store_result.trades_stored == 0
    assert store_result.result is not None
    assert store_result.result.total_trades == 0
    store.write_points.assert_not_awaited()


async def test_run_and_store_no_trades_writes_nothing() -> None:
    """HOLD-Strategie (keine Trades) → kein Punkt (keine Handelsinformation)."""
    backtest_store, store = make_backtest_store()

    store_result = await backtest_store.run_and_store(
        SYMBOL, TIMEFRAME, make_candles(), hold_strategy, EXCHANGE
    )

    assert store_result.stored is False
    assert store_result.trades_stored == 0
    store.write_points.assert_not_awaited()


async def test_run_and_store_handles_engine_exceptions_gracefully(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Engine-Fehler → geloggt, leeres BacktestResult, stored=False, keine Exception."""
    backtest_store, store = make_backtest_store()

    store_result = await backtest_store.run_and_store(SYMBOL, TIMEFRAME, make_candles(), exploding_strategy)

    assert store_result.stored is False
    assert store_result.trades_stored == 0
    assert store_result.result is not None
    assert store_result.result.total_trades == 0
    assert store_result.result.symbol == SYMBOL
    assert store_result.result.timeframe == TIMEFRAME
    store.write_points.assert_not_awaited()
    assert "Backtest run failed" in caplog.text


async def test_run_and_store_unavailable_store_does_not_write() -> None:
    """Store nicht verfügbar → kein Schreibversuch, Backtest lief trotzdem."""
    backtest_store, store = make_backtest_store(available=False)

    store_result = await backtest_store.run_and_store(
        SYMBOL, TIMEFRAME, make_candles(), sma_strategy(), EXCHANGE
    )

    assert store_result.stored is False
    assert store_result.trades_stored >= 1
    store.write_points.assert_not_awaited()


async def test_run_and_store_precomputed_result_contract() -> None:
    """P8-3-Kontrakt: (symbol, timeframe, BacktestResult, exchange=…) → nur persistieren."""
    backtest_store, store = make_backtest_store()
    result = BacktestEngine().run(make_candles(), sma_strategy(), symbol=SYMBOL, timeframe=TIMEFRAME)

    store_result = await backtest_store.run_and_store(SYMBOL, TIMEFRAME, result, exchange=EXCHANGE)

    assert store_result.stored is True
    assert store_result.result is result
    assert store_result.trades_stored == result.total_trades
    store.write_points.assert_awaited_once()
    assert store.write_points.await_args.kwargs["measurement"] == BACKTESTS_MEASUREMENT
    assert store.write_points.await_args.kwargs["tags"] == {"symbol": SYMBOL, "exchange": EXCHANGE}


async def test_run_and_store_precomputed_zero_trades_writes_nothing() -> None:
    """Fremdberechnetes Ergebnis ohne Trades → kein Punkt, stored=False."""
    backtest_store, store = make_backtest_store()
    result = BacktestEngine().run(make_candles(1), sma_strategy(), symbol=SYMBOL, timeframe=TIMEFRAME)

    store_result = await backtest_store.run_and_store(SYMBOL, TIMEFRAME, result)

    assert result.total_trades == 0
    assert store_result.stored is False
    assert store_result.trades_stored == 0
    assert store_result.result is result
    store.write_points.assert_not_awaited()


async def test_run_and_store_custom_exchange_in_tags() -> None:
    backtest_store, store = make_backtest_store()

    await backtest_store.run_and_store(SYMBOL, TIMEFRAME, make_candles(), sma_strategy(), "bybit")

    assert store.write_points.await_args.kwargs["tags"] == {"symbol": SYMBOL, "exchange": "bybit"}


async def test_run_and_store_rejects_unknown_kwargs() -> None:
    backtest_store, _ = make_backtest_store()

    with pytest.raises(TypeError, match="unexpected keyword arguments"):
        await backtest_store.run_and_store(SYMBOL, TIMEFRAME, make_candles(), sma_strategy(), foo="bar")


async def test_run_and_store_rejects_too_few_positional_args() -> None:
    backtest_store, _ = make_backtest_store()

    with pytest.raises(TypeError, match="requires \\(symbol, timeframe"):
        await backtest_store.run_and_store(SYMBOL, TIMEFRAME, make_candles())


async def test_run_and_store_rejects_non_string_symbol() -> None:
    """Kerzen-First-Aufruf (kein Consumer) wird abgelehnt: symbol muss String sein."""
    backtest_store, _ = make_backtest_store()

    with pytest.raises(TypeError, match="symbol and timeframe must be strings"):
        await backtest_store.run_and_store(make_candles(), sma_strategy(), SYMBOL, TIMEFRAME)


async def test_get_results_builds_correct_flux_query() -> None:
    backtest_store, store = make_backtest_store()
    store.query.return_value = []

    await backtest_store.get_results(SYMBOL, TIMEFRAME, "2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z")

    store.query.assert_awaited_once()
    flux = store.query.await_args.args[0]
    assert 'from(bucket: "quant_test")' in flux
    assert 'range(start: "2024-01-01T00:00:00Z", stop: "2024-01-02T00:00:00Z")' in flux
    assert f'r._measurement == "{BACKTESTS_MEASUREMENT}"' in flux
    assert f'r["symbol"] == "{SYMBOL}"' in flux
    assert 'pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")' in flux
    assert 'sort(columns: ["_time"])' in flux


async def test_get_results_defaults_to_lookback_without_start() -> None:
    backtest_store, store = make_backtest_store()
    store.query.return_value = []

    await backtest_store.get_results(SYMBOL, TIMEFRAME)

    flux = store.query.await_args.args[0]
    assert "now() - 30d" in flux
    assert "stop:" not in flux


async def test_get_results_returns_results() -> None:
    """Pivot-Records → Einträge mit UTC-ISO-Zeit, int/float-Mapping; Fremd-Timeframes gefiltert."""
    backtest_store, store = make_backtest_store()
    last_time = START + timedelta(minutes=CANDLE_COUNT - 1)
    store.query.return_value = [
        {
            "_time": int(last_time.timestamp()) * 1_000_000_000,
            "symbol": SYMBOL,
            "exchange": EXCHANGE,
            "timeframe": TIMEFRAME,
            "total_trades": 2,
            "winning_trades": 1,
            "losing_trades": 1,
            "win_rate": 0.5,
            "total_pnl": 123.45,
            "total_pnl_pct": 1.23,
            "max_drawdown": 0.4,
            "sharpe_ratio": 1.1,
            "avg_trade_pnl": 61.72,
            "profit_factor": 1.5,
        },
        {
            "_time": int(last_time.timestamp()) * 1_000_000_000,
            "symbol": SYMBOL,
            "exchange": EXCHANGE,
            "timeframe": "5m",
            "total_trades": 9,
        },
    ]

    results = await backtest_store.get_results(SYMBOL, TIMEFRAME)

    assert len(results) == 1
    entry = results[0]
    assert entry["time"] == last_time.isoformat().replace("+00:00", "Z")
    assert entry["symbol"] == SYMBOL
    assert entry["exchange"] == EXCHANGE
    assert entry["timeframe"] == TIMEFRAME
    assert entry["total_trades"] == 2
    assert isinstance(entry["total_trades"], int)
    assert entry["winning_trades"] == 1
    assert isinstance(entry["winning_trades"], int)
    assert entry["win_rate"] == 0.5
    assert entry["total_pnl"] == 123.45


async def test_get_results_unavailable_returns_empty() -> None:
    backtest_store, store = make_backtest_store(available=False)

    assert await backtest_store.get_results(SYMBOL, TIMEFRAME) == []
    store.query.assert_not_awaited()


async def test_get_backtests_aliases_get_results() -> None:
    """P8-3-Endpunkt-Vertrag: ``get_backtests`` ruft dieselbe Query ab."""
    backtest_store, store = make_backtest_store()
    store.query.return_value = []

    await backtest_store.get_backtests(SYMBOL, TIMEFRAME, "2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z")

    store.query.assert_awaited_once()
    flux = store.query.await_args.args[0]
    assert 'range(start: "2024-01-01T00:00:00Z", stop: "2024-01-02T00:00:00Z")' in flux


async def test_get_results_invalid_start_timestamp_raises() -> None:
    backtest_store, store = make_backtest_store()

    with pytest.raises(ValueError, match="invalid start timestamp"):
        await backtest_store.get_results(SYMBOL, TIMEFRAME, "not-a-timestamp")
    store.query.assert_not_awaited()
