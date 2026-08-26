"""Backtest-Integration (Quant-Plattform, Phase 8, P8-4).

End-to-End-Kette Engine → Speicherung → Rücklese mit ausschließlich Mocks:
``BacktestEngine`` (P8-1) → ``BacktestStore`` (P8-2) → ``InfluxDBStore``
(P1-4, gemockt). Kein Netzwerk, kein Docker, keine echte InfluxDB.

Der BacktestStore wird gegen die echte Implementierung (``quant/
backtest_store.py``) getestet; nur der InfluxDB-Store wird gemockt.
Geprüfte Kontrakte:

- ``run_and_store(symbol, timeframe, candles, strategy, exchange=...)``
  führt den Backtest mit dem echten ``BacktestEngine`` aus und schreibt bei
  verfügbarem Store genau einen Punkt im ``backtests``-Measurement
  (Tags ``symbol``/``exchange``, Statistik-Fields + Kontext ``timeframe``).
- ``get_results`` liest die Punkte per Flux-Query zurück (Roundtrip).
- Store nicht verfügbar / Lauf ohne Trades → kein Write (``stored=False``),
  der Backtest läuft trotzdem.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from trading_harness.quant.backtest_store import (
    BACKTEST_STAT_FIELDS,
    BACKTESTS_MEASUREMENT,
    BacktestStore,
)
from trading_harness.quant.backtesting import BacktestEngine, BacktestResult, Signal

SYMBOL = "BTCUSDT"
EXCHANGE = "binance"
TIMEFRAME = "1h"
_NS = 1_000_000_000


# ----------------------------------------------------------------------
# Test-Helfer
# ----------------------------------------------------------------------


def make_candles(closes: list[float]) -> list[dict]:
    """OHLCV-Kerzen aus einer Close-Reihe (open = Vor-Close, deterministisch)."""
    candles: list[dict] = []
    previous = closes[0] if closes else 100.0
    for index, close in enumerate(closes):
        candles.append(
            {
                "time": f"2026-01-01T{index // 60:02d}:{index % 60:02d}:00Z",
                "open": previous,
                "high": max(previous, close),
                "low": min(previous, close),
                "close": close,
                "volume": 1000.0,
            }
        )
        previous = close
    return candles


def uptrend_candles(count: int = 60) -> list[dict]:
    """60 saubere Aufwärtskerzen (close steigt linear um 0.5 pro Kerze ab 100)."""
    return make_candles([100.0 + 0.5 * i for i in range(count)])


def make_mock_store(available: bool = True) -> MagicMock:
    """InfluxDBStore-Mock: Verfügbarkeits-Flag + async write/query, keine echte Verbindung."""
    store = MagicMock()
    store._bucket = "quant"
    store.is_available = available
    store.write_points = AsyncMock()
    store.query = AsyncMock(return_value=[])
    return store


def point_of(call: Any) -> dict[str, Any]:
    """Normalisiert einen ``write_points``-Aufruf (args + kwargs) zu einem Punkt-Dict."""
    names = ("measurement", "tags", "fields", "timestamp")
    return dict(zip(names, call.args)) | call.kwargs


def sma_strategy(fast: int = 5, slow: int = 20) -> Callable[[list[dict], int], Signal]:
    """SMA-Crossover-Strategie der BacktestEngine (Standardparameter)."""
    return BacktestEngine().simple_moving_average_strategy(fast_period=fast, slow_period=slow)


# ----------------------------------------------------------------------
# 1. Engine: Uptrend → positives PnL
# ----------------------------------------------------------------------


def test_engine_uptrend_positive_pnl() -> None:
    """SMA-Strategie im sauberen Uptrend → Long-Trades mit positivem Gesamt-PnL."""
    engine = BacktestEngine(initial_capital=10_000)
    result = engine.run(uptrend_candles(), sma_strategy(), SYMBOL, TIMEFRAME)

    assert isinstance(result, BacktestResult)
    assert result.total_trades >= 1
    assert result.total_pnl > 0.0
    assert result.total_pnl_pct > 0.0
    assert result.winning_trades >= 1
    assert all(trade.direction == "long" for trade in result.trades)
    # Equity startet beim Startkapital und endet über dem Startkapital.
    assert result.equity_curve[0] == 10_000.0
    assert result.equity_curve[-1] > 10_000.0


# ----------------------------------------------------------------------
# 2. Engine → Store → (mockte) InfluxDB: Roundtrip
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_engine_to_store_roundtrip() -> None:
    """Uptrend → Backtest läuft → Punkt wird geschrieben und via get_results gelesen."""
    store = make_mock_store()
    engine = BacktestEngine(initial_capital=10_000)
    backtest_store = BacktestStore(store, engine=engine)
    candles = uptrend_candles()
    strategy = sma_strategy()
    expected = engine.run(candles, strategy, SYMBOL, TIMEFRAME)
    assert expected.total_pnl > 0.0

    result = await backtest_store.run_and_store(SYMBOL, TIMEFRAME, candles, strategy, EXCHANGE)

    assert result.stored is True
    assert result.trades_stored == expected.total_trades >= 1
    store.write_points.assert_awaited_once()
    point = point_of(store.write_points.await_args)
    assert point["measurement"] == BACKTESTS_MEASUREMENT
    assert point["tags"]["symbol"] == SYMBOL
    assert point["tags"]["exchange"] == EXCHANGE
    # Alle Schema-Statistik-Fields vorhanden; Gespeichertes = Engine-Output, 1:1.
    assert set(BACKTEST_STAT_FIELDS) <= set(point["fields"])
    assert point["fields"]["total_trades"] == expected.total_trades
    assert point["fields"]["total_pnl"] == pytest.approx(expected.total_pnl)
    assert point["fields"]["timeframe"] == TIMEFRAME
    # Timestamp der letzten Kerze (Nanosekunden, UTC).
    assert point["timestamp"] == int(datetime.fromisoformat(candles[-1]["time"]).timestamp()) * _NS

    # Rücklese: die gemockte Flux-Antwort liefert genau den geschriebenen Punkt.
    record: dict[str, Any] = {
        "_time": point["timestamp"],
        "symbol": point["tags"]["symbol"],
        "exchange": point["tags"]["exchange"],
        **point["fields"],
    }
    store.query = AsyncMock(return_value=[record])
    entries = await backtest_store.get_results(SYMBOL, TIMEFRAME)

    assert len(entries) == 1
    entry = entries[0]
    assert entry["time"] == "2026-01-01T00:59:00Z"
    assert entry["symbol"] == SYMBOL
    assert entry["exchange"] == EXCHANGE
    assert entry["timeframe"] == TIMEFRAME
    assert entry["total_trades"] == expected.total_trades
    assert entry["total_pnl"] == pytest.approx(expected.total_pnl)
    assert entry["total_pnl"] > 0.0


@pytest.mark.asyncio
async def test_engine_to_store_unavailable() -> None:
    """Store nicht verfügbar (is_available=False) → kein Write, Backtest läuft trotzdem."""
    store = make_mock_store(available=False)
    backtest_store = BacktestStore(store)
    candles = uptrend_candles()

    result = await backtest_store.run_and_store(SYMBOL, TIMEFRAME, candles, sma_strategy(), EXCHANGE)

    assert result.stored is False
    assert result.trades_stored >= 1  # Engine-Lauf bleibt unbeeinträchtigt
    store.write_points.assert_not_awaited()
    store.query.assert_not_awaited()


@pytest.mark.asyncio
async def test_engine_to_store_no_trades_no_point() -> None:
    """Lauf ohne Trades (eine Kerze) → keine Handelsinformation → kein Punkt."""
    store = make_mock_store()
    backtest_store = BacktestStore(store)

    result = await backtest_store.run_and_store(
        SYMBOL, TIMEFRAME, make_candles([100.0]), sma_strategy(), EXCHANGE
    )

    assert result.stored is False
    assert result.trades_stored == 0
    store.write_points.assert_not_awaited()


# ----------------------------------------------------------------------
# 3. SMA-Strategie: Signal-Generierung
# ----------------------------------------------------------------------


def test_sma_strategy_generates_signals() -> None:
    """SMA-Crossover erzeugt HOLD (vor Fenstern), LONG (Uptrend) und SHORT (nach Umkehr)."""
    strategy = sma_strategy(fast=5, slow=20)
    # 40 Kerzen Aufwärts (100 → 119.5), dann 40 Kerzen Abwärts (119.5 → 99.5).
    closes = [100.0 + 0.5 * i for i in range(40)] + [119.5 - 0.5 * i for i in range(40)]
    candles = make_candles(closes)

    assert strategy(candles, 10) == Signal.HOLD  # zu früh: langsames Fenster nicht voll
    assert strategy(candles, 30) == Signal.LONG  # stabiler Uptrend: fast > slow
    assert strategy(candles, 55) == Signal.SHORT  # nach Umkehr: fast < slow

    signals = {strategy(candles, i) for i in range(1, len(candles))}
    assert Signal.LONG in signals
    assert Signal.SHORT in signals
    assert Signal.HOLD in signals


# ----------------------------------------------------------------------
# 4. Unzureichende Daten → leeres Ergebnis
# ----------------------------------------------------------------------


def test_insufficient_data_returns_empty() -> None:
    """Keine oder eine einzige Kerze → kein Trade, PnL 0, Equity auf Startkapital."""
    engine = BacktestEngine(initial_capital=10_000)
    strategy = sma_strategy()

    empty = engine.run([], strategy, SYMBOL, TIMEFRAME)
    assert empty.total_trades == 0
    assert empty.total_pnl == 0.0
    assert empty.win_rate == 0.0
    assert empty.trades == []
    assert empty.equity_curve == [10_000.0]

    single = engine.run(make_candles([100.0]), strategy, SYMBOL, TIMEFRAME)
    assert single.total_trades == 0
    assert single.total_pnl == 0.0
    assert single.trades == []
    assert single.equity_curve == [10_000.0]
