"""Tests für den optionalen OHLCV-Ingestion-Hook im Shadow-Trading-Loop (P1-7).

Der Hook ist rein additiv und optional: ohne ``ohlcv_ingestion`` ändert sich
das Loop-Verhalten nicht; mit ``ohlcv_ingestion`` wird pro Symbol und
Iteration ``ingest_candles`` aufgerufen, wenn das Ticker-Datum einen
``ohlcv``-Dict enthält. Fehler der Ingestion sind nicht kritisch und müssen
den Entscheidungs-Flow niemals unterbrechen.

Alle ShadowTradingLoop-Abhängigkeiten sind MagicMock — der Loop wird nicht
gestartet; nur ``_decide_symbol`` (eine Iteration pro Symbol) wird ausgeführt.
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from trading_harness.quant.ohlcv_ingestion import OHLCVIngestion
from trading_harness.services.shadow_trading_loop import ShadowTradingLoop

T0 = datetime(2026, 8, 25, 12, 0, 0, tzinfo=UTC)
SYMBOL = "BTCUSDT"

OHLCV_TICKER: dict[str, object] = {
    "price": 50000.0,
    "ohlcv": {
        "open": 49900.0,
        "high": 50100.0,
        "low": 49800.0,
        "close": 50000.0,
        "volume": 12.5,
    },
}


def make_loop(ohlcv_ingestion: object | None = ...) -> ShadowTradingLoop:
    """Erzeugt einen ShadowTradingLoop mit durchgehend gemockten Abhängigkeiten."""
    agent_source = MagicMock()
    # Leere Agentenliste -> deterministischer NO_TRADE-Pfad ohne Analyse.
    agent_source.list_all.return_value = []
    kwargs: dict[str, OHLCVIngestion | None] = {}
    if ohlcv_ingestion is not ...:
        kwargs["ohlcv_ingestion"] = cast(OHLCVIngestion | None, ohlcv_ingestion)
    return ShadowTradingLoop(
        market_data=MagicMock(),
        backend=MagicMock(),
        agent_runtime=MagicMock(),
        trading_run_service=MagicMock(),
        snapshot_store=MagicMock(),
        risk_engine=MagicMock(),
        kill_switch=MagicMock(),
        agent_source=agent_source,
        settings=MagicMock(),
        state_store=MagicMock(),
        clock=lambda: T0,
        **kwargs,
    )


def run_decide_symbol(loop: ShadowTradingLoop, ticker_data: dict[str, object]):
    """Führt genau eine _decide_symbol-Iteration aus (kein Loop-Start)."""
    return loop._decide_symbol(
        SYMBOL,
        iteration=1,
        last=50000.0,
        ticker_data=dict(ticker_data),
        session_id="sess-test",
        run_id_holder=[None],
        config_version="test-config",
    )


def state_store_of(loop: ShadowTradingLoop) -> MagicMock:
    return cast(MagicMock, loop._state_store)


def test_init_signature_has_optional_ohlcv_ingestion_defaulting_to_none() -> None:
    param = inspect.signature(ShadowTradingLoop.__init__).parameters["ohlcv_ingestion"]
    assert param.default is None
    assert param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD


def test_init_accepts_ohlcv_ingestion_none_default() -> None:
    loop = make_loop()
    assert loop._ohlcv_ingestion is None


def test_init_accepts_ohlcv_ingestion_instance() -> None:
    # Echte OHLCVIngestion-Instanz (Store gemockt, keine Verbindung).
    ingestion = OHLCVIngestion(MagicMock())
    loop = make_loop(ohlcv_ingestion=ingestion)
    assert loop._ohlcv_ingestion is ingestion


@pytest.mark.asyncio
async def test_hook_calls_ingest_candles_when_data_has_ohlcv() -> None:
    ingestion = MagicMock()
    ingestion.ingest_candles = AsyncMock()
    loop = make_loop(ohlcv_ingestion=ingestion)

    entry, decided, _ = await run_decide_symbol(loop, OHLCV_TICKER)

    ingestion.ingest_candles.assert_awaited_once()
    kwargs = ingestion.ingest_candles.await_args.kwargs
    assert kwargs["symbols"] == [SYMBOL]
    assert kwargs["timeframe"] == "1m"
    assert kwargs["candles"] == [
        {
            "time": T0.isoformat(),
            "open": 49900.0,
            "high": 50100.0,
            "low": 49800.0,
            "close": 50000.0,
            "volume": 12.5,
        }
    ]
    # Der Entscheidungs-Flow läuft regulär weiter (NO_TRADE ohne Agenten).
    assert entry["decision"] == "NO_TRADE"
    assert decided is False
    state_store_of(loop).add_record.assert_called_once()


@pytest.mark.asyncio
async def test_hook_not_called_when_data_has_no_ohlcv_key() -> None:
    ingestion = MagicMock()
    ingestion.ingest_candles = AsyncMock()
    loop = make_loop(ohlcv_ingestion=ingestion)

    entry, decided, _ = await run_decide_symbol(loop, {"price": 50000.0})

    ingestion.ingest_candles.assert_not_awaited()
    assert entry["decision"] == "NO_TRADE"
    assert decided is False
    state_store_of(loop).add_record.assert_called_once()


@pytest.mark.asyncio
async def test_hook_ignores_non_dict_ohlcv_value() -> None:
    ingestion = MagicMock()
    ingestion.ingest_candles = AsyncMock()
    loop = make_loop(ohlcv_ingestion=ingestion)

    await run_decide_symbol(loop, {"price": 50000.0, "ohlcv": [49900.0, 50000.0]})

    ingestion.ingest_candles.assert_not_awaited()


@pytest.mark.asyncio
async def test_hook_failure_is_non_critical_and_loop_continues() -> None:
    ingestion = MagicMock()
    ingestion.ingest_candles = AsyncMock(side_effect=RuntimeError("influx down"))
    loop = make_loop(ohlcv_ingestion=ingestion)

    entry, decided, _ = await run_decide_symbol(loop, OHLCV_TICKER)

    # Der Aufruf wurde getätigt und abgefangen — die Iteration läuft zu Ende.
    assert ingestion.ingest_candles.await_count == 1
    assert entry["decision"] == "NO_TRADE"
    assert decided is False
    state_store_of(loop).add_record.assert_called_once()


@pytest.mark.asyncio
async def test_hook_skipped_when_ohlcv_ingestion_is_none() -> None:
    loop = make_loop()  # ohlcv_ingestion=None (Default)

    entry, decided, _ = await run_decide_symbol(loop, OHLCV_TICKER)

    assert entry["decision"] == "NO_TRADE"
    assert decided is False
    state_store_of(loop).add_record.assert_called_once()
