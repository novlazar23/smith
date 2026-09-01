"""Tests für die Binance-Quellenauswahl des Market-Data-Producer.

Deckung: Source-Selektion (Parameter + Fallback bei unbekanntem Wert),
Symbol-Mapping, Auswahl der letzten geschlossenen Kerze, Dummy-Fallback
bei Binance-Fehlern, automatisches Resümieren des Live-Pfades und
Event-Format-Parität zwischen Dummy- und Binance-Pfad.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from apps.market_producer import producer as producer_module
from apps.market_producer.producer import (
    DummyMarketDataProducer,
    select_last_closed_candle,
    to_exchange_symbol,
)
from packages.ingestion.adapter.base import (
    ConnectionError as AdapterConnectionError,
)
from packages.ingestion.adapter.base import (
    ConnectionState,
)
from packages.ingestion.adapter.binance import BinanceAdapter
from packages.ingestion.adapter.dummy import DummyAdapter

SYMBOL = "BTC/USDT"
BINANCE_VENUE = "BINANCE_FUTURES"
DUMMY_VENUE = "DUMMY_EXCHANGE"


class FakeProducer:
    """Stellvertreter fuer den confluent_kafka.Producer (rekordiert produce-Aufrufe)."""

    def __init__(self, config: dict[str, str] | None = None) -> None:
        self.config = config
        self.calls: list[tuple[str, bytes | None, bytes | None]] = []
        self.poll_count = 0

    def produce(
        self,
        topic: str,
        key: bytes | None = None,
        value: bytes | None = None,
        **kwargs: Any,
    ) -> None:
        self.calls.append((topic, key, value))

    def poll(self, timeout: float = 0.0) -> int:
        self.poll_count += 1
        return 0


def _live_candle(now: datetime | None = None, **overrides: Any) -> dict[str, Any]:
    """Geschlossene Kerze, die bei ``now`` endet (gestempelt wie vom Base-Adapter)."""
    moment = now or datetime.now(UTC)
    candle: dict[str, Any] = {
        "open_time": moment - timedelta(minutes=1),
        "close_time": moment,
        "open": 77900.0,
        "high": 77950.5,
        "low": 77850.25,
        "close": 77925.0,
        "volume": 321.0,
        "trade_count": 4242,
        "is_closed": True,
        "type": "candle",
        "instrument": "BTCUSDT",
        "venue": BINANCE_VENUE,
    }
    candle.update(overrides)
    return candle


def _forming_candle(now: datetime | None = None) -> dict[str, Any]:
    """Kerze in Ausbildung: close_time liegt in der Zukunft."""
    moment = now or datetime.now(UTC)
    return _live_candle(now=moment, open_time=moment, close_time=moment + timedelta(minutes=1))


async def _dummy_fetch_one(
    self: DummyAdapter, symbol: str, interval: str = "1m", limit: int = 100
) -> list[dict[str, Any]]:
    """Dummy-Fallback-Kerze im Adapter-Format (exakt wie der DummyAdapter)."""
    moment = datetime.now(UTC)
    return [{
        "open_time": moment - timedelta(minutes=1),
        "close_time": moment,
        "open": 67500.0,
        "high": 67510.5,
        "low": 67490.25,
        "close": 67505.0,
        "volume": 123.456,
        "trade_count": 777,
        "is_closed": True,
        "type": "candle",
        "instrument": SYMBOL,
        "venue": DUMMY_VENUE,
    }]


class BinanceStub:
    """Zustand des gestubten BinanceAdapter.fetch_candles.

    ``results`` ist eine Liste von Ergebnislisten oder Exceptions; der
    letzte Eintrag gilt dauerhaft, solange er nicht ersetzt wird.
    """

    def __init__(self, results: list[list[dict[str, Any]] | Exception]) -> None:
        self.calls: list[tuple[str, str, int]] = []
        self.results = list(results)


def _bind_binance_stub(stub: BinanceStub) -> Any:
    """Erzeugt die an den Adapter zu bindende fetch_candles-Ersatzfunktion."""

    async def fetch_candles(
        self: BinanceAdapter, symbol: str, interval: str = "1m", limit: int = 100
    ) -> list[dict[str, Any]]:
        del self
        stub.calls.append((symbol, interval, limit))
        result = stub.results.pop(0) if len(stub.results) > 1 else stub.results[0]
        if isinstance(result, Exception):
            raise result
        return list(result)

    return fetch_candles


async def _noop_connect(self: BinanceAdapter) -> None:
    """Vermeidet eine echte aiohttp-Session in Unit-Tests."""
    self._state = ConnectionState.CONNECTED


async def _noop_disconnect(self: BinanceAdapter) -> None:
    """Vermeidet eine echte aiohttp-Session in Unit-Tests."""
    self._state = ConnectionState.DISCONNECTED


@pytest.fixture
def binance_producer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> tuple[DummyMarketDataProducer, FakeProducer, BinanceStub]:
    """Binance-Producer mit gestubten Adaptern, Fake-Producer und isolierter Heartbeat."""
    monkeypatch.setattr(DummyAdapter, "fetch_candles", _dummy_fetch_one)
    monkeypatch.setattr(BinanceAdapter, "connect", _noop_connect)
    monkeypatch.setattr(BinanceAdapter, "disconnect", _noop_disconnect)
    monkeypatch.setattr(producer_module, "HEARTBEAT_PATH", str(tmp_path / "heartbeat"))
    stub = BinanceStub([[_live_candle(), _forming_candle()]])
    monkeypatch.setattr(BinanceAdapter, "fetch_candles", _bind_binance_stub(stub))
    fake = FakeProducer()
    p = DummyMarketDataProducer([SYMBOL], bootstrap_servers="localhost:9092", source="binance")
    p._producer = fake
    return p, fake, stub


@pytest.fixture
def dummy_producer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> tuple[DummyMarketDataProducer, FakeProducer]:
    """Dummy-Producer (Standardquelle) mit gestubtem Fetch und isolierter Heartbeat."""
    monkeypatch.setattr(DummyAdapter, "fetch_candles", _dummy_fetch_one)
    monkeypatch.setattr(producer_module, "HEARTBEAT_PATH", str(tmp_path / "heartbeat"))
    fake = FakeProducer()
    p = DummyMarketDataProducer([SYMBOL], bootstrap_servers="localhost:9092")
    p._producer = fake
    return p, fake


# -- Source-Selektion ---------------------------------------------------


def test_source_binance_creates_futures_adapter() -> None:
    """source='binance' erzeugt einen keylosen Futures-Adapter."""
    p = DummyMarketDataProducer([SYMBOL], bootstrap_servers="localhost:9092", source="binance")

    assert p._source == "binance"
    assert isinstance(p._binance_adapter, BinanceAdapter)
    assert p._binance_adapter.venue == BINANCE_VENUE
    assert p._binance_adapter.config.api_key == ""
    assert p._binance_adapter.config.api_secret == ""


def test_default_source_is_dummy() -> None:
    """Ohne source-Parameter bleibt der Producer im Dummy-Modus (Rueckwaertskompatibilitaet)."""
    p = DummyMarketDataProducer([SYMBOL], bootstrap_servers="localhost:9092")

    assert p._source == "dummy"
    assert p._binance_adapter is None


def test_unknown_source_falls_back_to_dummy() -> None:
    """Unbekannter Quellname faellt mit Warnung auf den Dummy-Modus zurueck."""
    p = DummyMarketDataProducer([SYMBOL], bootstrap_servers="localhost:9092", source="coinbase")

    assert p._source == "dummy"
    assert p._binance_adapter is None


# -- Symbol-Mapping -----------------------------------------------------


def test_symbol_mapping() -> None:
    """Kanonische Instrumente werden auf das Binance-Symbolformat gemappt."""
    assert to_exchange_symbol("BTC/USDT") == "BTCUSDT"
    assert to_exchange_symbol("ETH/USDT") == "ETHUSDT"
    assert to_exchange_symbol("eth/usdt") == "ETHUSDT"
    assert to_exchange_symbol("BTCUSDT") == "BTCUSDT"


# -- Auswahl der geschlossenen Kerze ------------------------------------


def test_select_last_closed_candle_picks_latest_closed() -> None:
    """Aus [alt, neu, in-ausbildung] wird die letzte geschlossene Kerze gewaehlt."""
    moment = datetime.now(UTC)
    old = _live_candle(now=moment - timedelta(minutes=1))
    new = _live_candle(now=moment)
    forming = _forming_candle(now=moment)

    selected = select_last_closed_candle([old, new, forming], now=moment)

    assert selected is new


def test_select_last_closed_candle_returns_none_when_none_closed() -> None:
    """Nur Kerzen in Ausbildung → None (Trigger fuer den Dummy-Fallback)."""
    moment = datetime.now(UTC)
    assert select_last_closed_candle([_forming_candle(now=moment)], now=moment) is None
    assert select_last_closed_candle([], now=moment) is None


# -- Live-Tick ----------------------------------------------------------


async def test_binance_tick_emits_live_event(
    binance_producer: tuple[DummyMarketDataProducer, FakeProducer, BinanceStub]
) -> None:
    """Binance-Tick publish die Live-Kerze (venue=BINANCE_FUTURES, Exchange-Symbol)."""
    p, fake, stub = binance_producer
    expected = stub.results[0][0]

    produced = await p._tick()

    assert produced == 1
    topic, key, value = fake.calls[0]
    assert topic == "market_data"
    assert key == SYMBOL.encode()
    event = json.loads(value)
    assert event["venue"] == BINANCE_VENUE
    assert event["symbol"] == SYMBOL
    assert event["instrument"] == SYMBOL
    assert event["type"] == "candle"
    assert event["open"] == 77900.0
    assert event["close"] == 77925.0
    assert event["trade_count"] == 4242
    assert event["open_time"] == expected["open_time"].isoformat()
    assert event["close_time"] == expected["close_time"].isoformat()
    # Der Adapter wird mit dem Exchange-Symbol und 3 Kerzen abgefragt
    assert stub.calls == [("BTCUSDT", "1m", 3)]


async def test_binance_fallback_on_fetch_error(
    binance_producer: tuple[DummyMarketDataProducer, FakeProducer, BinanceStub]
) -> None:
    """Binance-Fehler (Netzwerk/HTTP) → Dummy-Kerze desselben Ticks, kein Crash."""
    p, fake, stub = binance_producer
    stub.results = [AdapterConnectionError("network unreachable")]

    produced = await p._tick()

    assert produced == 1
    event = json.loads(fake.calls[0][2])
    assert event["venue"] == DUMMY_VENUE
    assert event["close"] == 67505.0
    assert event["instrument"] == SYMBOL


async def test_binance_fallback_when_only_forming_candle(
    binance_producer: tuple[DummyMarketDataProducer, FakeProducer, BinanceStub]
) -> None:
    """Ohne geschlossene Kerze (nur in Ausbildung) → Dummy-Fallback."""
    p, fake, stub = binance_producer
    stub.results = [[_forming_candle()]]

    produced = await p._tick()

    assert produced == 1
    event = json.loads(fake.calls[0][2])
    assert event["venue"] == DUMMY_VENUE


async def test_binance_resumes_after_failure(
    binance_producer: tuple[DummyMarketDataProducer, FakeProducer, BinanceStub]
) -> None:
    """Nach einem fehlgeschlagenen Tick resuemiert der Live-Pfad automatisch."""
    p, fake, stub = binance_producer
    stub.results = [AdapterConnectionError("timeout"), [_live_candle(), _forming_candle()]]

    await p._tick()
    await p._tick()

    venues = [json.loads(value)["venue"] for (_, _, value) in fake.calls]
    assert venues == [DUMMY_VENUE, BINANCE_VENUE]


# -- Event-Format-Paritaet ----------------------------------------------


async def test_event_format_parity_dummy_vs_binance(
    dummy_producer: tuple[DummyMarketDataProducer, FakeProducer],
    binance_producer: tuple[DummyMarketDataProducer, FakeProducer, BinanceStub],
) -> None:
    """Beide Pfade liefern flache Events mit identischer Feldmenge und Typen."""
    (dp, dfake), (bp, bfake, _stub) = dummy_producer, binance_producer

    assert await dp._tick() == 1
    assert await bp._tick() == 1

    dummy_event = json.loads(dfake.calls[0][2])
    binance_event = json.loads(bfake.calls[0][2])

    assert set(dummy_event) == set(binance_event)
    for key in dummy_event:
        assert type(dummy_event[key]) is type(binance_event[key]), key
