"""Tests fuer DummyMarketDataProducer — apps/market_producer/producer.py."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest
from apps.market_producer import producer as producer_module
from apps.market_producer.producer import DummyMarketDataProducer
from packages.ingestion.adapter.dummy import DummyAdapter

SYMBOL = "BTC/USDT"
OPEN_TIME = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
CLOSE_TIME = datetime(2026, 1, 1, 12, 1, 0, tzinfo=UTC)
VENUE = "DUMMY_EXCHANGE"

REQUIRED_EVENT_KEYS = {
    "symbol",
    "timestamp",
    "open_time",
    "close_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "is_closed",
    "type",
    "instrument",
    "venue",
}


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


def _candle() -> dict[str, Any]:
    """Konstruiert eine manuelle Kerze im DummyAdapter-Format."""
    return {
        "open_time": OPEN_TIME,
        "close_time": CLOSE_TIME,
        "open": 67500.0,
        "high": 67510.5,
        "low": 67490.25,
        "close": 67505.0,
        "volume": 123.456,
        "trade_count": 777,
        "is_closed": True,
        "type": "candle",
        "instrument": SYMBOL,
        "venue": VENUE,
    }


async def _fetch_one(
    self: DummyAdapter, symbol: str, interval: str = "1m", limit: int = 100
) -> list[dict[str, Any]]:
    """Stellvertreter fuer DummyAdapter.fetch_candles — liefert genau eine Kerze."""
    return [_candle()]


async def _fetch_none(
    self: DummyAdapter, symbol: str, interval: str = "1m", limit: int = 100
) -> list[dict[str, Any]]:
    """Stellvertreter fuer DummyAdapter.fetch_candles — liefert keine Kerzen."""
    return []


def _attach_fake_producer(
    producer: DummyMarketDataProducer,
    fake: FakeProducer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tauscht den confluent-Producer gegen den Test-Doppel aus."""
    monkeypatch.setattr(producer, "_producer", fake)


def _event(fake: FakeProducer, index: int = -1) -> dict[str, Any]:
    """Dekodiert ein aufgezeichnetes Producer-Event."""
    value = fake.calls[index][2]
    assert value is not None
    return json.loads(value)


@pytest.fixture
def fake_producer() -> FakeProducer:
    """Instanz des Fake-Producer."""
    return FakeProducer()


@pytest.fixture
def producer(
    fake_producer: FakeProducer, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> DummyMarketDataProducer:
    """Producer mit gestubtem fetch_candles, Fake-Producer und isolierter Heartbeat."""
    monkeypatch.setattr(DummyAdapter, "fetch_candles", _fetch_one)
    monkeypatch.setattr(producer_module, "HEARTBEAT_PATH", str(tmp_path / "heartbeat"))
    p = DummyMarketDataProducer([SYMBOL], bootstrap_servers="localhost:9092", topic="market_data")
    _attach_fake_producer(p, fake_producer, monkeypatch)
    return p


async def test_tick_produces_flat_candle_event(producer: DummyMarketDataProducer, fake_producer: FakeProducer) -> None:
    """_tick() publish genau ein flaches Candle-Event mit allen erforderten Feldern."""
    produced = await producer._tick()

    assert produced == 1
    assert len(fake_producer.calls) == 1
    topic, key, _value = fake_producer.calls[0]
    assert topic == "market_data"
    assert key == SYMBOL.encode()
    assert fake_producer.poll_count >= 1

    event = _event(fake_producer)
    assert set(event) >= REQUIRED_EVENT_KEYS
    assert event["type"] == "candle"
    assert event["symbol"] == SYMBOL
    assert event["instrument"] == SYMBOL
    assert event["venue"] == VENUE
    assert event["timestamp"] == OPEN_TIME.isoformat()
    assert event["open_time"] == OPEN_TIME.isoformat()
    assert event["close_time"] == CLOSE_TIME.isoformat()
    assert event["open"] == 67500.0
    assert event["high"] == 67510.5
    assert event["low"] == 67490.25
    assert event["close"] == 67505.0
    assert event["volume"] == 123.456
    assert event["trade_count"] == 777
    assert event["is_closed"] is True


async def test_unknown_symbol_falls_back_to_base_price_100(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Unbekanntes Symbol bekommt Basispreis 100.0 und crasht nicht."""
    monkeypatch.setattr(DummyAdapter, "fetch_candles", _fetch_one)
    monkeypatch.setattr(producer_module, "HEARTBEAT_PATH", str(tmp_path / "heartbeat"))
    fake = FakeProducer()
    p = DummyMarketDataProducer(["DOGE/USDT"], bootstrap_servers="localhost:9092")
    _attach_fake_producer(p, fake, monkeypatch)

    assert p._adapters["DOGE/USDT"]._base_price == 100.0

    produced = await p._tick()

    assert produced == 1
    topic, key, _value = fake.calls[0]
    assert topic == "market_data"
    assert key == b"DOGE/USDT"
    event = _event(fake)
    assert event["symbol"] == "DOGE/USDT"
    assert event["instrument"] == "DOGE/USDT"
    assert event["type"] == "candle"


async def test_empty_fetch_produces_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Leeres Fetch-Ergebnis → keine produce-Aufrufe, Rueckgabe 0."""
    monkeypatch.setattr(DummyAdapter, "fetch_candles", _fetch_none)
    monkeypatch.setattr(producer_module, "HEARTBEAT_PATH", str(tmp_path / "heartbeat"))
    fake = FakeProducer()
    p = DummyMarketDataProducer([SYMBOL], bootstrap_servers="localhost:9092")
    _attach_fake_producer(p, fake, monkeypatch)

    assert await p._tick() == 0
    assert fake.calls == []


class FakeBinanceAdapter:
    """Stellvertreter fuer BinanceAdapter ohne Netzwerk."""

    def __init__(
        self,
        candles: list[dict[str, Any]] | None = None,
        error: Exception | None = None,
        *,
        connected: bool = True,
        drop_on_error: bool = False,
    ) -> None:
        self.venue = "BINANCE_FUTURES"
        self._candles = candles if candles is not None else []
        self._error = error
        self._connected = connected
        self._drop_on_error = drop_on_error
        self.fetch_calls = 0
        self.connect_calls = 0

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        self.connect_calls += 1
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def fetch_candles(
        self, symbol: str, interval: str = "1m", limit: int = 100
    ) -> list[dict[str, Any]]:
        self.fetch_calls += 1
        if self._error is not None:
            if self._drop_on_error:
                self._connected = False
            raise self._error
        return [dict(candle) for candle in self._candles]


def _live_candle(close: float = 100.0, open_time: Any = OPEN_TIME) -> dict[str, Any]:
    """Konstruiert eine geschlossene Live-Kerze im Binance-Format."""
    return {
        "open_time": open_time,
        "close_time": CLOSE_TIME,
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": 10.0,
        "trade_count": 42,
        "is_closed": True,
        "type": "candle",
        "instrument": SYMBOL,
        "venue": "BINANCE_FUTURES",
    }


async def test_binance_source_publishes_live_candle_with_guard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Binance-Modus publishLive-Kerze und pflegt Health/Guard."""
    fake_binance = FakeBinanceAdapter(candles=[_live_candle()])
    monkeypatch.setattr(producer_module, "BinanceAdapter", lambda: fake_binance)
    monkeypatch.setattr(DummyAdapter, "fetch_candles", _fetch_one)
    monkeypatch.setattr(producer_module, "HEARTBEAT_PATH", str(tmp_path / "heartbeat"))

    fake = FakeProducer()
    p = DummyMarketDataProducer(
        [SYMBOL], bootstrap_servers="localhost:9092", source="binance"
    )
    _attach_fake_producer(p, fake, monkeypatch)

    produced = await p._tick()

    assert produced == 1
    assert fake_binance.fetch_calls == 1
    assert p._ingestion_guard is not None
    assert p._ingestion_guard.active_venue == "BINANCE_FUTURES"
    assert p._ingestion_guard._health.get_state("BINANCE_FUTURES").overall_healthy is True
    assert p._ingestion_guard._failover.health_scores["BINANCE_FUTURES"] == 1.0

    event = _event(fake)
    assert event["venue"] == "BINANCE_FUTURES"
    assert event["close"] == 100.0


async def test_binance_fetch_failure_falls_back_to_dummy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Einzelne Binance-Exception führt zu Dummy-Event, aber noch nicht zu Failover."""
    fake_binance = FakeBinanceAdapter(error=RuntimeError("network down"))
    monkeypatch.setattr(producer_module, "BinanceAdapter", lambda: fake_binance)
    monkeypatch.setattr(DummyAdapter, "fetch_candles", _fetch_one)
    monkeypatch.setattr(producer_module, "HEARTBEAT_PATH", str(tmp_path / "heartbeat"))

    fake = FakeProducer()
    p = DummyMarketDataProducer(
        [SYMBOL], bootstrap_servers="localhost:9092", source="binance"
    )
    _attach_fake_producer(p, fake, monkeypatch)

    produced = await p._tick()

    assert produced == 1
    event = _event(fake)
    assert event["venue"] == VENUE
    assert p._ingestion_guard is not None
    assert p._ingestion_guard.active_venue == "BINANCE_FUTURES"
    assert p._ingestion_guard._health.get_state("BINANCE_FUTURES").overall_healthy is True
    assert p._ingestion_guard._failover.health_scores["BINANCE_FUTURES"] == 1.0


async def test_binance_quality_fail_falls_back_to_dummy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Unrealistischer Preisprung wird vom Quality Gate blockiert."""
    first_candle = _live_candle(close=100.0)
    second_candle = _live_candle(close=200.0)
    fake_binance = FakeBinanceAdapter(candles=[first_candle])
    monkeypatch.setattr(producer_module, "BinanceAdapter", lambda: fake_binance)
    monkeypatch.setattr(DummyAdapter, "fetch_candles", _fetch_one)
    monkeypatch.setattr(producer_module, "HEARTBEAT_PATH", str(tmp_path / "heartbeat"))

    fake = FakeProducer()
    p = DummyMarketDataProducer(
        [SYMBOL], bootstrap_servers="localhost:9092", source="binance"
    )
    _attach_fake_producer(p, fake, monkeypatch)

    assert await p._tick() == 1
    assert _event(fake, 0)["venue"] == "BINANCE_FUTURES"

    fake_binance._candles = [second_candle]
    assert await p._tick() == 1
    assert fake_binance.fetch_calls == 2
    assert _event(fake, 1)["venue"] == VENUE


async def test_binance_repeated_fetch_failures_failover_to_backup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Drei aufeinanderfolgende Fetch-Fehler aktivieren den Failover auf das Backup."""
    fake_binance = FakeBinanceAdapter(error=RuntimeError("network down"))
    monkeypatch.setattr(producer_module, "BinanceAdapter", lambda: fake_binance)
    monkeypatch.setattr(DummyAdapter, "fetch_candles", _fetch_one)
    monkeypatch.setattr(producer_module, "HEARTBEAT_PATH", str(tmp_path / "heartbeat"))

    fake = FakeProducer()
    p = DummyMarketDataProducer(
        [SYMBOL], bootstrap_servers="localhost:9092", source="binance"
    )
    _attach_fake_producer(p, fake, monkeypatch)
    assert p._ingestion_guard is not None

    try:
        for _ in range(3):
            assert await p._tick() == 1
        assert fake_binance.fetch_calls == 3
        assert p._ingestion_guard.active_venue == "BINANCE_FUTURES"

        assert await p._tick() == 1
        assert fake_binance.fetch_calls == 3
        assert p._ingestion_guard.active_venue == VENUE
        assert _event(fake, 3)["venue"] == VENUE
    finally:
        await p._ingestion_guard.close()


async def test_binance_reconnect_hook_starts_background_task(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Getrennter Adapter triggert AutoReconnector über den registrierten Hook."""
    import asyncio

    real_sleep = asyncio.sleep

    async def fake_sleep(delay: float) -> None:
        del delay

    monkeypatch.setattr("asyncio.sleep", fake_sleep)
    fake_binance = FakeBinanceAdapter(
        error=RuntimeError("disconnected"),
        connected=False,
        drop_on_error=True,
    )
    monkeypatch.setattr(producer_module, "BinanceAdapter", lambda: fake_binance)
    monkeypatch.setattr(DummyAdapter, "fetch_candles", _fetch_one)
    monkeypatch.setattr(producer_module, "HEARTBEAT_PATH", str(tmp_path / "heartbeat"))

    fake = FakeProducer()
    p = DummyMarketDataProducer(
        [SYMBOL], bootstrap_servers="localhost:9092", source="binance"
    )
    _attach_fake_producer(p, fake, monkeypatch)
    assert p._ingestion_guard is not None

    try:
        await p._tick()
        for _ in range(10):
            await real_sleep(0)

        assert fake_binance.connect_calls >= 1
        assert fake_binance.is_connected is True
        reconnector = p._ingestion_guard._reconnectors["BINANCE_FUTURES"]
        assert reconnector.success_count == 1
    finally:
        await p._ingestion_guard.close()
