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
    """Stellvertreter fuer DummyAdapter._fetch_candles_raw — liefert genau eine Kerze."""
    return [_candle()]


async def _fetch_none(
    self: DummyAdapter, symbol: str, interval: str = "1m", limit: int = 100
) -> list[dict[str, Any]]:
    """Stellvertreter fuer DummyAdapter._fetch_candles_raw — liefert keine Kerzen."""
    return []


@pytest.fixture
def fake_producer() -> FakeProducer:
    """Instanz des Fake-Producer."""
    return FakeProducer()


@pytest.fixture
def producer(
    fake_producer: FakeProducer, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> DummyMarketDataProducer:
    """Producer mit gestubtem _fetch_candles_raw, Fake-Producer und isolierter Heartbeat."""
    monkeypatch.setattr(DummyAdapter, "_fetch_candles_raw", _fetch_one)
    monkeypatch.setattr(producer_module, "HEARTBEAT_PATH", str(tmp_path / "heartbeat"))
    p = DummyMarketDataProducer([SYMBOL], bootstrap_servers="localhost:9092", topic="market_data")
    p._producer = fake_producer
    return p


async def test_tick_produces_flat_candle_event(producer: DummyMarketDataProducer, fake_producer: FakeProducer) -> None:
    """_tick() publish genau ein flaches Candle-Event mit allen erforderten Feldern."""
    produced = await producer._tick()

    assert produced == 1
    assert len(fake_producer.calls) == 1
    topic, key, value = fake_producer.calls[0]
    assert topic == "market_data"
    assert key == SYMBOL.encode()
    assert fake_producer.poll_count >= 1

    event = json.loads(value)
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
    monkeypatch.setattr(DummyAdapter, "_fetch_candles_raw", _fetch_one)
    monkeypatch.setattr(producer_module, "HEARTBEAT_PATH", str(tmp_path / "heartbeat"))
    fake = FakeProducer()
    p = DummyMarketDataProducer(["DOGE/USDT"], bootstrap_servers="localhost:9092")
    p._producer = fake

    assert p._adapters["DOGE/USDT"]._base_price == 100.0

    produced = await p._tick()

    assert produced == 1
    topic, key, value = fake.calls[0]
    assert topic == "market_data"
    assert key == b"DOGE/USDT"
    event = json.loads(value)
    assert event["symbol"] == "DOGE/USDT"
    assert event["instrument"] == "DOGE/USDT"
    assert event["type"] == "candle"


async def test_empty_fetch_produces_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Leeres Fetch-Ergebnis → keine produce-Aufrufe, Rueckgabe 0."""
    monkeypatch.setattr(DummyAdapter, "_fetch_candles_raw", _fetch_none)
    monkeypatch.setattr(producer_module, "HEARTBEAT_PATH", str(tmp_path / "heartbeat"))
    fake = FakeProducer()
    p = DummyMarketDataProducer([SYMBOL], bootstrap_servers="localhost:9092")
    p._producer = fake

    assert await p._tick() == 0
    assert fake.calls == []
