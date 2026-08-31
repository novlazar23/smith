"""Tests für DataIngestionService.consume() mit gestutztem Kafka-Consumer."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

from apps.ingestion import consumer as consumer_module
from apps.ingestion.consumer import DataIngestionService


class _FakeMessage:
    """Imitiert ein confluent_kafka Message-Objekt."""

    def __init__(self, value: bytes | None, error: Exception | None = None) -> None:
        self._value = value
        self._error = error

    def value(self) -> bytes | None:
        return self._value

    def error(self) -> Exception | None:
        return self._error


class _FakeKafkaConsumer:
    """Skriptierter Substitut für den confluent_kafka Consumer.

    Liefert die skriptierten Messages sequenziell aus ``poll``;
    nach Script-Auslauf gibt ``poll`` dauerhaft ``None`` zurück.
    """

    def __init__(self, script: list[_FakeMessage | None]) -> None:
        self._script = list(script)
        self.config: dict[str, str] | None = None
        self.topics: list[str] | None = None
        self.poll_calls = 0
        self.closed = False

    def __call__(self, config: dict[str, str]) -> _FakeKafkaConsumer:
        self.config = config
        return self

    def subscribe(self, topics: list[str]) -> None:
        self.topics = topics

    def poll(self, timeout: float | None = None) -> _FakeMessage | None:
        self.poll_calls += 1
        if self._script:
            return self._script.pop(0)
        return None

    def close(self) -> None:
        self.closed = True


def _candle_payload() -> dict[str, Any]:
    """Erzeugt ein Producer-konformes Candle-Event (flache Struktur)."""
    return {
        "symbol": "AAPL",
        "instrument": "AAPL",
        "timestamp": "2025-06-15T10:00:00+00:00",
        "open_time": "2025-06-15T10:00:00+00:00",
        "close_time": "2025-06-15T10:00:59+00:00",
        "open": 150.0,
        "high": 155.0,
        "low": 148.0,
        "close": 152.5,
        "volume": 10000.0,
        "trade_count": 42,
        "is_closed": True,
        "type": "candle",
        "venue": "DUMMY_EXCHANGE",
    }


def _tick_payload() -> dict[str, Any]:
    """Erzeugt ein Producer-konformes Tick-Event."""
    return {
        "symbol": "AAPL",
        "timestamp": "2025-06-15T10:00:01+00:00",
        "price": 152.5,
        "volume": 100.0,
        "type": "tick",
    }


class TestConsume:
    """Testet den consume()-Loop von DataIngestionService."""

    def _run(
        self,
        script: list[_FakeMessage | None],
        max_messages: int | None,
    ) -> tuple[list[dict[str, Any]], _FakeKafkaConsumer, DataIngestionService]:
        """Führt consume() mit gestutztem Consumer aus."""
        fake = _FakeKafkaConsumer(script)
        svc = DataIngestionService(topic="market_data", bootstrap_servers=["redpanda:9092"])
        seen: list[dict[str, Any]] = []
        with (
            patch.object(consumer_module, "_KafkaConsumer", fake),
            patch.object(consumer_module, "REDPANDA_AVAILABLE", new=True),
        ):
            svc.consume(seen.append, max_messages=max_messages)
        return seen, fake, svc

    def test_consume_processes_candle_and_skips_garbage(self) -> None:
        """Verarbeitet Candle-Events und überspringt nicht parsierbare Messages."""
        script = [
            _FakeMessage(b"das ist kein JSON {{"),
            _FakeMessage(json.dumps(_candle_payload()).encode("utf-8")),
            _FakeMessage(None),
        ]
        seen, fake, _ = self._run(script, max_messages=1)

        assert len(seen) == 1
        event = seen[0]
        assert event["symbol"] == "AAPL"
        assert event["open"] == 150.0
        assert event["high"] == 155.0
        assert event["low"] == 148.0
        assert event["close"] == 152.5
        assert event["volume"] == 10000.0
        assert event["type"] == "candle"
        # Garbage-Message wurde gepollt und übersprungen (zählt nicht
        # gegen max_messages); Loop stoppt nach dem Candle-Event.
        assert fake.poll_calls == 2
        assert fake.closed is True

    def test_consume_max_messages_stops_loop(self) -> None:
        """max_messages beendet den Loop nach N verarbeiteten Events."""
        script = [
            _FakeMessage(json.dumps(_candle_payload()).encode("utf-8")),
            _FakeMessage(json.dumps(_candle_payload()).encode("utf-8")),
            _FakeMessage(json.dumps(_candle_payload()).encode("utf-8")),
        ]
        seen, fake, _ = self._run(script, max_messages=1)

        assert len(seen) == 1
        # Loop stoppt direkt nach dem ersten verarbeiteten Event —
        # weitere Polls erfolgen nicht.
        assert fake.poll_calls == 1
        assert fake.closed is True

    def test_consume_tick_fallback(self) -> None:
        """Tick-Events werden per Fallback über process_tick verarbeitet."""
        script = [
            _FakeMessage(json.dumps(_tick_payload()).encode("utf-8")),
        ]
        seen, _, _ = self._run(script, max_messages=1)

        assert len(seen) == 1
        assert seen[0]["type"] == "tick"
        assert seen[0]["price"] == 152.5

    def test_consume_skips_unsupported_events(self) -> None:
        """Events, die weder Candle noch Tick sind, werden übersprungen."""
        script = [
            _FakeMessage(json.dumps({"symbol": "X", "type": "weird"}).encode("utf-8")),
            _FakeMessage(json.dumps(_candle_payload()).encode("utf-8")),
        ]
        seen, fake, _ = self._run(script, max_messages=1)

        assert len(seen) == 1
        assert seen[0]["type"] == "candle"
        assert fake.poll_calls == 2
        assert fake.closed is True

    def test_consume_tracked_in_processed_events(self) -> None:
        """Verarbeitete Events landen in _processed_events."""
        script = [
            _FakeMessage(json.dumps(_candle_payload()).encode("utf-8")),
        ]
        _, _, svc = self._run(script, max_messages=1)

        assert len(svc.get_processed_stream()) == 1
        assert svc.get_processed_stream()[0]["type"] == "candle"

    def test_consume_config_and_subscription(self) -> None:
        """Consumer wird mit korrekter Config und Topic abonniert."""
        script = [
            _FakeMessage(json.dumps(_candle_payload()).encode("utf-8")),
        ]
        _, fake, _ = self._run(script, max_messages=1)

        assert fake.topics == ["market_data"]
        assert fake.config is not None
        assert fake.config["bootstrap.servers"] == "redpanda:9092"
        assert fake.config["group.id"] == "ingestion-consumer"
        assert fake.config["auto.offset.reset"] == "earliest"
        assert fake.config["enable.auto.commit"] is True

    def test_consume_without_kafka_available_returns_early(self) -> None:
        """Ohne verfügbaren Kafka-Client wird early return gemacht."""
        svc = DataIngestionService(topic="market_data", bootstrap_servers=["redpanda:9092"])
        seen: list[dict[str, Any]] = []
        with (
            patch.object(consumer_module, "_KafkaConsumer", None),
            patch.object(consumer_module, "REDPANDA_AVAILABLE", new=False),
        ):
            svc.consume(seen.append, max_messages=1)

        assert seen == []
        assert svc.get_processed_stream() == []
