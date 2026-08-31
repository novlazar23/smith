"""Tests für ClickHouseMarketDataSink (persistence.py)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest
from apps.ingestion.persistence import ClickHouseMarketDataSink
from packages.persistence.clickhouse.engine import ClickHouseConfig


def _config() -> ClickHouseConfig:
    """Erzeugt eine Test-Config für die trading_events-Datenbank."""
    return ClickHouseConfig(
        host="clickhouse",
        port=8123,
        database="trading_events",
        user="orchestra",
        password="test-password",
    )


def _candle_event() -> dict[str, Any]:
    """Erzeugt ein verarbeitetes Candle-Event (Standard-Format)."""
    return {
        "symbol": "BTCUSDT",
        "type": "candle",
        "open": 100.5,
        "high": 102.0,
        "low": 99.0,
        "close": 101.25,
        "volume": 1234.5,
        "timestamp": datetime(2025, 6, 15, 10, 0, 0, tzinfo=UTC),
    }


def _candle_event_with_full_fields() -> dict[str, Any]:
    """Erzeugt ein Candle-Event mit vollständigen Producer-Feldern."""
    event = _candle_event()
    event.update(
        {
            "open_time": "2025-06-15T10:00:00+00:00",
            "close_time": "2025-06-15T10:00:59+00:00",
            "trade_count": 42,
            "is_closed": True,
            "venue": "DUMMY_EXCHANGE",
        }
    )
    return event


def _tick_event() -> dict[str, Any]:
    """Erzeugt ein verarbeitetes Tick-Event."""
    return {
        "symbol": "BTCUSDT",
        "type": "tick",
        "price": 101.25,
        "volume": 1.0,
        "timestamp": datetime(2025, 6, 15, 10, 0, 0, tzinfo=UTC),
    }


class TestClickHouseMarketDataSink:
    """Testet Persistierung von Events in ClickHouse."""

    def _sink_with_mocked_engine(self) -> tuple[ClickHouseMarketDataSink, MagicMock]:
        sink = ClickHouseMarketDataSink(config=_config())
        engine = MagicMock()
        sink._engine = engine
        return sink, engine

    def test_persist_candle_builds_insert_sql(self) -> None:
        """Candle-Events erzeugen ein INSERT INTO trading_events.candles."""
        sink, engine = self._sink_with_mocked_engine()

        result = sink.persist(_candle_event())

        assert result is True
        engine._execute.assert_called_once()
        sql = engine._execute.call_args[0][0]
        assert "INSERT INTO trading_events.candles" in sql
        assert "BTCUSDT" in sql
        assert "100.5" in sql
        assert "102.0" in sql
        assert "99.0" in sql
        assert "101.25" in sql
        assert "1234.5" in sql

    def test_persist_candle_full_fields(self) -> None:
        """Candle mit Producer-Feldern übernimmt venue, trade_count, is_closed."""
        sink, engine = self._sink_with_mocked_engine()

        result = sink.persist(_candle_event_with_full_fields())

        assert result is True
        sql = engine._execute.call_args[0][0]
        assert "DUMMY_EXCHANGE" in sql
        assert "dummy-adapter" in sql
        assert "42" in sql

    def test_persist_candle_open_time_from_timestamp(self) -> None:
        """open_time fällt auf timestamp zurück und wird als DateTime formatiert."""
        sink, engine = self._sink_with_mocked_engine()

        result = sink.persist(_candle_event())

        assert result is True
        sql = engine._execute.call_args[0][0]
        assert "2025-06-15 10:00:00" in sql

    def test_persist_tick_returns_false_without_sql(self) -> None:
        """Tick-Events werden ignoriert (False, keine SQL-Execution)."""
        sink, engine = self._sink_with_mocked_engine()

        result = sink.persist(_tick_event())

        assert result is False
        engine._execute.assert_not_called()

    def test_persist_unknown_type_returns_false(self) -> None:
        """Unbekannte Event-Typen werden ignoriert."""
        sink, engine = self._sink_with_mocked_engine()

        result = sink.persist({"symbol": "X", "type": "orderbook"})

        assert result is False
        engine._execute.assert_not_called()

    def test_persist_sql_error_returns_false_no_raise(self) -> None:
        """SQL-Fehler werden geloggt und als False zurückgegeben (kein Raise)."""
        sink, engine = self._sink_with_mocked_engine()
        engine._execute.side_effect = Exception("ClickHouse down")

        result = sink.persist(_candle_event())

        assert result is False

    def test_persist_invalid_candle_returns_false_no_raise(self) -> None:
        """Ungültige Candle-Felder (z. B. fehlender OHLC-Wert) → False, kein Raise."""
        sink, engine = self._sink_with_mocked_engine()
        event = _candle_event()
        del event["close"]

        result = sink.persist(event)

        assert result is False
        engine._execute.assert_not_called()

    def test_ensure_schema_calls_create_tables(self) -> None:
        """ensure_schema() führt create_tables() auf der Engine aus."""
        sink = ClickHouseMarketDataSink(config=_config())
        engine = MagicMock()
        sink._engine = engine

        sink.ensure_schema()

        engine.create_tables.assert_called_once()

    def test_default_config_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ohne Config wird die Umgebung (CH_HOST/CH_PORT/CH_DB/CH_PASSWORD) genutzt."""
        monkeypatch.setenv("CH_HOST", "ch-host")
        monkeypatch.setenv("CH_PORT", "9000")
        monkeypatch.setenv("CH_DB", "my_events")
        monkeypatch.setenv("CH_PASSWORD", "s3cret")

        sink = ClickHouseMarketDataSink()

        assert sink._config.host == "ch-host"
        assert sink._config.port == 9000
        assert sink._config.database == "my_events"
        assert sink._config.user == "orchestra"
        assert sink._config.password == "s3cret"
