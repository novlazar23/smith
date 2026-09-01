"""Tests für die Data Ingestion — MarketDataProcessor und DataIngestionService."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from apps.ingestion.consumer import DataIngestionService, MarketDataProcessor


def _candle_event() -> dict[str, Any]:
    """Returns a valid raw candle event."""
    return {
        "symbol": "AAPL",
        "timestamp": datetime(2025, 6, 15, 10, 0, 0, tzinfo=UTC),
        "type": "candle",
        "open": 150.0,
        "high": 155.0,
        "low": 148.0,
        "close": 152.5,
        "volume": 10000.0,
    }


def _tick_event() -> dict[str, Any]:
    """Returns a valid raw tick event."""
    return {
        "symbol": "AAPL",
        "timestamp": datetime(2025, 6, 15, 10, 0, 0, tzinfo=UTC),
        "type": "tick",
        "price": 152.5,
        "volume": 100.0,
    }


class TestMarketDataProcessor:
    """Testet MarketDataProcessor für Candle und Tick Events."""

    def test_process_candle(self) -> None:
        """Verarbeitet ein gültiges Candle-Event korrekt."""
        proc = MarketDataProcessor()
        raw = _candle_event()
        result = proc.process_candle(raw)

        assert result["symbol"] == "AAPL"
        assert result["open"] == 150.0
        assert result["high"] == 155.0
        assert result["low"] == 148.0
        assert result["close"] == 152.5
        assert result["volume"] == 10000.0
        assert result["type"] == "candle"
        assert result["timestamp"] == datetime(2025, 6, 15, 10, 0, 0, tzinfo=UTC)

    def test_process_tick(self) -> None:
        """Verarbeitet ein gültiges Tick-Event korrekt."""
        proc = MarketDataProcessor()
        raw = _tick_event()
        result = proc.process_tick(raw)

        assert result["symbol"] == "AAPL"
        assert result["price"] == 152.5
        assert result["volume"] == 100.0
        assert result["type"] == "tick"
        assert isinstance(result["timestamp"], datetime)

    def test_missing_field_raises(self) -> None:
        """Wirft ValueError wenn erforderliche Felder fehlen."""
        proc = MarketDataProcessor()
        raw = _candle_event()
        del raw["symbol"]

        with pytest.raises(ValueError, match="Missing"):
            proc.process_candle(raw)

    def test_process_candle_preserves_venue_metadata(self) -> None:
        """Behaelt Quell-Metadaten (venue, Instrument, Zeitgrenzen) bei."""
        proc = MarketDataProcessor()
        raw = _candle_event()
        raw["venue"] = "BINANCE_FUTURES"
        raw["instrument"] = "BTC/USDT"
        raw["open_time"] = "2026-09-01T12:00:00+00:00"
        raw["close_time"] = "2026-09-01T12:00:59.999000+00:00"
        raw["trade_count"] = 1252
        raw["is_closed"] = True

        result = proc.process_candle(raw)

        assert result["venue"] == "BINANCE_FUTURES"
        assert result["instrument"] == "BTC/USDT"
        assert result["open_time"] == "2026-09-01T12:00:00+00:00"
        assert result["close_time"] == "2026-09-01T12:00:59.999000+00:00"
        assert result["trade_count"] == 1252
        assert result["is_closed"] is True

    def test_process_candle_without_metadata_unchanged(self) -> None:
        """Events ohne Metadaten bleiben ohne Metadaten-Felder."""
        proc = MarketDataProcessor()
        result = proc.process_candle(_candle_event())

        assert "venue" not in result
        assert "instrument" not in result

    def test_high_low_swap(self) -> None:
        """Tauscht high/low wenn high < low."""
        proc = MarketDataProcessor()
        raw = _candle_event()
        raw["high"] = 145.0
        raw["low"] = 155.0

        result = proc.process_candle(raw)

        assert result["high"] == 155.0
        assert result["low"] == 145.0

    def test_volume_non_negative(self) -> None:
        """Validiert dass volume >= 0 ist."""
        proc = MarketDataProcessor()
        raw = _candle_event()
        raw["volume"] = -10.0

        with pytest.raises(ValueError, match=">= 0"):
            proc.process_candle(raw)

        raw_tick = _tick_event()
        raw_tick["volume"] = -5.0
        with pytest.raises(ValueError, match=">= 0"):
            proc.process_tick(raw_tick)


class TestDataIngestionService:
    """Testet DataIngestionService batch-Verarbeitung."""

    def test_process_batch(self) -> None:
        """Verarbeitet einen Batch von Events korrekt."""
        svc = DataIngestionService()
        events = [_candle_event(), _tick_event()]
        results = svc.process_batch(events)

        assert len(results) == 2
        assert results[0]["type"] == "candle"
        assert results[1]["type"] == "tick"
        assert len(svc.get_processed_stream()) == 2

    def test_process_batch_skips_invalid(self) -> None:
        """Überspringt ungültige Events im Batch."""
        svc = DataIngestionService()
        events = [_candle_event(), {"bad": "event"}]
        results = svc.process_batch(events)

        assert len(results) == 1
        assert results[0]["type"] == "candle"

    def test_validate_event_valid(self) -> None:
        """Gibt True für gültige Events zurück."""
        proc = MarketDataProcessor()
        assert proc.validate_event(_candle_event()) is True
        assert proc.validate_event(_tick_event()) is True

    def test_validate_event_invalid(self) -> None:
        """Gibt False für ungültige Events zurück."""
        proc = MarketDataProcessor()
        assert proc.validate_event({}) is False
        assert proc.validate_event({"symbol": "AAPL"}) is False
        assert proc.validate_event({"type": "tick"}) is False

    def test_ingestion_service_process_batch(self) -> None:
        """Integrations-Check: Batch → Stream Konsistenz."""
        svc = DataIngestionService()
        events = [_candle_event(), _candle_event(), _tick_event()]
        results = svc.process_batch(events)

        assert len(results) == 3
        stream = svc.get_processed_stream()
        assert len(stream) == 3
        assert all(r["symbol"] == "AAPL" for r in stream)
