"""Regime-Integration (Quant-Plattform, Phase 4, P4-4).

End-to-End-Kette Detektion → Speicherung mit ausschließlich Mocks:
``RegimeDetector`` (P4-1) → ``RegimeStore`` (P4-2) → ``InfluxDBStore``
(P1-4). Kein Netzwerk, kein Docker, keine echte InfluxDB.

Das ``RegimeStore`` wird parallel gebaut (P4-2, siehe
``docs/quant-platform-phase04-plan.md``). Solange
``quant/regime_store.py`` nicht importierbar ist, laufen diese Tests
gegen einen funktionalen Test-Double, der den Kontrakt der
Implementation (``detect_and_store(symbol, timeframe, candles,
exchange=...) -> RegimeStoreResult`` mit ``regime``/``confidence``/
``duration``/``stored``, Schema-Fields ``REGIME_FIELDS``, Tags
``REGIME_TAGS``) mit dem echten ``RegimeDetector`` und den echten
Schema-Konstanten nachbildet. Ist das echte Modul vorhanden, greifen
dieselben Tests unverändert darauf zu.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from trading_harness.quant import schema as quant_schema
from trading_harness.quant.regime_detection import REGIME_NAMES, RegimeDetector

SYMBOL = "BTCUSDT"
EXCHANGE = "binance"
TIMEFRAME = "1h"
_NS = 1_000_000_000

try:  # P4-2 integriert → Tests laufen gegen die echte Implementierung.
    from trading_harness.quant.regime_store import RegimeStore
except ImportError:  # P4-2 in Arbeit → funktionaler Double des Implementierungs-Kontrakts.

    @dataclass(frozen=True)
    class RegimeStoreResult:
        """Detect+Store-Ergebnis (Kontrakt P4-2, analog zu ``AnomalyResult``)."""

        symbol: str
        timeframe: str
        exchange: str
        regime: str
        confidence: float
        duration: int
        stored: bool

    def _parse_epoch_ns(value: object) -> int | None:
        """ISO-8601-Zeitstempel einer Kerze in Epoch-Nanosekunden (None bei Parse-Fehler)."""
        if not isinstance(value, str):
            return None
        try:
            moment = datetime.fromisoformat(value)
        except ValueError:
            return None
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        return int(moment.timestamp()) * _NS

    class RegimeStore:
        """Funktionaler Double für ``quant/regime_store.RegimeStore`` (Kontrakt P4-2).

        Echter ``RegimeDetector`` + echte Schema-Konstanten; das Storage
        läuft über den (gemockten) InfluxDBStore: genau ein Punkt pro
        Detect+Store-Lauf im ``regime``-Measurement, Tags nach
        ``REGIME_TAGS``, Fields nach ``REGIME_FIELDS``, Timestamp der
        letzten Kerze in Nanosekunden.
        """

        def __init__(self, store: Any, detector: RegimeDetector | None = None) -> None:
            self._store = store
            self._detector = detector or RegimeDetector()

        async def detect_and_store(
            self,
            symbol: str,
            timeframe: str,
            candles: list[dict],
            exchange: str = "binance",
        ) -> RegimeStoreResult:
            detected = self._detector.detect(candles)
            stored = False
            if self._store.is_available and candles:
                timestamp = _parse_epoch_ns(candles[-1].get("time"))
                if timestamp is not None:
                    tags = {
                        "symbol": symbol,
                        "exchange": exchange,
                    }
                    fields: dict[str, float | str | int] = {
                        "regime_name": detected.regime,
                        "regime_confidence": detected.confidence,
                        "regime_duration": detected.duration,
                    }
                    await self._store.write_points(
                        measurement=quant_schema.REGIME_MEASUREMENT,
                        tags=tags,
                        fields=fields,
                        timestamp=timestamp,
                    )
                    stored = True
            return RegimeStoreResult(
                symbol=symbol,
                timeframe=timeframe,
                exchange=exchange,
                regime=detected.regime,
                confidence=detected.confidence,
                duration=detected.duration,
                stored=stored,
            )


# ----------------------------------------------------------------------
# Test-Helfer
# ----------------------------------------------------------------------


def uptrend_candles() -> list[dict]:
    """80 saubere Aufwärtskerzen (close steigt linear um 0.5 pro Kerze).

    open = Vor-Close, high = close, low = open: deterministisch
    ``strong_bull`` (ADX ≈ 100, SMA-Crossover bullisch, Volatilität
    normal, kein Crash/Recovery).
    """
    base = 100.0
    step = 0.5
    candles: list[dict] = []
    previous_close = base
    for index in range(80):
        close = base + step * index
        candles.append(
            {
                "time": f"2026-01-01T{index // 60:02d}:{index % 60:02d}:00Z",
                "open": previous_close,
                "high": close,
                "low": previous_close,
                "close": close,
                "volume": 1000.0,
            }
        )
        previous_close = close
    return candles


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


# ----------------------------------------------------------------------
# Integration: Detector → Store → (mockte) InfluxDB
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_regime_detector_to_store_roundtrip() -> None:
    """Uptrend → Detector erkennt Regime → genau ein Punkt wird in InfluxDB geschrieben."""
    store = make_mock_store()
    regime_store = RegimeStore(store)
    candles = uptrend_candles()
    expected = RegimeDetector().detect(candles)
    assert expected.regime in REGIME_NAMES

    result = await regime_store.detect_and_store(SYMBOL, TIMEFRAME, candles, EXCHANGE)

    assert result.stored is True
    store.write_points.assert_awaited_once()
    point = point_of(store.write_points.await_args)
    assert point["measurement"] == quant_schema.REGIME_MEASUREMENT
    assert point["tags"]["symbol"] == SYMBOL
    assert point["tags"]["exchange"] == EXCHANGE
    # Alle Schema-Fields vorhanden und mit den REGIME_TAGS konsistent.
    assert set(quant_schema.REGIME_FIELDS) <= set(point["fields"])
    assert set(point["tags"]) == set(quant_schema.REGIME_TAGS)
    # Timestamp der letzten Kerze (Nanosekunden, UTC).
    assert point["timestamp"] == (
        int(datetime.fromisoformat(candles[-1]["time"]).timestamp()) * _NS
    )


@pytest.mark.asyncio
async def test_regime_values_match_detector() -> None:
    """Gespeicherte Werte (Name, Confidence, Duration) = RegimeDetector-Output, 1:1."""
    store = make_mock_store()
    regime_store = RegimeStore(store)
    candles = uptrend_candles()
    expected = RegimeDetector().detect(candles)

    await regime_store.detect_and_store(SYMBOL, TIMEFRAME, candles, EXCHANGE)

    fields = point_of(store.write_points.await_args)["fields"]
    assert fields["regime_name"] == expected.regime
    assert fields["regime_confidence"] == pytest.approx(expected.confidence)
    assert fields["regime_duration"] == expected.duration
    # Der Store darf den Detektor nicht uminterpretieren: gleiches Ergebnis wie Direktcall.
    assert fields["regime_name"] == RegimeDetector().detect(candles).regime


@pytest.mark.asyncio
async def test_regime_store_unavailable() -> None:
    """Store nicht verfügbar (is_available=False) → kein Write, keine Exception, stored=False."""
    store = make_mock_store(available=False)
    regime_store = RegimeStore(store)

    result = await regime_store.detect_and_store(SYMBOL, TIMEFRAME, uptrend_candles(), EXCHANGE)

    assert result.stored is False
    assert result.regime in REGIME_NAMES  # Detektion läuft trotzdem
    store.write_points.assert_not_awaited()
    store.query.assert_not_awaited()


# ----------------------------------------------------------------------
# Schemakontrakt
# ----------------------------------------------------------------------


def test_regime_schema_constants() -> None:
    """REGIME_*-Konstanten sind untereinander und mit den Schema-Hilfsfunktionen konsistent."""
    assert quant_schema.REGIME_MEASUREMENT == "regime"
    assert quant_schema.REGIME_TAGS == ("symbol", "exchange")
    assert quant_schema.REGIME_FIELDS == ("regime_name", "regime_confidence", "regime_duration")

    assert quant_schema.validate_measurement_name(quant_schema.REGIME_MEASUREMENT) is True
    info = quant_schema.get_measurement_info(quant_schema.REGIME_MEASUREMENT)
    assert info is not None
    assert info["tags"] == quant_schema.REGIME_TAGS
    assert info["fields"] == quant_schema.REGIME_FIELDS

    assert len(set(quant_schema.REGIME_TAGS)) == len(quant_schema.REGIME_TAGS)
    assert len(set(quant_schema.REGIME_FIELDS)) == len(quant_schema.REGIME_FIELDS)
    # Kein Tag/Field-Namekonflikt (InfluxDB: Tag und Field dürfen nicht kollidieren).
    assert set(quant_schema.REGIME_TAGS).isdisjoint(quant_schema.REGIME_FIELDS)
    tags = {name: "placeholder" for name in quant_schema.REGIME_TAGS}
    assert quant_schema.validate_tags(tags, quant_schema.REGIME_TAGS) == []
    fields = {name: 1.0 for name in quant_schema.REGIME_FIELDS}
    assert quant_schema.validate_fields(fields, quant_schema.REGIME_FIELDS) == []
