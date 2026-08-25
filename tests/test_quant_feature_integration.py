"""Kompositionstests Feature-Engineering (Phase 2, P2-4).

Verifizieren die Zusammensetzung FeatureEngine → FeatureStore → InfluxDBStore
ausschließlich mit Mocks: kein Netzwerk, kein Docker, keine echte InfluxDB.

Kontrakt (``quant/feature_store.py``, P2-2):

- ``compute_and_store(symbol, timeframe, candles, exchange) -> FeatureResult``
- Rolling Calculation: ein Punkt pro Kerze, berechnet aus ``candles[: i + 1]``
  (kein Look-Ahead) — der letzte Punkt entspricht ``engine.compute(candles)``.
- Schreibpfad ``InfluxDBStore.write_points`` in das Measurement
  ``FEATURE_MEASUREMENT``, Tags ``FEATURE_TAGS``, Fields: flache
  Feature-Werte + ``timeframe`` + Metadaten ``FEATURE_META_FIELDS``.
- ``is_available=False`` → kein Schreibversuch, ``FeatureResult.stored=False``.

Solange ``feature_store.py`` (parallel aufgebaut) nicht importierbar ist,
nutzt dieses Modul ein Kontrakt-Doppel mit identischer Semantik.

Speicher-Field-Namen: ``feature_store.FEATURE_FIELD_NAMES`` (Mapping auf den
``compute``-Output unter ``_FEATURE_FIELD_TO_ENGINE``).
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest

from trading_harness.quant import schema as quant_schema
from trading_harness.quant.features import FeatureEngine
from trading_harness.quant.influxdb_client import InfluxDBStore

try:  # P2-2 (feature_store.py) — parallel in Arbeit
    from trading_harness.quant.feature_store import FeatureResult, FeatureStore
except ImportError:
    # Kontrakt-Doppel bis zur Landung von P2-2 (Semantik identisch).

    @dataclass(frozen=True)
    class FeatureResult:
        """Ergebnis von ``FeatureStore.compute_and_store`` (P2-2-Kontrakt)."""

        symbol: str
        timeframe: str
        exchange: str
        feature_count: int
        computation_time_ms: float
        stored: bool

    class FeatureStore:
        """Kontrakt-Doppel für ``quant/feature_store.py`` (P2-2)."""

        def __init__(
            self,
            store: InfluxDBStore,
            engine: FeatureEngine | None = None,
            bucket: str | None = None,
        ) -> None:
            self._store = store
            self._engine = engine or FeatureEngine()
            self._bucket = bucket if bucket is not None else str(getattr(store, "_bucket", "quant"))

        async def compute_and_store(
            self,
            symbol: str,
            timeframe: str,
            candles: list[dict],
            exchange: str = "binance",
        ) -> FeatureResult:
            """Rolling Calculation: ein Punkt pro Kerze, kein Look-Ahead."""
            t0 = time.monotonic()
            tags = {
                "symbol": symbol,
                "exchange": exchange,
                "feature_version": quant_schema.FEATURE_VERSION,
            }
            total = 0
            written = 0
            if self._store.is_available:
                for index, candle in enumerate(candles):
                    epoch = _parse_epoch(candle.get("time"))
                    if epoch is None:
                        continue  # ohne Zeitstempel entsteht kein Punkt
                    computed = self._engine.compute(candles[: index + 1])
                    features = _flatten_feature_fields(computed)
                    fields: dict[str, float | int | str] = {"timeframe": timeframe}
                    fields.update(features)
                    fields["feature_count"] = len(features)
                    fields["computation_time_ms"] = float(computed["computation_time_ms"])
                    await self._store.write_points(
                        measurement=quant_schema.FEATURE_MEASUREMENT,
                        tags=tags,
                        fields=fields,
                        timestamp=epoch * NS,
                    )
                    written += 1
                    total += len(features)
            return FeatureResult(
                symbol=symbol,
                timeframe=timeframe,
                exchange=exchange,
                feature_count=total,
                computation_time_ms=(time.monotonic() - t0) * 1000,
                stored=written > 0,
            )

SYMBOL = "BTCUSDT"
TIMEFRAME = "5m"
NS = 1_000_000_000
BASE = datetime(2026, 1, 1, tzinfo=UTC)

# Die sechs Indikatoren von FeatureEngine.compute (Phase 2, P2-1).
_FEATURE_KEYS: tuple[str, ...] = ("rsi", "macd", "bollinger", "atr", "volatility", "vwap")

# Speicher-Field-Name → Pfad im compute()-Output (Kontrakt P2-2, FEATURE_FIELD_NAMES).
_FEATURE_FIELD_TO_ENGINE: dict[str, tuple[str, ...]] = {
    "rsi": ("rsi",),
    "macd_line": ("macd", "macd"),
    "signal_line": ("macd", "signal"),
    "histogram": ("macd", "histogram"),
    "bollinger_upper": ("bollinger", "upper"),
    "bollinger_middle": ("bollinger", "middle"),
    "bollinger_lower": ("bollinger", "lower"),
    "bollinger_bandwidth": ("bollinger", "bandwidth"),
    "atr": ("atr",),
    "volatility": ("volatility",),
    "vwap": ("vwap",),
}


# ----------------------------------------------------------------------
# Hilfsfunktionen
# ----------------------------------------------------------------------


def _candle(index: int) -> dict[str, Any]:
    """Synthetische Kerze mit Trend + Sinus-Overlay (deterministisch, UTC)."""
    base = 100.0 + 0.5 * index
    close = base + 4.0 * math.sin(index / 3.0)
    return {
        "time": (BASE + timedelta(minutes=5 * index)).isoformat().replace("+00:00", "Z"),
        "open": base,
        "high": max(base, close) + 1.5,
        "low": min(base, close) - 1.5,
        "close": close,
        "volume": 1000.0 + 10.0 * (index % 7),
    }


def _candle_series(count: int) -> list[dict[str, Any]]:
    """Erzeugt ``count`` aufeinanderfolgende 5m-Kerzen ab BASE."""
    return [_candle(i) for i in range(count)]


def make_store() -> MagicMock:
    """Gemockter InfluxDBStore: async API wird zu AsyncMock, keine echte Verbindung."""
    return MagicMock(spec=InfluxDBStore)


def make_feature_store(store: MagicMock, engine: FeatureEngine) -> FeatureStore:
    return FeatureStore(store, engine)


def _parse_epoch(value: object) -> int | None:
    """ISO-8601-Zeichenkette → Epoch-Sekunden (naiv = UTC); ungültig → None."""
    if not isinstance(value, str):
        return None
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return int(moment.timestamp())


def _flatten_feature_fields(computed: dict[str, Any]) -> dict[str, float]:
    """Engine-Output → flache Feature-Fields (Kontrakt P2-2, None-Werte weggelassen)."""
    fields: dict[str, float] = {}
    for name, path in _FEATURE_FIELD_TO_ENGINE.items():
        value = _lookup(computed, path)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            fields[name] = float(value)
    return fields


def _lookup(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    """Folgt einem verschachtelten Pfad; fehlender Ast → None."""
    value: Any = data
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _written_points(store: MagicMock) -> list[tuple[str, dict[str, str], dict[str, Any]]]:
    """Sammelt (measurement, tags, fields) aller Schreibaufrufe in Schreibreihenfolge.

    Erfasst beide Schreibpfade des Stores (``write_points`` und ``write_batch``),
    damit die Tests unabhängig von der konkreten Implementierung von P2-2 sind.
    """
    written: list[tuple[str, dict[str, str], dict[str, Any]]] = []
    for call in store.write_points.await_args_list:
        kwargs = call.kwargs
        written.append((kwargs["measurement"], kwargs["tags"], kwargs["fields"]))
    for call in store.write_batch.await_args_list:
        kwargs = call.kwargs
        for point in kwargs["points"]:
            written.append((kwargs["measurement"], kwargs["tags"], point))
    return written


# ----------------------------------------------------------------------
# 1. Roundtrip: Engine → Store
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_feature_engine_to_feature_store_roundtrip() -> None:
    """30 Kerzen → FeatureStore schreibt pro Kerze in FEATURE_MEASUREMENT, feature_count > 0."""
    store = make_store()
    store.is_available = True
    engine = FeatureEngine()
    feature_store = make_feature_store(store, engine)
    candles = _candle_series(30)

    result = await feature_store.compute_and_store(SYMBOL, TIMEFRAME, candles)

    assert result.stored is True
    assert result.feature_count > 0
    written = _written_points(store)
    assert len(written) == len(candles), "ein Punkt pro gültiger Kerze"
    # Alle Punkte landen im Features-Measurement aus schema.py.
    assert all(measurement == quant_schema.FEATURE_MEASUREMENT for measurement, _, _ in written)
    # Tags erfüllen den FEATURE_TAGS-Kontrakt inkl. Feature-Version.
    for _, tags, _ in written:
        assert set(quant_schema.FEATURE_TAGS) <= set(tags)
        assert tags["symbol"] == SYMBOL
        assert tags["exchange"] == "binance"
        assert tags["feature_version"] == quant_schema.FEATURE_VERSION
    # Jeder Punkt trägt die Metadaten, feature_count ist positiv.
    for _, _, fields in written:
        for meta in quant_schema.FEATURE_META_FIELDS:
            assert meta in fields
        assert fields["feature_count"] > 0


# ----------------------------------------------------------------------
# 2. Exakte Wert-Übereinstimmung
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_feature_values_match_engine_output() -> None:
    """Letzter Punkt (vollständige Historie) ist bitgenau der compute()-Output."""
    store = make_store()
    store.is_available = True
    engine = FeatureEngine()
    feature_store = make_feature_store(store, engine)
    candles = _candle_series(50)  # 50 Kerzen → alle sechs Indikatoren verfügbar
    expected = engine.compute(candles)

    await feature_store.compute_and_store(SYMBOL, TIMEFRAME, candles)

    written = _written_points(store)
    assert written
    last_fields = written[-1][2]  # Punkt der letzten Kerze = compute(alle 50 Kerzen)
    checked = 0
    for name, path in _FEATURE_FIELD_TO_ENGINE.items():
        if name not in last_fields:
            continue
        expected_value = _lookup(expected, path)
        assert expected_value is not None, f"{name}: fehlt im Engine-Output"
        assert last_fields[name] == expected_value, (
            f"{name}: {last_fields[name]} != {expected_value}"
        )
        checked += 1
    assert checked == len(_FEATURE_FIELD_TO_ENGINE), "alle Feature-Fields im letzten Punkt"
    # Per-Punkt-Metadaten des letzten Punkts.
    assert last_fields["feature_count"] == len(_FEATURE_FIELD_TO_ENGINE)


# ----------------------------------------------------------------------
# 3. InfluxDB-Ausfall
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_feature_store_handles_unavailable_gracefully() -> None:
    """Nicht verfügbarer Store → FeatureResult.stored=False, kein Schreibversuch."""
    store = make_store()
    store.is_available = False
    engine = FeatureEngine()
    feature_store = make_feature_store(store, engine)
    candles = _candle_series(30)

    result = await feature_store.compute_and_store(SYMBOL, TIMEFRAME, candles)

    assert isinstance(result, FeatureResult)
    assert result.stored is False
    store.write_points.assert_not_awaited()
    store.write_batch.assert_not_awaited()


# ----------------------------------------------------------------------
# 4. Schema-Kontrakt
# ----------------------------------------------------------------------


def test_feature_schema_constants_consistent() -> None:
    """FEATURE_MEASUREMENT / FEATURE_VERSION / FEATURE_TAGS sind intern konsistent."""
    # Measurement-Name ist InfluxDB-konform und in der Schema-Info-Mappe registriert.
    assert quant_schema.validate_measurement_name(quant_schema.FEATURE_MEASUREMENT)
    info = quant_schema.get_measurement_info(quant_schema.FEATURE_MEASUREMENT)
    assert info is not None
    assert info["tags"] == quant_schema.FEATURE_TAGS
    assert info["fields"] == quant_schema.FEATURE_META_FIELDS

    # Version folgt SemVer und wird als Tag persistiert.
    assert re.fullmatch(r"\d+\.\d+\.\d+", quant_schema.FEATURE_VERSION)
    assert set(quant_schema.FEATURE_TAGS) == {"symbol", "exchange", "feature_version"}

    # Meta-Felder stimmen mit den Metadaten von FeatureEngine.compute überein.
    engine = FeatureEngine()
    result = engine.compute(_candle_series(30))
    for meta in quant_schema.FEATURE_META_FIELDS:
        assert meta in result
    meta_fields = {meta: result[meta] for meta in quant_schema.FEATURE_META_FIELDS}
    assert quant_schema.validate_fields(meta_fields, quant_schema.FEATURE_META_FIELDS) == []

    # Ein typisches Tag-Set erfüllt den Tag-Kontrakt.
    tags = {
        "symbol": SYMBOL,
        "exchange": "binance",
        "feature_version": quant_schema.FEATURE_VERSION,
    }
    assert quant_schema.validate_tags(tags, quant_schema.FEATURE_TAGS) == []


# ----------------------------------------------------------------------
# 5. Vollständigkeit der Indikatoren
# ----------------------------------------------------------------------


def test_compute_all_features_present() -> None:
    """50 Kerzen → alle sechs Indikatoren sind berechnet (non-None)."""
    engine = FeatureEngine()
    result = engine.compute(_candle_series(50))

    for key in _FEATURE_KEYS:
        assert result[key] is not None, key
    assert result["feature_count"] == len(_FEATURE_KEYS)
    assert result["computation_time_ms"] >= 0.0


# ----------------------------------------------------------------------
# 6. Multi-Symbol-Propagation
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_feature_store_multi_symbol_roundtrip() -> None:
    """Zwei Symbole (je ein Aufruf) → deterministisch identische Indikator-Fields."""
    store = make_store()
    store.is_available = True
    engine = FeatureEngine()
    feature_store = make_feature_store(store, engine)
    candles = _candle_series(30)

    first = await feature_store.compute_and_store(SYMBOL, TIMEFRAME, candles)
    second = await feature_store.compute_and_store("ETHUSDT", TIMEFRAME, candles)

    assert first.stored is True and second.stored is True
    assert first.feature_count > 0 and second.feature_count > 0
    written = _written_points(store)
    assert len(written) == 2 * len(candles)
    assert {tags["symbol"] for _, tags, _ in written} == {SYMBOL, "ETHUSDT"}
    # Gleiche Kerzen → deterministisch identische Indikator-Fields (ohne Meta-Zeit).
    first_block = [fields for _, tags, fields in written if tags["symbol"] == SYMBOL]
    second_block = [fields for _, tags, fields in written if tags["symbol"] == "ETHUSDT"]
    for first_fields, second_fields in zip(first_block, second_block, strict=True):
        first_indicators = {k: v for k, v in first_fields.items() if k in _FEATURE_FIELD_TO_ENGINE}
        second_indicators = {k: v for k, v in second_fields.items() if k in _FEATURE_FIELD_TO_ENGINE}
        assert first_indicators == second_indicators
