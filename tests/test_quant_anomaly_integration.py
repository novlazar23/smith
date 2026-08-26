"""Anomaly-Integration (Quant-Plattform, Phase 3, P3-4).

End-to-End-Kette Detektion → Speicherung mit ausschließlich Mocks:
``AnomalyDetector`` (P3-1) → ``AnomalyStore`` (P3-2) → ``InfluxDBStore``
(P1-4). Kein Netzwerk, kein Docker, keine echte InfluxDB.

Das ``AnomalyStore`` wird parallel gebaut (P3-2, siehe
``docs/quant-platform-phase03-plan.md``). Solange
``quant/anomaly_store.py`` nicht importierbar ist, laufen diese Tests
gegen einen funktionalen Test-Double, der den Kontrakt der
Implementation (``detect_and_store(symbol, ...) -> AnomalyResult``,
``get_anomalies``, Schema-Fields plus Kontext ``timeframe``/``value``/
``threshold``) mit dem echten ``AnomalyDetector`` und den echten
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
from trading_harness.quant.anomaly_detection import AnomalyDetector

SYMBOL = "BTCUSDT"
EXCHANGE = "binance"
TIMEFRAME = "1m"
_NS = 1_000_000_000

try:  # P3-2 integriert → Tests laufen gegen die echte Implementierung.
    from trading_harness.quant.anomaly_store import AnomalyResult, AnomalyStore
except ImportError:  # P3-2 in Arbeit → funktionaler Double des Implementierungs-Kontrakts.

    @dataclass(frozen=True)
    class AnomalyResult:
        """Detect+Store-Ergebnis (Kontrakt P3-2, analog zu ``feature_store``)."""

        symbol: str
        timeframe: str
        exchange: str
        anomalies_found: int
        stored: bool

    def _is_number(value: object) -> bool:
        """True für int/float (ausgeschlossen: Bool)."""
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def _parse_epoch_ns(value: object) -> int | None:
        """ISO-8601-Zeitstempel einer Anomalie in Epoch-Nanosekunden (None bei Parse-Fehler)."""
        if not isinstance(value, str):
            return None
        try:
            moment = datetime.fromisoformat(value)
        except ValueError:
            return None
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        return int(moment.timestamp()) * _NS

    def _to_iso(value: object) -> str | None:
        """Query-Zeitstempel (ns oder datetime) als UTC-ISO-Zeichenkette."""
        if isinstance(value, datetime):
            moment = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
            return moment.astimezone(UTC).isoformat().replace("+00:00", "Z")
        if _is_number(value):
            moment = datetime.fromtimestamp(value / _NS, tz=UTC)
            return moment.isoformat().replace("+00:00", "Z")
        return None

    class AnomalyStore:
        """Funktionaler Double für ``quant/anomaly_store.AnomalyStore`` (Kontrakt P3-2).

        Echter ``AnomalyDetector`` + echte Schema-Konstanten; das Storage
        läuft über den (gemockten) InfluxDBStore: ein Punkt pro Anomalie
        im ``anomalies``-Measurement, Tags nach ``ANOMALY_TAGS``, Fields
        nach ``ANOMALY_FIELDS`` plus Kontext ``timeframe``/``value``/
        ``threshold``, Nanosekunden-Timestamp.
        """

        def __init__(self, store: Any, detector: AnomalyDetector | None = None) -> None:
            self._store = store
            self._detector = detector or AnomalyDetector()

        async def detect_and_store(
            self,
            symbol: str,
            timeframe: str,
            candles: list[dict],
            exchange: str = "binance",
        ) -> AnomalyResult:
            anomalies = self._detector.detect(candles)
            written = 0
            if self._store.is_available:
                for anomaly in anomalies:
                    anomaly.symbol = symbol
                    timestamp = _parse_epoch_ns(anomaly.timestamp)
                    if timestamp is None:
                        continue  # Punkt ohne Zeitstempel wird nie geschrieben
                    tags = {
                        "symbol": symbol,
                        "exchange": exchange,
                        "anomaly_type": anomaly.anomaly_type,
                    }
                    fields: dict[str, float | str] = {
                        "severity": anomaly.severity,
                        "feature": anomaly.feature,
                        "timeframe": timeframe,
                        "value": anomaly.value,
                        "threshold": anomaly.threshold,
                    }
                    if anomaly.zscore is not None:
                        fields["anomaly_score"] = anomaly.zscore
                    await self._store.write_points(
                        measurement=quant_schema.ANOMALY_MEASUREMENT,
                        tags=tags,
                        fields=fields,
                        timestamp=timestamp,
                    )
                    written += 1
            return AnomalyResult(
                symbol=symbol,
                timeframe=timeframe,
                exchange=exchange,
                anomalies_found=len(anomalies),
                stored=written > 0,
            )

        async def get_anomalies(
            self,
            symbol: str,
            timeframe: str,
            anomaly_type: str | None = None,
            start: str | None = None,
            end: str | None = None,
        ) -> list[dict]:
            """Liest Anomalien per Flux-Query (analog zu ``FeatureStore.get_features``)."""
            if not self._store.is_available:
                return []
            start_clause = f'"{start}"' if start is not None else "now() - 30d"
            range_clause = f"|> range(start: {start_clause}"
            range_clause += f', stop: "{end}")' if end is not None else ")"
            lines = [
                range_clause,
                f'|> filter(fn: (r) => r._measurement == "{quant_schema.ANOMALY_MEASUREMENT}")',
                f'|> filter(fn: (r) => r["symbol"] == "{symbol}")',
            ]
            if anomaly_type is not None:
                lines.append(f'|> filter(fn: (r) => r["anomaly_type"] == "{anomaly_type}")')
            lines.append('|> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")')
            lines.append('|> sort(columns: ["_time"])')
            rows = await self._store.query("\n".join(lines))
            result: list[dict] = []
            for record in rows:
                if str(record.get("timeframe")) != timeframe:
                    continue
                time_iso = _to_iso(record.get("_time"))
                if time_iso is None:
                    continue
                entry: dict[str, Any] = {
                    "time": time_iso,
                    "symbol": record.get("symbol"),
                    "exchange": record.get("exchange"),
                    "anomaly_type": record.get("anomaly_type"),
                    "timeframe": timeframe,
                }
                for name in quant_schema.ANOMALY_FIELDS:
                    value = record.get(name)
                    if _is_number(value):
                        entry[name] = value
                for name in ("feature", "value", "threshold"):
                    if name in record:
                        entry[name] = record[name]
                result.append(entry)
            return result


# ----------------------------------------------------------------------
# Test-Helfer
# ----------------------------------------------------------------------


def spike_candles() -> list[dict]:
    """25 flache Kerzen (close 100, volume 1000) + Spike-Kerze (close 200, volume 50000).

    Die flache Baseline (Varianz 0) erzeugt deterministisch je einen
    ``price_shock`` und einen ``volume_spike`` am Ende der Reihe.
    """
    prices = [100.0] * 25 + [200.0]
    volumes = [1000.0] * 25 + [50000.0]
    return [
        {
            "time": f"2026-01-01T00:{index:02d}:00Z",
            "open": price,
            "high": price * 1.01,
            "low": price * 0.99,
            "close": price,
            "volume": volume,
        }
        for index, (price, volume) in enumerate(zip(prices, volumes))
    ]


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
async def test_anomaly_detector_to_store_roundtrip() -> None:
    """Spike in der Kerzenreihe → Detector erkennt ihn → ein Punkt pro Anomalie wird geschrieben."""
    store = make_mock_store()
    anomaly_store = AnomalyStore(store)
    candles = spike_candles()
    expected = AnomalyDetector().detect(candles)
    assert {a.anomaly_type for a in expected} == {"price_shock", "volume_spike"}

    result = await anomaly_store.detect_and_store(SYMBOL, TIMEFRAME, candles, EXCHANGE)

    assert isinstance(result, AnomalyResult)
    assert result.stored is True
    assert result.anomalies_found == len(expected)
    assert store.write_points.await_count == len(expected)
    for call, anomaly in zip(store.write_points.await_args_list, expected, strict=True):
        point = point_of(call)
        assert point["measurement"] == quant_schema.ANOMALY_MEASUREMENT
        assert point["tags"]["symbol"] == SYMBOL
        assert point["tags"]["exchange"] == EXCHANGE
        assert point["tags"]["anomaly_type"] == anomaly.anomaly_type
        # Alle Schema-Fields vorhanden (beide Anomalien haben einen Z-Score).
        assert set(quant_schema.ANOMALY_FIELDS) <= set(point["fields"])


@pytest.mark.asyncio
async def test_anomaly_values_match_detector_output() -> None:
    """Gespeicherte Werte (Score, Severity, Feature, Typ, Timestamp) = Detector-Output, 1:1."""
    store = make_mock_store()
    anomaly_store = AnomalyStore(store)
    candles = spike_candles()
    expected = AnomalyDetector().detect(candles)

    await anomaly_store.detect_and_store(SYMBOL, TIMEFRAME, candles, EXCHANGE)

    points = [point_of(call) for call in store.write_points.await_args_list]
    assert len(points) == len(expected)
    for point, anomaly in zip(points, expected, strict=True):
        fields = point["fields"]
        assert point["tags"]["anomaly_type"] == anomaly.anomaly_type
        assert fields["severity"] == pytest.approx(anomaly.severity)
        assert fields["feature"] == anomaly.feature
        if anomaly.zscore is not None:
            assert fields["anomaly_score"] == pytest.approx(anomaly.zscore)
        else:
            assert "anomaly_score" not in fields  # nie eine halbe Zahl
        assert point["timestamp"] == int(datetime.fromisoformat(anomaly.timestamp).timestamp()) * _NS


@pytest.mark.asyncio
async def test_anomaly_store_unavailable_gracefully() -> None:
    """Store nicht verfügbar (is_available=False) → kein Write, keine Exception, stored=False."""
    store = make_mock_store(available=False)
    anomaly_store = AnomalyStore(store)

    result = await anomaly_store.detect_and_store(SYMBOL, TIMEFRAME, spike_candles(), EXCHANGE)

    assert result.stored is False
    assert result.anomalies_found >= 1  # Detektion läuft trotzdem
    store.write_points.assert_not_awaited()
    store.query.assert_not_awaited()


# ----------------------------------------------------------------------
# Schemakontrakt
# ----------------------------------------------------------------------


def test_anomaly_schema_constants() -> None:
    """ANOMALY_*-Konstanten sind untereinander und mit den Schema-Hilfsfunktionen konsistent."""
    assert quant_schema.ANOMALY_MEASUREMENT == "anomalies"
    assert quant_schema.ANOMALY_TAGS == ("symbol", "exchange", "anomaly_type")
    assert quant_schema.ANOMALY_FIELDS == ("anomaly_score", "severity", "feature")

    assert quant_schema.validate_measurement_name(quant_schema.ANOMALY_MEASUREMENT) is True
    info = quant_schema.get_measurement_info(quant_schema.ANOMALY_MEASUREMENT)
    assert info is not None
    assert info["tags"] == quant_schema.ANOMALY_TAGS
    assert info["fields"] == quant_schema.ANOMALY_FIELDS

    assert len(set(quant_schema.ANOMALY_TAGS)) == len(quant_schema.ANOMALY_TAGS)
    assert len(set(quant_schema.ANOMALY_FIELDS)) == len(quant_schema.ANOMALY_FIELDS)
    # Kein Tag/Field-Namekonflikt (InfluxDB: Tag und Field dürfen nicht kollidieren).
    assert set(quant_schema.ANOMALY_TAGS).isdisjoint(quant_schema.ANOMALY_FIELDS)
    tags = {name: "placeholder" for name in quant_schema.ANOMALY_TAGS}
    assert quant_schema.validate_tags(tags, quant_schema.ANOMALY_TAGS) == []
    fields = {name: 1.0 for name in quant_schema.ANOMALY_FIELDS}
    assert quant_schema.validate_fields(fields, quant_schema.ANOMALY_FIELDS) == []


@pytest.mark.asyncio
async def test_anomaly_get_anomalies_readback() -> None:
    """get_anomalies: Query liefert gespeicherte Punkte → normalisiertes list[dict];
    nicht verfügbarer Store → leere Liste ohne Query."""
    base_ns = int(datetime(2026, 1, 1, tzinfo=UTC).timestamp()) * _NS
    rows = [
        {
            "_time": base_ns,
            "symbol": SYMBOL,
            "exchange": EXCHANGE,
            "anomaly_type": "price_shock",
            "timeframe": TIMEFRAME,
            "anomaly_score": 6.0,
            "severity": 1.0,
            "feature": "close",
            "value": 0.6931471805599453,
            "threshold": 3.0,
        },
        {
            "_time": base_ns + 60_000_000_000,
            "symbol": SYMBOL,
            "exchange": EXCHANGE,
            "anomaly_type": "volume_spike",
            "timeframe": TIMEFRAME,
            "anomaly_score": 6.0,
            "severity": 1.0,
            "feature": "volume",
            "value": 50000.0,
            "threshold": 3.0,
        },
    ]
    store = make_mock_store()
    store.query = AsyncMock(return_value=rows)
    anomaly_store = AnomalyStore(store)

    entries = await anomaly_store.get_anomalies(SYMBOL, TIMEFRAME, None, None, None)

    assert store.query.await_count == 1
    flux = store.query.await_args.args[0]
    assert quant_schema.ANOMALY_MEASUREMENT in flux
    assert SYMBOL in flux
    assert len(entries) == 2
    for entry, row in zip(entries, rows, strict=True):
        assert entry["time"].endswith("Z")
        assert entry["symbol"] == SYMBOL
        assert entry["anomaly_type"] == row["anomaly_type"]
        assert entry["timeframe"] == TIMEFRAME
        assert entry["severity"] == pytest.approx(row["severity"])
        assert entry["anomaly_score"] == pytest.approx(row["anomaly_score"])
        assert entry["feature"] == row["feature"]

    # Store nicht verfügbar → leere Liste ohne Query.
    offline_store = make_mock_store(available=False)
    offline = AnomalyStore(offline_store)
    assert await offline.get_anomalies(SYMBOL, TIMEFRAME) == []
    offline_store.query.assert_not_awaited()
