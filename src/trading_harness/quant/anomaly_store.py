"""Anomaly-Storage für die Quant-Plattform (Phase 3, P3-2).

Verbindet ``AnomalyDetector`` (P3-1) mit ``InfluxDBStore`` (P1-4):
erkannte Anomalien werden je Fund als ein Punkt in das ``anomalies``-
Measurement geschrieben und per Flux-Query zurückgelesen.

Schemakontrakt (``quant/schema.py``):

- Measurement: ``anomalies``
- Tags: ``symbol``, ``exchange``, ``anomaly_type``
- Fields: ``anomaly_score``, ``severity``, ``feature`` plus Kontext
  ``timeframe``, ``value``, ``threshold``

Semantik:

- Pro erkannte Anomalie ein Punkt (kein ``write_batch``: der Tag
  ``anomaly_type`` variiert je Anomalie und kann nicht geteilt werden).
- ``AnomalyDetector.detect`` liefert Anomalien mit leerem ``symbol``;
  der Store setzt den ``symbol``-Parameter auf jede Anomalie, bevor der
  Punkt gebaut wird.
- ``anomaly_score`` trägt den Z-Score des Detektors; Anomalien ohne
  Z-Score (``volatility_outlier``, IQR-basiert) werden ohne dieses
  Field geschrieben — nie eine halbe Zahl (Muster: ``feature_store``).
- Kerzen mit unparsebarem ``time`` erzeugen keine Punkte (Muster:
  ``feature_store`` / ``ohlcv_ingestion``).
- Store nicht verfügbar (``is_available=False``): kein Schreibversuch
  (``stored=False``), ``get_anomalies`` liefert eine leere Liste.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, TypeGuard

from trading_harness.quant import schema as quant_schema
from trading_harness.quant.anomaly_detection import Anomaly, AnomalyDetector
from trading_harness.quant.influxdb_client import FieldDict, InfluxDBStore

_NANOSECONDS_PER_SECOND: int = 1_000_000_000

# Rückblickfenster für get_anomalies ohne Startzeitpunkt (Flux-Dauer, keine Magie).
_DEFAULT_LOOKBACK: str = "now() - 30d"


def _is_number(value: object) -> TypeGuard[int | float]:
    """True für int/float (ausgeschlossen: Bool, das in Python ein int ist)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _parse_timestamp(value: object) -> int | None:
    """Parst einen Kerzen-Zeitstempel (datetime oder ISO-8601) zu Epoch-Sekunden.

    Naive Zeiten werden als UTC interpretiert; ungültige Werte liefern None.
    """
    if isinstance(value, datetime):
        moment = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return int(moment.timestamp())
    if isinstance(value, str):
        try:
            moment = datetime.fromisoformat(value)
        except ValueError:
            return None
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        return int(moment.timestamp())
    return None


def _timestamp_to_iso(value: object) -> str | None:
    """Formatiert einen Query-Zeitstempel (datetime oder ns) als UTC-ISO-String."""
    if isinstance(value, datetime):
        moment = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return moment.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if _is_number(value):
        moment = datetime.fromtimestamp(value / _NANOSECONDS_PER_SECOND, tz=UTC)
        return moment.isoformat().replace("+00:00", "Z")
    return None


def _flux_escape(value: str) -> str:
    """Escape für String-Literale in Flux-Queries."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _anomaly_fields(anomaly: Anomaly, timeframe: str) -> FieldDict:
    """Mappt eine Anomalie auf ihre InfluxDB-Fields (Kontext ``timeframe``)."""
    fields: FieldDict = {
        "severity": anomaly.severity,
        "feature": anomaly.feature,
        "timeframe": timeframe,
        "value": anomaly.value,
        "threshold": anomaly.threshold,
    }
    if anomaly.zscore is not None:
        fields["anomaly_score"] = anomaly.zscore
    return fields


@dataclass(frozen=True)
class AnomalyResult:
    """Ergebnis eines Detect+Store-Laufs.

    ``anomalies_found`` ist die Anzahl erkannter Anomalien (unabhängig
    davon, ob der Store verfügbar war); ``stored`` ist True nur, wenn
    mindestens ein Punkt in einen verfügbaren Store geschrieben wurde.
    """

    symbol: str
    timeframe: str
    exchange: str
    anomalies_found: int
    stored: bool


class AnomalyStore:
    """Schreibt erkannte Anomalien in InfluxDB und liest sie zurück.

    - Ein Punkt pro Anomalie im ``anomalies``-Measurement; Tags
      ``symbol`` / ``exchange`` / ``anomaly_type``, Fields laut Schema.
    - ``bucket`` wird aus der Store-Konfiguration gelesen
      (``store._bucket``), damit Queries ohne zusätzliche Konfiguration
      laufen (Muster: ``FeatureStore``).
    - Alle Store-Zugriffe laufen asynchron (``InfluxDBStore`` ist async).
    """

    def __init__(
        self,
        store: InfluxDBStore,
        detector: AnomalyDetector | None = None,
    ) -> None:
        """Initialisiert den AnomalyStore auf dem gegebenen Store (lazy, keine Verbindung)."""
        self._store = store
        self._detector = detector or AnomalyDetector()
        self._bucket = str(getattr(store, "_bucket", "quant"))

    # ------------------------------------------------------------------
    # Schreibpfad
    # ------------------------------------------------------------------

    async def detect_and_store(
        self,
        symbol: str,
        timeframe: str,
        candles: list[dict],
        exchange: str = "binance",
    ) -> AnomalyResult:
        """Erkennt Anomalien und schreibt sie in InfluxDB.

        Pro erkannter Anomalie entsteht ein Punkt im ``anomalies``-
        Measurement; der ``symbol``-Parameter wird auf jede Anomalie
        gesetzt, bevor der Punkt gebaut wird. Kerzen mit unparsebarem
        ``time`` erzeugen keine Punkte. Bei nicht verfügbarem Store
        (``is_available=False``) wird nicht geschrieben: ``stored=False``.
        """
        anomalies = self._detector.detect(candles)
        points: list[tuple[int, dict[str, str], FieldDict]] = []
        for anomaly in anomalies:
            anomaly.symbol = symbol
            epoch = _parse_timestamp(anomaly.timestamp)
            if epoch is None:
                continue  # strukturell ungültiger Zeitstempel → überspringen
            tags = {
                "symbol": symbol,
                "exchange": exchange,
                "anomaly_type": anomaly.anomaly_type,
            }
            points.append(
                (epoch * _NANOSECONDS_PER_SECOND, tags, _anomaly_fields(anomaly, timeframe))
            )

        written = 0
        if self._store.is_available:
            for timestamp_ns, tags, fields in points:
                await self._store.write_points(
                    measurement=quant_schema.ANOMALY_MEASUREMENT,
                    tags=tags,
                    fields=fields,
                    timestamp=timestamp_ns,
                )
                written += 1

        return AnomalyResult(
            symbol=symbol,
            timeframe=timeframe,
            exchange=exchange,
            anomalies_found=len(anomalies),
            stored=written > 0,
        )

    # ------------------------------------------------------------------
    # Lesezugriff (Flux-Query über InfluxDBStore.query)
    # ------------------------------------------------------------------

    async def get_anomalies(
        self,
        symbol: str,
        timeframe: str,
        anomaly_type: str | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> list[dict]:
        """Liest Anomalien aus InfluxDB (ein Dict pro Punkt, aufsteigend).

        Jeder Eintrag enthält ``time`` (UTC-ISO-String), ``symbol``,
        ``exchange``, ``anomaly_type``, ``timeframe`` sowie die
        vorhandenen Anomalie-Felder. ``start``/``end`` sind
        UTC-ISO-Zeichenketten (Default-Rückblick ohne ``start``: 30
        Tage). Bei nicht verfügbarem Store (``is_available=False``)
        leere Liste; ungültige Zeitstempel werfen ``ValueError``.
        """
        if not self._store.is_available:
            return []
        if start is not None and _parse_timestamp(start) is None:
            raise ValueError(f"invalid start timestamp: {start!r}")
        if end is not None and _parse_timestamp(end) is None:
            raise ValueError(f"invalid end timestamp: {end!r}")
        flux = self._build_query(symbol, anomaly_type, start, end)
        rows = await self._store.query(flux)

        result: list[dict] = []
        for record in rows:
            # timeframe ist ein Field (kein Tag) → Filterung nach dem Pivot.
            if str(record.get("timeframe")) != timeframe:
                continue
            time_iso = _timestamp_to_iso(record.get("_time"))
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

    # ------------------------------------------------------------------
    # Query-Bau (privat)
    # ------------------------------------------------------------------

    def _build_query(
        self,
        symbol: str,
        anomaly_type: str | None,
        start: str | None,
        end: str | None,
    ) -> str:
        """Flux: Anomalien im Zeitraum, gepivotet, aufsteigend sortiert.

        Bei ``anomaly_type`` wird zusätzlich auf den Tag gefiltert
        (``anomaly_type`` ist Teil von ``schema.ANOMALY_TAGS``).
        """
        start_clause = f'"{_flux_escape(start)}"' if start is not None else _DEFAULT_LOOKBACK
        range_clause = f"range(start: {start_clause}"
        if end is not None:
            range_clause += f', stop: "{_flux_escape(end)}"'
        range_clause += ")"
        lines = [
            f'from(bucket: "{_flux_escape(self._bucket)}")',
            f"|> {range_clause}",
            f'|> filter(fn: (r) => r._measurement == "{quant_schema.ANOMALY_MEASUREMENT}")',
            f'|> filter(fn: (r) => r["symbol"] == "{_flux_escape(symbol)}")',
        ]
        if anomaly_type is not None:
            lines.append(
                f'|> filter(fn: (r) => r["anomaly_type"] == "{_flux_escape(anomaly_type)}")'
            )
        lines.append('|> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")')
        lines.append('|> sort(columns: ["_time"])')
        return "\n".join(lines)
