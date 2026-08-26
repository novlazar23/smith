"""Regime-Storage für die Quant-Plattform (Phase 4, P4-2).

Verbindet ``RegimeDetector`` (P4-1) mit ``InfluxDBStore`` (P1-4):
das erkannte Marktregime wird als ein Punkt in das ``regime``-Measurement
geschrieben und per Flux-Query als Historie zurückgelesen.

Schemakontrakt (``quant/schema.py``):

- Measurement: ``regime``
- Tags: ``symbol``, ``exchange``
- Fields: ``regime_name``, ``regime_confidence``, ``regime_duration``
  plus Kontext ``timeframe``

Semantik:

- Ein Punkt pro Detect+Store-Lauf — das aktuelle Regime (für eine
  Rolling-Serie steht ``RegimeDetector.detect_series`` bereit).
- Zeitstempel des Punkts: Zeit der letzten Kerze mit parsebarem
  ``time`` (UTC); ohne Kerzen der aktuelle Zeitpunkt (Muster:
  ``AnomalyStore`` / ``FeatureStore``).
- Store nicht verfügbar (``is_available=False``): kein Schreibversuch
  (``stored=False``), ``get_regime`` liefert eine leere Liste.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, TypeGuard

from trading_harness.quant import schema as quant_schema
from trading_harness.quant.influxdb_client import FieldDict, InfluxDBStore
from trading_harness.quant.regime_detection import RegimeDetector

_NANOSECONDS_PER_SECOND: int = 1_000_000_000

# Rückblickfenster für get_regime ohne Startzeitpunkt (Flux-Dauer, keine Magie).
_DEFAULT_LOOKBACK: str = "now() - 30d"


@dataclass(frozen=True)
class RegimeStoreResult:
    """Ergebnis eines Detect+Store-Laufs.

    ``regime`` und ``confidence`` sind die erkannten Werte (unabhängig
    davon, ob der Store verfügbar war); ``stored`` ist True nur, wenn
    der Punkt tatsächlich in einen verfügbaren Store geschrieben wurde.
    """

    symbol: str
    timeframe: str
    exchange: str
    regime: str
    confidence: float
    stored: bool


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


def _detect_epoch(candles: list[dict]) -> int:
    """Zeitstempel des Regime-Punkts (Epoch-Sekunden).

    Zeit der letzten Kerze mit parsebarem ``time``; bei keiner Kerze
    (oder ohne parsebare Zeitstempel) der aktuelle UTC-Zeitpunkt.
    """
    for candle in reversed(candles):
        epoch = _parse_timestamp(candle.get("time"))
        if epoch is not None:
            return epoch
    return int(time.time())


class RegimeStore:
    """Schreibt erkannte Regimes in InfluxDB und liest die Historie zurück.

    - Ein Punkt pro Lauf im ``regime``-Measurement; Tags ``symbol`` /
      ``exchange``, Fields laut ``schema.REGIME_FIELDS`` plus Kontext
      ``timeframe`` (Field, kein Tag).
    - ``bucket`` wird aus der Store-Konfiguration gelesen
      (``store._bucket``), damit Queries ohne zusätzliche Konfiguration
      laufen (Muster: ``FeatureStore``).
    - Alle Store-Zugriffe laufen asynchron (``InfluxDBStore`` ist async).
    """

    def __init__(
        self,
        store: InfluxDBStore,
        detector: RegimeDetector | None = None,
    ) -> None:
        """Initialisiert den RegimeStore auf dem gegebenen Store (lazy, keine Verbindung)."""
        self._store = store
        self._detector = detector or RegimeDetector()
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
    ) -> RegimeStoreResult:
        """Erkennt das Regime und schreibt es in InfluxDB.

        Der erkannte ``RegimeResult`` wird als ein Punkt im
        ``regime``-Measurement geschrieben: Tags ``symbol`` /
        ``exchange``, Fields ``regime_name`` / ``regime_confidence`` /
        ``regime_duration`` / ``timeframe``. Bei nicht verfügbarem
        Store (``is_available=False``) wird nicht geschrieben:
        ``stored=False`` (die Erkennung läuft trotzdem).
        """
        result = self._detector.detect(candles)
        stored = False
        if self._store.is_available:
            tags = {"symbol": symbol, "exchange": exchange}
            fields: FieldDict = {
                "regime_name": result.regime,
                "regime_confidence": float(result.confidence),
                "regime_duration": int(result.duration),
                "timeframe": timeframe,
            }
            await self._store.write_points(
                measurement=quant_schema.REGIME_MEASUREMENT,
                tags=tags,
                fields=fields,
                timestamp=_detect_epoch(candles) * _NANOSECONDS_PER_SECOND,
            )
            stored = True

        return RegimeStoreResult(
            symbol=symbol,
            timeframe=timeframe,
            exchange=exchange,
            regime=result.regime,
            confidence=float(result.confidence),
            stored=stored,
        )

    # ------------------------------------------------------------------
    # Lesezugriff (Flux-Query über InfluxDBStore.query)
    # ------------------------------------------------------------------

    async def get_regime(
        self,
        symbol: str,
        timeframe: str,
        start: str | None = None,
        end: str | None = None,
    ) -> list[dict]:
        """Liest die Regime-Historie aus InfluxDB (ein Dict pro Punkt, aufsteigend).

        Jeder Eintrag enthält ``time`` (UTC-ISO-String), ``symbol``,
        ``exchange``, ``timeframe``, ``regime`` (aus ``regime_name``),
        ``confidence`` (aus ``regime_confidence``) und ``duration``
        (aus ``regime_duration``). ``start``/``end`` sind
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
        flux = self._build_query(symbol, start, end)
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
                "timeframe": timeframe,
                "regime": record.get("regime_name"),
            }
            confidence = record.get("regime_confidence")
            if _is_number(confidence):
                entry["confidence"] = float(confidence)
            duration = record.get("regime_duration")
            if _is_number(duration):
                entry["duration"] = int(duration)
            result.append(entry)
        return result

    # ------------------------------------------------------------------
    # Query-Bau (privat)
    # ------------------------------------------------------------------

    def _build_query(
        self,
        symbol: str,
        start: str | None,
        end: str | None,
    ) -> str:
        """Flux: Regime-Punkte im Zeitraum, gepivotet, aufsteigend sortiert."""
        start_clause = f'"{_flux_escape(start)}"' if start is not None else _DEFAULT_LOOKBACK
        range_clause = f"range(start: {start_clause}"
        if end is not None:
            range_clause += f', stop: "{_flux_escape(end)}"'
        range_clause += ")"
        lines = [
            f'from(bucket: "{_flux_escape(self._bucket)}")',
            f"|> {range_clause}",
            f'|> filter(fn: (r) => r._measurement == "{quant_schema.REGIME_MEASUREMENT}")',
            f'|> filter(fn: (r) => r["symbol"] == "{_flux_escape(symbol)}")',
            '|> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")',
            '|> sort(columns: ["_time"])',
        ]
        return "\n".join(lines)
