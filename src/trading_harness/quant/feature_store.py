"""Feature-Storage für die Quant-Plattform (Phase 2, P2-2).

Verbindet ``FeatureEngine`` (P2-1) mit ``InfluxDBStore`` (P1-4):
berechnete Indikatorwerte werden je Kerze als ein Punkt in das
``features``-Measurement geschrieben und per Flux-Query als Zeitreihe
zurückgelesen.

Schemakontrakt (``quant/schema.py``):

- Measurement: ``features``
- Tags: ``symbol``, ``exchange``, ``feature_version``
- Fields: dynamische Feature-Werte (``rsi``, ``macd_line``, ...) +
  ``timeframe`` + Metadaten ``feature_count`` / ``computation_time_ms``

Semantik:

- Rolling Calculation: Der Punkt der Kerze *i* wird ausschließlich aus
  ``candles[: i + 1]`` berechnet — kein Look-Ahead (ein Featurewert zum
  Zeitpunkt *t* nutzt nie Daten jünger als *t*).
- Guard-Klauseln: ``None``-Werte des Engines (unzureichende Historie)
  werden im jeweiligen Punkt weggelassen — nie eine halbe Zahl.
- Kerzen ohne parsebares ``time`` werden übersprungen (analog der
  OHLCV-Ingestion); es entsteht kein Punkt ohne Zeitstempel.
- Store nicht verfügbar (``is_available=False``): kein Schreibversuch
  (``stored=False``), ``get_features`` liefert eine leere Liste.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, TypeGuard

from trading_harness.quant import schema as quant_schema
from trading_harness.quant.features import FeatureEngine
from trading_harness.quant.influxdb_client import FieldDict, InfluxDBStore

# Flache Feature-Feldnamen, die in den Punkt geschrieben werden.
FEATURE_FIELD_NAMES: tuple[str, ...] = (
    "rsi",
    "macd_line",
    "signal_line",
    "histogram",
    "bollinger_upper",
    "bollinger_middle",
    "bollinger_lower",
    "bollinger_bandwidth",
    "atr",
    "volatility",
    "vwap",
)

# Metadatenfelder des features-Measurements (siehe schema.FEATURE_META_FIELDS).
_META_FIELD_NAMES: tuple[str, ...] = quant_schema.FEATURE_META_FIELDS

_NANOSECONDS_PER_SECOND: int = 1_000_000_000

# Rückblickfenster für get_features ohne Startzeitpunkt (Flux-Dauer, keine Magie).
_DEFAULT_LOOKBACK: str = "now() - 30d"

# Feld-Mappings: Key im compute()-Output → Feldname im InfluxDB-Punkt.
_MACD_FIELD_MAP: tuple[tuple[str, str], ...] = (
    ("macd", "macd_line"),
    ("signal", "signal_line"),
    ("histogram", "histogram"),
)
_BOLLINGER_FIELD_MAP: tuple[tuple[str, str], ...] = (
    ("upper", "bollinger_upper"),
    ("middle", "bollinger_middle"),
    ("lower", "bollinger_lower"),
    ("bandwidth", "bollinger_bandwidth"),
)


@dataclass(frozen=True)
class FeatureResult:
    """Ergebnis eines Compute+Store-Laufs.

    ``feature_count`` ist die Gesamtzahl geschriebener Feature-Werte über
    alle Punkte (Summe der per-Punkt-``feature_count``-Felder);
    ``computation_time_ms`` die Gesamtlaufzeit des Laufs; ``stored`` ist
    True nur, wenn mindestens ein Punkt in einen verfügbaren Store
    geschrieben wurde.
    """

    symbol: str
    timeframe: str
    exchange: str
    feature_count: int
    computation_time_ms: float
    stored: bool


def _is_number(value: object) -> TypeGuard[int | float]:
    """True für int/float (ausgeschlossen: Bool, das in Python ein int ist)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _flatten_features(computed: dict[str, Any]) -> dict[str, float]:
    """Mappt das ``compute``-Ergebnis auf flache Feature-Felder.

    ``None``-Werte (unzureichende Historie) werden weggelassen — es
    landen nie leere oder halbe Werte im Punkt.
    """
    fields: dict[str, float] = {}
    rsi = computed.get("rsi")
    if _is_number(rsi):
        fields["rsi"] = float(rsi)
    macd = computed.get("macd")
    if isinstance(macd, dict):
        for source, target in _MACD_FIELD_MAP:
            value = macd.get(source)
            if _is_number(value):
                fields[target] = float(value)
    bollinger = computed.get("bollinger")
    if isinstance(bollinger, dict):
        for source, target in _BOLLINGER_FIELD_MAP:
            value = bollinger.get(source)
            if _is_number(value):
                fields[target] = float(value)
    for name in ("atr", "volatility", "vwap"):
        value = computed.get(name)
        if _is_number(value):
            fields[name] = float(value)
    return fields


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


class FeatureStore:
    """Schreibt berechnete Features in InfluxDB und liest sie zurück.

    - Ein Punkt pro Kerze: alle zur Verfügung stehenden Features der
      Kerze als Fields (``timeframe`` als Field, kein Tag — laut
      ``schema.FEATURE_TAGS``).
    - ``bucket`` wird optional explizit übergeben, sonst aus der
      Store-Konfiguration gelesen (``store._bucket``), damit Queries ohne
      zusätzliche Konfiguration laufen (Muster: OHLCVIngestion).
    - Alle Store-Zugriffe laufen asynchron (``InfluxDBStore`` ist async).
    """

    def __init__(
        self,
        store: InfluxDBStore,
        engine: FeatureEngine | None = None,
        bucket: str | None = None,
    ) -> None:
        """Initialisiert den FeatureStore auf dem gegebenen Store (lazy, keine Verbindung)."""
        self._store = store
        self._engine = engine or FeatureEngine()
        self._bucket = bucket if bucket is not None else str(getattr(store, "_bucket", "quant"))

    # ------------------------------------------------------------------
    # Schreibpfad
    # ------------------------------------------------------------------

    async def compute_and_store(
        self,
        symbol: str,
        timeframe: str,
        candles: list[dict],
        exchange: str = "binance",
    ) -> FeatureResult:
        """Berechnet Features und schreibt sie in InfluxDB.

        Pro Kerze entsteht ein Punkt im ``features``-Measurement, dessen
        Fields alle zur Verfügung stehenden Feature-Werte enthalten
        (``rsi``, ``macd_line``, ``signal_line``, ``histogram``,
        ``bollinger_upper/middle/lower/bandwidth``, ``atr``, ``volatility``,
        ``vwap``) plus ``timeframe`` und die Metadaten ``feature_count`` /
        ``computation_time_ms``. Der Punkt der Kerze *i* wird nur aus
        ``candles[: i + 1]`` berechnet (kein Look-Ahead). Kerzen ohne
        parsebares ``time`` werden übersprungen. Bei nicht verfügbarem
        Store (``is_available=False``) wird nicht geschrieben:
        ``stored=False``.
        """
        t0 = time.monotonic()
        tags = {
            "symbol": symbol,
            "exchange": exchange,
            "feature_version": quant_schema.FEATURE_VERSION,
        }
        points: list[tuple[int, FieldDict]] = []
        total_features = 0
        for index, candle in enumerate(candles):
            epoch = _parse_timestamp(candle.get("time"))
            if epoch is None:
                continue  # strukturell ungültige Kerze → überspringen (wie Ingestion)
            computed = self._engine.compute(candles[: index + 1])
            features = _flatten_features(computed)
            fields: FieldDict = {"timeframe": timeframe}
            fields.update(features)
            fields["feature_count"] = len(features)
            fields["computation_time_ms"] = float(computed["computation_time_ms"])
            points.append((epoch * _NANOSECONDS_PER_SECOND, fields))
            total_features += len(features)

        written = 0
        if self._store.is_available:
            for timestamp_ns, fields in points:
                await self._store.write_points(
                    measurement=quant_schema.FEATURE_MEASUREMENT,
                    tags=tags,
                    fields=fields,
                    timestamp=timestamp_ns,
                )
                written += 1

        elapsed_ms = (time.monotonic() - t0) * 1000
        return FeatureResult(
            symbol=symbol,
            timeframe=timeframe,
            exchange=exchange,
            feature_count=total_features,
            computation_time_ms=elapsed_ms,
            stored=written > 0,
        )

    # ------------------------------------------------------------------
    # Lesezugriff (Flux-Query über InfluxDBStore.query)
    # ------------------------------------------------------------------

    async def get_features(
        self,
        symbol: str,
        timeframe: str,
        feature_names: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        exchange: str = "binance",
    ) -> list[dict]:
        """Liest Features aus InfluxDB (ein Dict pro Punkt, aufsteigend).

        Jeder Eintrag enthält ``time`` (UTC-ISO-String), ``symbol``,
        ``exchange``, ``timeframe`` sowie die vorhandenen Feature- und
        Metadaten-Felder. ``start``/``end`` sind UTC-ISO-Zeichenketten
        (Default-Rückblick ohne ``start``: 30 Tage). Bei nicht
        verfügbarem Store (``is_available=False``) leere Liste;
        ungültige Zeitstempel werfen ``ValueError``.
        """
        if not self._store.is_available:
            return []
        if start is not None and _parse_timestamp(start) is None:
            raise ValueError(f"invalid start timestamp: {start!r}")
        if end is not None and _parse_timestamp(end) is None:
            raise ValueError(f"invalid end timestamp: {end!r}")
        flux = self._build_query(symbol, timeframe, exchange, start, end, feature_names)
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
            }
            for name in (*FEATURE_FIELD_NAMES, *_META_FIELD_NAMES):
                value = record.get(name)
                if _is_number(value):
                    entry[name] = value
            result.append(entry)
        return result

    # ------------------------------------------------------------------
    # Query-Bau (privat)
    # ------------------------------------------------------------------

    def _build_query(
        self,
        symbol: str,
        timeframe: str,
        exchange: str,
        start: str | None,
        end: str | None,
        feature_names: list[str] | None,
    ) -> str:
        """Flux: Features im Zeitraum, gepivotet, aufsteigend sortiert.

        Bei ``feature_names`` wird zusätzlich auf ``r._field`` gefiltert
        (inklusive ``timeframe``, damit der Zeitstempel-Feld-Filter nach
        dem Pivot weiterfunktioniert).
        """
        start_clause = f'"{_flux_escape(start)}"' if start is not None else _DEFAULT_LOOKBACK
        range_clause = f"range(start: {start_clause}"
        if end is not None:
            range_clause += f', stop: "{_flux_escape(end)}"'
        range_clause += ")"
        lines = [
            f'from(bucket: "{_flux_escape(self._bucket)}")',
            f"|> {range_clause}",
            f'|> filter(fn: (r) => r._measurement == "{quant_schema.FEATURE_MEASUREMENT}")',
            f'|> filter(fn: (r) => r["symbol"] == "{_flux_escape(symbol)}")',
            f'|> filter(fn: (r) => r["exchange"] == "{_flux_escape(exchange)}")',
            f'|> filter(fn: (r) => r["feature_version"] == "{quant_schema.FEATURE_VERSION}")',
        ]
        if feature_names:
            field_list = ", ".join(f'"{_flux_escape(name)}"' for name in (*feature_names, "timeframe"))
            lines.append(f"|> filter(fn: (r) => r._field in [{field_list}])")
        lines.append('|> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")')
        lines.append('|> sort(columns: ["_time"])')
        return "\n".join(lines)
