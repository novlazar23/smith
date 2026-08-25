"""OHLCV-Ingestion für die Quant-Plattform (Phase 1, P1-6).

Schreibt Rohkerzen und abgeleitete (downgesampelte) Kerzen in das einzelne
``ohlcv``-Measurement — der Timeframe ist ein Tag, keine eigene Messung.
Die Schnittstelle folgt ``influxdb_client.InfluxDBStore``: alle Schreib- und
Lesezugriffe laufen asynchron; der Store puffert bei InfluxDB-Ausfall selbst.

Schemakontrakt (``quant/schema.py``):

- Measurement: ``ohlcv`` (ein einziges für alle Timeframes)
- Tags: ``symbol``, ``exchange``, ``timeframe``
- Fields: ``open``, ``high``, ``low``, ``close``, ``volume``

Semantik:

- ``ingest_candles`` schreibt dieselbe Kerzenreihe für jedes gelistete Symbol
  (entspricht ``/quant/ingest/ohlcv``): ``written = Symbole × gültige Kerzen``.
- Strukturell ungültige Kerzen (ungültiges ``time``, fehlende oder nicht-
  numerische OHLCV-Fields) werden deterministisch übersprungen und in
  ``skipped`` gezählt — es wird nie eine halbe Kerze geschrieben.
- ``downsample_candles`` erzeugt ausschließlich aus vollständigen Gruppen
  Kerzen des Ziel-Intervalls (Kerzenzeit = Ziel-Intervall-Start). Unvollständige
  Gruppen (z. B. die offene Kerze am Reihenende) werden verworfen, damit keine
  Look-Ahead- oder Teilkerzen in Backtests entstehen.
- Alle Schreibvorgänge nutzen ``InfluxDBStore.write_batch`` (ein Aufruf pro
  Symbol); alle Lesevorgänge nutzen ``InfluxDBStore.query`` mit Flux.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from trading_harness.quant import schema as quant_schema
from trading_harness.quant.influxdb_client import FieldDict, InfluxDBStore

# Dauer der unterstützten Timeframes in Sekunden (aus schema.py abgeleitet).
_TIMEFRAME_SECONDS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}

_NANOSECONDS_PER_SECOND: int = 1_000_000_000

# Rückblickfenster für get_latest_candle (Flux-Dauer, keine Konstanten-Magie).
_LATEST_CANDLE_LOOKBACK: str = "now() - 7d"


@dataclass(frozen=True)
class IngestResult:
    """Ergebnis eines Ingest-Laufs: geschriebene + übersprungene Punkte."""

    written: int
    skipped: int
    measurement: str


def _validate_timeframe(timeframe: str) -> str:
    """Prüft den Timeframe gegen ``SUPPORTED_TIMEFRAMES``; sonst ValueError."""
    if timeframe not in quant_schema.SUPPORTED_TIMEFRAMES:
        raise ValueError(
            f"unsupported timeframe '{timeframe}', "
            f"expected one of: {', '.join(quant_schema.SUPPORTED_TIMEFRAMES)}"
        )
    return timeframe


def _parse_timestamp(value: object) -> int | None:
    """Parst Zeitstempel (datetime oder ISO-8601-String) zu Epoch-Sekunden.

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
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        moment = datetime.fromtimestamp(value / _NANOSECONDS_PER_SECOND, tz=UTC)
        return moment.isoformat().replace("+00:00", "Z")
    return None


def _epoch_to_iso(epoch: int) -> str:
    """Epoch-Sekunden → UTC-ISO-String (z. B. ``2026-01-01T00:05:00Z``)."""
    return datetime.fromtimestamp(epoch, tz=UTC).isoformat().replace("+00:00", "Z")


def _candle_to_point(candle: Mapping[str, object]) -> tuple[int, FieldDict] | None:
    """Kerzen-Dict → (Epoch-Sekunden, Fields); strukturell ungültig → None.

    Erforderlich: parsebares ``time`` und numerische (nicht-Bool) Werte für
    alle ``OHLCV_FIELDS``.
    """
    epoch = _parse_timestamp(candle.get("time"))
    if epoch is None:
        return None
    fields: FieldDict = {}
    for name in quant_schema.OHLCV_FIELDS:
        value = candle.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        fields[name] = value
    return epoch, fields


def _flux_escape(value: str) -> str:
    """Escape für String-Literalen in Flux-Queries."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


class OHLCVIngestion:
    """OHLCV-Ingestion + Downsampling gegen einen ``InfluxDBStore``.

    - Ein einziges Measurement (``ohlcv``); der Timeframe ist ein Tag.
    - ``exchange`` stammt aus dem Konstruktor, sofern nicht pro Aufruf übergeben.
    - ``bucket`` wird optional explizit übergeben, sonst aus der Store-Konfiguration
      gelesen (``store._bucket``), damit Queries ohne zusätzliche Konfiguration laufen.
    """

    def __init__(
        self,
        store: InfluxDBStore,
        default_exchange: str = "binance",
        bucket: str | None = None,
    ) -> None:
        """Initialisiert die Ingestion auf dem gegebenen Store (lazy, keine Verbindung)."""
        self._store = store
        self._default_exchange = default_exchange
        self._bucket = bucket if bucket is not None else str(getattr(store, "_bucket", "quant"))

    # ------------------------------------------------------------------
    # Ingestion (Schreibpfade)
    # ------------------------------------------------------------------

    async def ingest_candles(
        self,
        symbols: Sequence[str],
        timeframe: str,
        candles: Sequence[Mapping[str, object]],
        exchange: str | None = None,
    ) -> IngestResult:
        """Schreibt Rohkerzen in das ``ohlcv``-Measurement (Timeframe als Tag).

        Die Kerzenreihe wird für jedes Symbol in ``symbols`` geschrieben
        (``written = Symbole × gültige Kerzen``). Strukturell ungültige Kerzen
        zählen in ``skipped``.
        """
        validated_tf = _validate_timeframe(timeframe)
        exch = exchange or self._default_exchange
        written, skipped = await self._write_candles(symbols, validated_tf, exch, candles)
        return IngestResult(written=written, skipped=skipped, measurement=quant_schema.OHLCV_MEASUREMENT)

    async def ingest_and_downsample(
        self,
        symbols: Sequence[str],
        source_tf: str,
        target_tf: str,
        source_candles: Sequence[Mapping[str, object]],
        exchange: str | None = None,
    ) -> IngestResult:
        """Schreibt Quellkerzen UND abgeleitete Kerzen (Downsampling).

        Mapping-Beispiele: 1m→5m (5 Kerzen), 1m→1h (60), 5m→1h (12).
        Aggregation: open=erste, high=max, low=min, close=letzte, volume=Summe.
        """
        source = _validate_timeframe(source_tf)
        target = _validate_timeframe(target_tf)
        _validate_downsample_pair(source, target)
        exch = exchange or self._default_exchange

        downsampled = self.downsample_candles(source_candles, source, target)
        source_written, source_skipped = await self._write_candles(
            symbols, source, exch, source_candles
        )
        target_written, target_skipped = await self._write_candles(
            symbols, target, exch, downsampled
        )
        return IngestResult(
            written=source_written + target_written,
            skipped=source_skipped + target_skipped,
            measurement=quant_schema.OHLCV_MEASUREMENT,
        )

    async def _write_candles(
        self,
        symbols: Sequence[str],
        timeframe: str,
        exchange: str,
        candles: Sequence[Mapping[str, object]],
    ) -> tuple[int, int]:
        """Schreibt eine Kerzenreihe pro Symbol via ``write_batch`` → (written, skipped)."""
        points: list[FieldDict] = []
        timestamps: list[int] = []
        skipped = 0
        for candle in candles:
            point = _candle_to_point(candle)
            if point is None:
                skipped += 1
                continue
            epoch, fields = point
            points.append(fields)
            timestamps.append(epoch * _NANOSECONDS_PER_SECOND)
        if not points or not symbols:
            return 0, skipped
        for symbol in symbols:
            tags = {"symbol": symbol, "exchange": exchange, "timeframe": timeframe}
            await self._store.write_batch(
                measurement=quant_schema.OHLCV_MEASUREMENT,
                tags=tags,
                points=points,
                timestamps=timestamps,
            )
        return len(points) * len(symbols), skipped

    # ------------------------------------------------------------------
    # Downsampling (rein deterministisch, kein Store-Zugriff)
    # ------------------------------------------------------------------

    def downsample_candles(
        self,
        candles: Sequence[Mapping[str, object]],
        source_tf: str,
        target_tf: str,
    ) -> list[dict[str, float | str]]:
        """Gruppiert Quellkerzen in Zielkerzen (nur vollständige Gruppen).

        Kerzenzeit des Ergebnisses = Start des Ziel-Intervalls (UTC-ISO-String).
        Unvollständige Gruppen (fehlende Quellkerzen, offene Kerze am Ende)
        werden verworfen — es entstehen keine Teilkerzen.
        """
        source = _validate_timeframe(source_tf)
        target = _validate_timeframe(target_tf)
        _validate_downsample_pair(source, target)
        target_seconds = _TIMEFRAME_SECONDS[target]
        group_size = target_seconds // _TIMEFRAME_SECONDS[source]

        parsed: list[tuple[int, FieldDict]] = []
        for candle in candles:
            point = _candle_to_point(candle)
            if point is not None:
                parsed.append(point)
        parsed.sort(key=lambda item: item[0])

        buckets: dict[int, list[FieldDict]] = {}
        for epoch, fields in parsed:
            bucket_start = (epoch // target_seconds) * target_seconds
            buckets.setdefault(bucket_start, []).append(fields)

        result: list[dict[str, float | str]] = []
        for bucket_start in sorted(buckets):
            group = buckets[bucket_start]
            if len(group) != group_size:
                continue  # unvollständig → keine Teilkerze (kein Look-Ahead)
            result.append(
                {
                    "time": _epoch_to_iso(bucket_start),
                    "open": float(group[0]["open"]),
                    "high": max(float(f["high"]) for f in group),
                    "low": min(float(f["low"]) for f in group),
                    "close": float(group[-1]["close"]),
                    "volume": sum(float(f["volume"]) for f in group),
                }
            )
        return result

    # ------------------------------------------------------------------
    # Lesezugriffe (Flux-Queries über InfluxDBStore.query)
    # ------------------------------------------------------------------

    async def get_latest_candle(
        self,
        symbol: str,
        timeframe: str,
        exchange: str | None = None,
    ) -> dict[str, float | str] | None:
        """Neueste Kerze für Symbol/Timeframe/Exchange (Rückblick: 7 Tage)."""
        validated_tf = _validate_timeframe(timeframe)
        exch = exchange or self._default_exchange
        rows = await self._store.query(self._latest_candle_query(symbol, validated_tf, exch))
        for record in rows:
            candle = self._record_to_candle(record)
            if candle is not None:
                return candle
        return None

    async def get_candle_range(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
        exchange: str | None = None,
    ) -> list[dict[str, float | str]]:
        """Kerzen im Zeitraum [start, stop] (UTC-ISO-Zeichenketten, aufsteigend)."""
        validated_tf = _validate_timeframe(timeframe)
        if _parse_timestamp(start) is None:
            raise ValueError(f"invalid start timestamp: {start!r}")
        if _parse_timestamp(end) is None:
            raise ValueError(f"invalid end timestamp: {end!r}")
        exch = exchange or self._default_exchange
        rows = await self._store.query(
            self._candle_range_query(symbol, validated_tf, exch, start, end)
        )
        result: list[dict[str, float | str]] = []
        for record in rows:
            candle = self._record_to_candle(record)
            if candle is not None:
                result.append(candle)
        return result

    # ------------------------------------------------------------------
    # Query-Bau + Ergebnis-Konvertierung (privat)
    # ------------------------------------------------------------------

    def _base_query_lines(
        self, symbol: str, timeframe: str, exchange: str, start: str, stop: str | None
    ) -> list[str]:
        """Gemeinsamer Flux-Block: Bucket, Zeitraum, Measurement, Tag-Filter, Pivot."""
        field_list = ", ".join(f'"{name}"' for name in quant_schema.OHLCV_FIELDS)
        range_clause = f"range(start: {start}"
        if stop is not None:
            range_clause += f", stop: {stop}"
        range_clause += ")"
        return [
            f'from(bucket: "{_flux_escape(self._bucket)}")',
            f"|> {range_clause}",
            f'|> filter(fn: (r) => r._measurement == "{quant_schema.OHLCV_MEASUREMENT}")',
            f'|> filter(fn: (r) => r._field in [{field_list}])',
            f'|> filter(fn: (r) => r["symbol"] == "{_flux_escape(symbol)}")',
            f'|> filter(fn: (r) => r["exchange"] == "{_flux_escape(exchange)}")',
            f'|> filter(fn: (r) => r["timeframe"] == "{_flux_escape(timeframe)}")',
            '|> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")',
        ]

    def _latest_candle_query(self, symbol: str, timeframe: str, exchange: str) -> str:
        """Flux: neueste Kerze (Rückblick 7 Tage), absteigend, limit 1."""
        lines = self._base_query_lines(symbol, timeframe, exchange, _LATEST_CANDLE_LOOKBACK, None)
        lines.append('|> sort(columns: ["_time"], desc: true)')
        lines.append("|> limit(n: 1)")
        return "\n".join(lines)

    def _candle_range_query(
        self, symbol: str, timeframe: str, exchange: str, start: str, end: str
    ) -> str:
        """Flux: Kerzen im Zeitraum [start, stop], aufsteigend sortiert."""
        lines = self._base_query_lines(
            symbol,
            timeframe,
            exchange,
            f'"{_flux_escape(start)}"',
            f'"{_flux_escape(end)}"',
        )
        lines.append('|> sort(columns: ["_time"])')
        return "\n".join(lines)

    @staticmethod
    def _record_to_candle(record: Mapping[str, object]) -> dict[str, float | str] | None:
        """Pivotierter Flux-Record → Kerzen-Dict; unvollständige Records → None."""
        time_iso = _timestamp_to_iso(record.get("_time"))
        if time_iso is None:
            return None
        candle: dict[str, float | str] = {"time": time_iso}
        for name in quant_schema.OHLCV_FIELDS:
            value = record.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            candle[name] = float(value)
        return candle


def _validate_downsample_pair(source: str, target: str) -> None:
    """Prüft, dass das Ziel-Intervall ein größeres ganzzahliges Vielfaches der Quelle ist."""
    source_seconds = _TIMEFRAME_SECONDS[source]
    target_seconds = _TIMEFRAME_SECONDS[target]
    if target_seconds <= source_seconds:
        raise ValueError(
            f"target timeframe '{target}' must be larger than source timeframe '{source}'"
        )
    if target_seconds % source_seconds != 0:
        raise ValueError(
            f"cannot downsample '{source}' to '{target}': "
            "target must be a whole multiple of the source interval"
        )
