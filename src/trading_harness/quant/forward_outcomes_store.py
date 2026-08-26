"""Forward-Outcome-Storage für die Quant-Plattform (Phase 6, P6-2).

Verbindet ``ForwardOutcomeEngine`` (P6-1) mit ``InfluxDBStore`` (P1-4):
berechnete Forward-Outcome-Statistics werden pro Horizont als ein Punkt in
das ``forward_outcomes``-Measurement geschrieben und per Flux-Query
zurückgelesen.

Schemakontrakt:

- Measurement: ``forward_outcomes``
- Tags: ``symbol``, ``exchange``, ``horizon`` (der Horizont variiert je
  Punkt — Muster: ``anomaly_store`` mit ``anomaly_type``)
- Fields: ``mean_return``, ``median_return``, ``hit_rate``, ``profit_factor``,
  ``expectancy``, ``std_return``, ``sample_size``, ``max_gain``, ``max_loss``
  plus Kontext ``timeframe`` und ``pattern_length``

Semantik:

- Ein Punkt pro Horizont mit ``sample_size > 0`` — Horizonte ohne Daten
  erzeugen keine Punkte (sie tragen keine Information).
- ``horizons_computed`` zählt die Horizonte mit tatsächlich berechneten
  Forward Returns; bei unzureichenden Kerzen ist es 0 und es wird nichts
  geschrieben.
- Engine-Fehler (z. B. fehlendes ``close``-Feld in einer Kerze) werden
  abgefangen und geloggt: der Lauf kommt mit ``horizons_computed=0`` /
  ``stored=False`` zurück, statt die Exception an den Aufrufer zu werfen.
- Store nicht verfügbar (``is_available=False``): kein Schreibversuch
  (``stored=False``), ``get_outcomes`` liefert eine leere Liste.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, TypeGuard

from trading_harness.quant.forward_outcomes import ForwardOutcome, ForwardOutcomeEngine
from trading_harness.quant.influxdb_client import FieldDict, InfluxDBStore

logger = logging.getLogger(__name__)

_NANOSECONDS_PER_SECOND: int = 1_000_000_000

# Measurement-Name (Schema-Kontrakt, siehe Modul-Docstring).
FORWARD_OUTCOMES_MEASUREMENT: str = "forward_outcomes"

# Statistik-Fields eines ForwardOutcome (Schema-Kontrakt).
OUTCOME_STAT_FIELDS: tuple[str, ...] = (
    "mean_return",
    "median_return",
    "hit_rate",
    "profit_factor",
    "expectancy",
    "std_return",
    "sample_size",
    "max_gain",
    "max_loss",
)

# Rückblickfenster für get_outcomes ohne Startzeitpunkt (Flux-Dauer, keine Magie).
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


def _outcome_fields(outcome: ForwardOutcome, timeframe: str, pattern_length: int) -> FieldDict:
    """Mappt ein ForwardOutcome auf seine InfluxDB-Fields (Kontext-Zeilen)."""
    return {
        "mean_return": float(outcome.mean_return),
        "median_return": float(outcome.median_return),
        "hit_rate": float(outcome.hit_rate),
        "profit_factor": float(outcome.profit_factor),
        "expectancy": float(outcome.expectancy),
        "std_return": float(outcome.std_return),
        "sample_size": int(outcome.sample_size),
        "max_gain": float(outcome.max_gain),
        "max_loss": float(outcome.max_loss),
        "timeframe": timeframe,
        "pattern_length": int(pattern_length),
    }


def _compute_epoch(candles: list[dict]) -> int:
    """Zeitstempel der Outcome-Punkte (Epoch-Sekunden).

    Zeit der letzten Kerze mit parsebarem ``time``; bei keiner Kerze
    (oder ohne parsebare Zeitstempel) der aktuelle UTC-Zeitpunkt
    (Muster: ``RegimeStore`` / ``AnomalyStore``).
    """
    for candle in reversed(candles):
        epoch = _parse_timestamp(candle.get("time"))
        if epoch is not None:
            return epoch
    return int(time.time())


@dataclass(frozen=True)
class ForwardOutcomeStoreResult:
    """Ergebnis eines Compute+Store-Laufs.

    ``horizons_computed`` ist die Anzahl der Horizonte mit tatsächlich
    berechneten Forward Returns (unabhängig davon, ob der Store verfügbar
    war); ``stored`` ist True nur, wenn mindestens ein Punkt in einen
    verfügbaren Store geschrieben wurde.
    """

    symbol: str
    timeframe: str
    pattern_length: int
    horizons_computed: int
    stored: bool


class ForwardOutcomeStore:
    """Schreibt berechnete Forward Outcomes in InfluxDB und liest sie zurück.

    - Ein Punkt pro Horizont mit Daten im ``forward_outcomes``-Measurement;
      Tags ``symbol`` / ``exchange`` / ``horizon``, Fields laut
      Schema-Kontrakt.
    - ``bucket`` wird aus der Store-Konfiguration gelesen
      (``store._bucket``), damit Queries ohne zusätzliche Konfiguration
      laufen (Muster: ``FeatureStore``).
    - Alle Store-Zugriffe laufen asynchron (``InfluxDBStore`` ist async).
    """

    def __init__(
        self,
        store: InfluxDBStore,
        engine: ForwardOutcomeEngine | None = None,
    ) -> None:
        """Initialisiert den Store auf dem gegebenen Store (lazy, keine Verbindung)."""
        self._store = store
        self._engine = engine or ForwardOutcomeEngine()
        self._bucket = str(getattr(store, "_bucket", "quant"))

    # ------------------------------------------------------------------
    # Schreibpfad
    # ------------------------------------------------------------------

    async def compute_and_store(
        self,
        symbol: str,
        timeframe: str,
        candles: list[dict],
        pattern_length: int = 10,
        exchange: str = "binance",
    ) -> ForwardOutcomeStoreResult:
        """Berechnet Forward Outcomes und schreibt sie in InfluxDB.

        Pro Horizont mit ``sample_size > 0`` entsteht ein Punkt im
        ``forward_outcomes``-Measurement (Tags ``symbol`` / ``exchange`` /
        ``horizon``). Bei unzureichenden Kerzen (kein Horizont mit Daten)
        wird nichts geschrieben. Bei Engine-Fehlern (z. B. strukturell
        ungültige Kerzen) wird die Exception abgefangen und geloggt; der
        Lauf liefert dann ``horizons_computed=0`` / ``stored=False``.
        Bei nicht verfügbarem Store (``is_available=False``) wird nicht
        geschrieben: ``stored=False`` (die Berechnung läuft trotzdem).
        """
        try:
            result = self._engine.compute(
                candles,
                pattern_length=pattern_length,
                symbol=symbol,
                timeframe=timeframe,
            )
        except Exception:
            logger.warning(
                "Forward outcome computation failed for %s/%s (pattern_length=%d) — "
                "returning empty result",
                symbol,
                timeframe,
                pattern_length,
                exc_info=True,
            )
            return ForwardOutcomeStoreResult(
                symbol=symbol,
                timeframe=timeframe,
                pattern_length=pattern_length,
                horizons_computed=0,
                stored=False,
            )

        computed = [outcome for outcome in result.outcomes.values() if outcome.sample_size > 0]
        stored = False
        if self._store.is_available and computed:
            epoch = _compute_epoch(candles)
            for outcome in sorted(computed, key=lambda o: o.horizon):
                await self._store.write_points(
                    measurement=FORWARD_OUTCOMES_MEASUREMENT,
                    tags={
                        "symbol": symbol,
                        "exchange": exchange,
                        "horizon": str(outcome.horizon),
                    },
                    fields=_outcome_fields(outcome, timeframe, pattern_length),
                    timestamp=epoch * _NANOSECONDS_PER_SECOND,
                )
            stored = True

        return ForwardOutcomeStoreResult(
            symbol=symbol,
            timeframe=timeframe,
            pattern_length=pattern_length,
            horizons_computed=len(computed),
            stored=stored,
        )

    # ------------------------------------------------------------------
    # Lesezugriff (Flux-Query über InfluxDBStore.query)
    # ------------------------------------------------------------------

    async def get_outcomes(
        self,
        symbol: str,
        timeframe: str,
        start: str | None = None,
        end: str | None = None,
    ) -> list[dict]:
        """Liest gespeicherte Forward Outcomes aus InfluxDB (aufsteigend).

        Jeder Eintrag enthält ``time`` (UTC-ISO-String), ``symbol``,
        ``exchange``, ``timeframe``, ``horizon`` (int, aus dem Tag),
        ``pattern_length`` sowie die vorhandenen Statistik-Fields.
        ``start``/``end`` sind UTC-ISO-Zeichenketten (Default-Rückblick
        ohne ``start``: 30 Tage). Bei nicht verfügbarem Store
        (``is_available=False``) leere Liste; ungültige Zeitstempel
        werfen ``ValueError``.
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
            horizon = record.get("horizon")
            # horizon ist ein Tag → in gepivoteten Records eine Zeichenkette.
            if isinstance(horizon, str) and horizon.isdigit():
                horizon = int(horizon)
            if not _is_number(horizon):
                continue  # ohne eindeutigen Horizont kein gültiger Outcome
            entry: dict[str, Any] = {
                "time": time_iso,
                "symbol": record.get("symbol"),
                "exchange": record.get("exchange"),
                "timeframe": timeframe,
                "horizon": int(horizon),
            }
            pattern_length = record.get("pattern_length")
            if _is_number(pattern_length):
                entry["pattern_length"] = int(pattern_length)
            for name in OUTCOME_STAT_FIELDS:
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
        start: str | None,
        end: str | None,
    ) -> str:
        """Flux: Forward-Outcome-Punkte im Zeitraum, gepivotet, sortiert."""
        start_clause = f'"{_flux_escape(start)}"' if start is not None else _DEFAULT_LOOKBACK
        range_clause = f"range(start: {start_clause}"
        if end is not None:
            range_clause += f', stop: "{_flux_escape(end)}"'
        range_clause += ")"
        lines = [
            f'from(bucket: "{_flux_escape(self._bucket)}")',
            f"|> {range_clause}",
            f'|> filter(fn: (r) => r._measurement == "{FORWARD_OUTCOMES_MEASUREMENT}")',
            f'|> filter(fn: (r) => r["symbol"] == "{_flux_escape(symbol)}")',
            '|> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")',
            '|> sort(columns: ["_time"])',
        ]
        return "\n".join(lines)
