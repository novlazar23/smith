"""Orchestrierung des Candle-Backfills.

Pro Instrument:

1. gewünschtes Fenster berechnen (``start``/``end`` oder jetzt minus Monate)
2. vorhandenes Kerzenfenster aus ClickHouse lesen (``existing_range``)
3. fehlende Teilmengen (Lücken) vor/nach dem vorhandenen Fenster berechnen
4. jede Lücke paginiert von Binance laden und in Batches à 5000 schreiben
5. Fortschritt pro 1000-Kerzen-Fenster und Final-Summary loggen

Im ``dry_run``-Modus werden nur der Plan (Lücken + geschätzte
Request-Zahl) und das vorhandene Fenster geloggt — es erfolgen **keine**
HTTP-Downloads und keine ClickHouse-Schreibzugriffe (die Lesezugriffe
für die Planberechnung bleiben erlaubt).
"""

from __future__ import annotations

import calendar
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from apps.backfill import storage
from apps.backfill.client import PAGE_SIZE
from packages.ingestion.adapter.binance import BINANCE_FUTURES_VENUE

if TYPE_CHECKING:
    from apps.backfill.client import KlineClient

logger = logging.getLogger(__name__)

ONE_MINUTE = timedelta(minutes=1)
_FMT = "%Y-%m-%d %H:%M"


@dataclass(frozen=True)
class BackfillConfig:
    """Laufzeit-Konfiguration des Backfills."""

    months: int
    instruments: tuple[str, ...]
    start: datetime | None = None
    end: datetime | None = None
    dry_run: bool = False
    venue: str = BINANCE_FUTURES_VENUE


@dataclass(frozen=True)
class InstrumentSummary:
    """Ergebnis des Backfills für ein einzelnes Instrument."""

    instrument: str
    start: datetime
    end: datetime
    ranges: tuple[tuple[datetime, datetime], ...]
    estimated_requests: int
    fetched_candles: int
    total_after: int | None


@dataclass(frozen=True)
class BackfillResult:
    """Gesamtergebnis eines Backfill-Laufs."""

    summaries: tuple[InstrumentSummary, ...]
    failures: tuple[tuple[str, str], ...]


def months_ago(moment: datetime, months: int) -> datetime:
    """Verschiebt ``moment`` um ``months`` Kalendermonate zurück.

    Der Tag wird bei kürzeren Monaten geklemmt
    (z. B. 31. März - 1 Monat = 28./29. Februar).
    """
    total = moment.month - 1 - months
    year = moment.year + total // 12
    month = total % 12 + 1
    day = min(moment.day, calendar.monthrange(year, month)[1])
    return moment.replace(year=year, month=month, day=day)


def compute_missing_ranges(
    start: datetime,
    end: datetime,
    existing: tuple[datetime, datetime] | None,
) -> list[tuple[datetime, datetime]]:
    """Berechnet die fehlenden Teilmengen des Fensters ``[start, end]``.

    ``existing`` ist das bereits in ClickHouse vorhandene Fenster
    ``(min, max)`` von ``open_time``. Überlappende Bereiche werden
    übersprungen; Lücken werden exakt an den vorhandenen Rand angelegt
    (vorderer Teil endet bei ``min - 1 Min``, hinterer beginnt bei
    ``max + 1 Min``).

    Args:
        start: Gewünschter Fensterbeginn.
        end: Gewünschtes Fensterende (inklusive).
        existing: Vorhandenes Fenster oder ``None`` (keine Kerzen).

    Returns:
        Liste fehlender ``(von, bis)``-Paare in Zeitreihenfolge
        (leer, wenn das Fenster vollständig abgedeckt ist).
    """
    if start >= end:
        return []
    if existing is None:
        return [(start, end)]
    existing_min, existing_max = existing
    ranges: list[tuple[datetime, datetime]] = []
    if start < existing_min:
        ranges.append((start, min(existing_min - ONE_MINUTE, end)))
    if end > existing_max:
        ranges.append((max(start, existing_max + ONE_MINUTE), end))
    return ranges


def candle_count(start: datetime, end: datetime) -> int:
    """Anzahl der 1m-Kerzen im (beidseitig inklusiven) Fenster."""
    return int((end - start).total_seconds() // 60) + 1


def estimate_requests(ranges: Sequence[tuple[datetime, datetime]]) -> int:
    """Schätzt die Klines-Request-Zahl (1 Request pro 1000-Kerzen-Fenster).

    Args:
        ranges: Zu ladende Lücken.

    Returns:
        Summe der ``ceil(Kerzen / PAGE_SIZE)`` über alle Lücken —
        exakt die Anzahl der Requests, die ``KlineClient.fetch_range``
        pro Lücke ausführt.
    """
    total = 0
    for range_start, range_end in ranges:
        total += math.ceil(candle_count(range_start, range_end) / PAGE_SIZE)
    return total


class BackfillService:
    """Führt den Backfill für alle konfigurierten Instrumente aus."""

    def __init__(
        self,
        config: BackfillConfig,
        client: KlineClient,
        engine: storage.CandleEngine,
        now: datetime | None = None,
    ) -> None:
        """Initialisiert die Orchestrierung.

        Args:
            config: Laufzeit-Konfiguration.
            client: Klines-Client (Binance-Futures).
            engine: ClickHouse-Engine.
            now: Referenzzeitpunkt (Default: jetzt; in Tests injizierbar).
        """
        self._config = config
        self._client = client
        self._engine = engine
        self._now = now or datetime.now(UTC)

    @property
    def config(self) -> BackfillConfig:
        """Aktuelle Konfiguration."""
        return self._config

    def window(self) -> tuple[datetime, datetime]:
        """Berechnet das gewünschte Fenster ``(start, end)``.

        ``start`` stammt aus ``config.start`` oder aus
        ``now - config.months`` (Kalendermonate); ``end`` aus
        ``config.end`` oder ``now``.
        """
        start = self._config.start
        if start is None:
            start = months_ago(self._now, self._config.months)
        end = self._config.end or self._now
        return start, end

    def run(self) -> BackfillResult:
        """Führt den Backfill für alle Instrumente aus.

        Ein Fehler bei einem Instrument beendet nicht den gesamten Lauf
        (Exception wird geloggt, nächstes Instrument weiter); die
        gesammelten Fehler stehen in ``BackfillResult.failures``.
        """
        summaries: list[InstrumentSummary] = []
        failures: list[tuple[str, str]] = []
        for instrument in self._config.instruments:
            try:
                summaries.append(self._run_instrument(instrument))
            except Exception as exc:
                logger.exception("Backfill für %s fehlgeschlagen: %s", instrument, exc)
                failures.append((instrument, str(exc)))
        return BackfillResult(summaries=tuple(summaries), failures=tuple(failures))

    def _run_instrument(self, instrument: str) -> InstrumentSummary:
        """Berechnet, lädt und persistiert die fehlenden Kerzen eines Instruments."""
        start, end = self.window()
        existing = storage.existing_range(self._engine, instrument, self._config.venue)
        ranges = compute_missing_ranges(start, end, existing)
        estimated = estimate_requests(ranges)
        self._log_plan(instrument, start, end, existing, ranges, estimated)
        if self._config.dry_run:
            return InstrumentSummary(
                instrument, start, end, tuple(ranges), estimated, 0, None
            )

        fetched = 0
        for range_start, range_end in ranges:
            candles = self._client.fetch_range(
                instrument,
                range_start,
                range_end,
                on_chunk=self._make_chunk_logger(instrument),
            )
            storage.insert_candles(self._engine, candles)
            fetched += len(candles)
            logger.info(
                "Backfill %s: %s → %s: %d Kerzen geschrieben",
                instrument,
                _fmt(range_start),
                _fmt(range_end),
                len(candles),
            )
        total_after = storage.count_candles(self._engine, instrument, self._config.venue)
        logger.info(
            "Backfill %s abgeschlossen: %d Kerzen geladen, candles enthält jetzt %d Zeilen",
            instrument,
            fetched,
            total_after,
        )
        return InstrumentSummary(
            instrument, start, end, tuple(ranges), estimated, fetched, total_after
        )

    def _log_plan(
        self,
        instrument: str,
        start: datetime,
        end: datetime,
        existing: tuple[datetime, datetime] | None,
        ranges: list[tuple[datetime, datetime]],
        estimated: int,
    ) -> None:
        """Loggt den Backfill-Plan (im Dry-Run der einzige sichtbare Teil)."""
        if existing is None:
            have = "keine Kerzen vorhanden"
        else:
            have = f"vorhanden {_fmt(existing[0])} → {_fmt(existing[1])}"
        mode = " [DRY-RUN]" if self._config.dry_run else ""
        logger.info(
            "Plan %s: Fenster %s → %s, %s, %d Lücke(n), ~%d Request(s)%s",
            instrument,
            _fmt(start),
            _fmt(end),
            have,
            len(ranges),
            estimated,
            mode,
        )
        for number, (range_start, range_end) in enumerate(ranges, start=1):
            logger.info(
                "  Lücke %d: %s → %s (%d Kerzen, ~%d Request(s))",
                number,
                _fmt(range_start),
                _fmt(range_end),
                candle_count(range_start, range_end),
                math.ceil(candle_count(range_start, range_end) / PAGE_SIZE),
            )

    def _make_chunk_logger(self, instrument: str) -> Callable[[datetime, datetime, int], None]:
        """Erzeugt den Progress-Callback pro 1000-Kerzen-Fenster."""

        def on_chunk(chunk_start: datetime, chunk_end: datetime, count: int) -> None:
            logger.info(
                "Backfill %s: %s → %s: %d Kerzen geladen",
                instrument,
                _fmt(chunk_start),
                _fmt(chunk_end),
                count,
            )

        return on_chunk


def _fmt(moment: datetime) -> str:
    """Formatiert einen Zeitstempel für Log-Nachrichten."""
    return moment.strftime(_FMT)
