"""Similarity-Storage für die Quant-Plattform (Phase 5, P5-2).

Verbindet ``SimilarityEngine`` (P5-1) mit ``InfluxDBStore`` (P1-4):
ähnliche historische Muster werden pro Lauf on-the-fly aus den
übergebenen Historiedaten berechnet und in einem thread-sicheren
In-Memory-Cache gehalten.

MVP-Semantik:

- Similarity-Ergebnisse sind nicht vorab gespeichert — die Berechnung
  läuft auf-the-fly gegen die übergebene ``history``. Der
  ``InfluxDBStore`` wird als Abhängigkeit für
  Schnittstellenkonsistenz mit ``RegimeStore`` / ``AnomalyStore``
  übernommen; eine InfluxDB-Persistierung ist eine geplante
  Erweiterung und findet im MVP nicht statt (die Store-Methoden sind
  synchron, die ``InfluxDBStore``-API asynchron).
- ``find_and_store`` berechnet die Matches über die Engine
  (``find_similar_with_candles``) und legt sie pro ``(symbol,
  timeframe)`` im Cache ab — die letzte Suche gewinnt.
- ``get_similar_patterns`` liest den Cache zurück (Kopien, in
  Engine-Reihenfolge nach aufsteigender Distanz); ``start``/``end``
  filtern nach Fensterstartzeit, ungültige Zeitstempel werfen
  ``ValueError`` (Muster: ``RegimeStore`` / ``AnomalyStore``).
- Engine-Fehler werden abgefangen und geloggt: ``find_and_store``
  liefert dann ein leeres Ergebnis statt einer Exception und berührt
  den Cache nicht (Muster: ``InfluxDBStore`` degradiert, statt zu
  werfen).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from trading_harness.quant.influxdb_client import InfluxDBStore
from trading_harness.quant.similarity import SimilarityEngine, SimilarMatch

logger = logging.getLogger(__name__)


@dataclass
class SimilarityStoreResult:
    """Ergebnis eines Find+Store-Laufs.

    ``matches_found`` und die Best-Werte beschreiben das Ergebnis der
    Engine (unabhängig vom Cache); bei Engine-Fehlern bleibt der Lauf
    graceful (0 Matches, ``None``-Bestwerte, keine Exception).
    """

    query_length: int
    matches_found: int
    best_distance: float | None
    best_correlation: float | None


def _parse_timestamp(value: object) -> int | None:
    """Parst einen Zeitstempel (datetime oder ISO-8601) zu Epoch-Sekunden.

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


def _epoch_to_iso(epoch: int) -> str:
    """Formatiert Epoch-Sekunden als UTC-ISO-String (``...Z``)."""
    return datetime.fromtimestamp(epoch, tz=UTC).isoformat().replace("+00:00", "Z")


def _window_time(match: SimilarMatch) -> str:
    """Zeit des Similarity-Fensters (erste Kerze) als UTC-ISO-String.

    Ohne Kerzen (oder ohne parsebaren Zeitstempel) der aktuelle
    UTC-Zeitpunkt (Muster: ``RegimeStore._detect_epoch``).
    """
    for candle in match.candles:
        epoch = _parse_timestamp(candle.get("time"))
        if epoch is not None:
            return _epoch_to_iso(epoch)
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


def _entry(
    match: SimilarMatch,
    symbol: str,
    timeframe: str,
    exchange: str,
) -> dict[str, Any]:
    """Mappt einen Similarity-Match auf einen Cache-/Query-Eintrag."""
    return {
        "time": _window_time(match),
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": timeframe,
        "start_index": match.start_index,
        "end_index": match.end_index,
        "distance": match.distance,
        "correlation": match.correlation,
    }


class SimilarityStore:
    """Berechnet Similarity-Ergebnisse über die Engine und puffert sie.

    - ``store`` (``InfluxDBStore``) wird für Schnittstellenkonsistenz
      mit den anderen Quant-Stores gehalten; der MVP liest und
      schreibt nicht dorthin (geplante Erweiterung).
    - Der Cache ist pro ``(symbol, timeframe)`` keyiert und wird pro
      Lauf ersetzt (letzte Suche gewinnt).
    - Thread-Safe: alle Cache-Zugriffe laufen unter einem RLock
      (Muster: ``services/kill_switch.py``).
    """

    def __init__(
        self,
        store: InfluxDBStore,
        engine: SimilarityEngine | None = None,
    ) -> None:
        """Initialisiert den SimilarityStore (lazy, keine Store-Verbindung)."""
        self._store = store
        self._engine = engine or SimilarityEngine()
        self._cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self._lock = threading.RLock()

    @property
    def engine(self) -> SimilarityEngine:
        """Die verwendete SimilarityEngine (Injektion oder Default)."""
        return self._engine

    # ------------------------------------------------------------------
    # Berechnung + Cache
    # ------------------------------------------------------------------

    def find_and_store(
        self,
        symbol: str,
        timeframe: str,
        query: list[dict],
        history: list[dict],
        exchange: str = "binance",
    ) -> SimilarityStoreResult:
        """Findet ähnliche Muster über die Engine und legt sie im Cache ab.

        Bei Engine-Fehlern (z. B. defekte Kerzendaten) wird gewarnt und
        ein leeres Ergebnis geliefert — der Store wirft aus diesem
        Grund keine Exception und berührt den Cache nicht.
        """
        try:
            result = self._engine.find_similar_with_candles(query, history)
        except Exception:
            logger.warning(
                "SimilarityEngine failed for %s/%s — returning empty result",
                symbol,
                timeframe,
                exc_info=True,
            )
            return SimilarityStoreResult(
                query_length=len(query),
                matches_found=0,
                best_distance=None,
                best_correlation=None,
            )

        entries = [
            _entry(match, symbol, timeframe, exchange) for match in result.matches
        ]

        with self._lock:
            self._cache[(symbol, timeframe)] = entries

        return SimilarityStoreResult(
            query_length=result.query_length,
            matches_found=len(result.matches),
            best_distance=result.best_distance,
            best_correlation=result.best_correlation,
        )

    # ------------------------------------------------------------------
    # Lesezugriff (Cache)
    # ------------------------------------------------------------------

    def get_similar_patterns(
        self,
        symbol: str,
        timeframe: str,
        start: str | None = None,
        end: str | None = None,
    ) -> list[dict]:
        """Liest die letzten Similarity-Ergebnisse aus dem Cache (Kopien).

        Jeder Eintrag enthält ``time`` (UTC-ISO-Zeit des
        Fensterbeginns), ``symbol``, ``exchange``, ``timeframe``,
        ``start_index``, ``end_index``, ``distance`` und
        ``correlation``. Die Reihenfolge ist die der Engine
        (aufsteigende Distanz). ``start``/``end`` sind
        UTC-ISO-Zeichenketten und filtern nach Fensterstartzeit;
        ungültige Zeitstempel werfen ``ValueError``.
        """
        if start is not None and _parse_timestamp(start) is None:
            raise ValueError(f"invalid start timestamp: {start!r}")
        if end is not None and _parse_timestamp(end) is None:
            raise ValueError(f"invalid end timestamp: {end!r}")

        with self._lock:
            entries = [dict(entry) for entry in self._cache.get((symbol, timeframe), [])]

        start_epoch = _parse_timestamp(start) if start is not None else None
        end_epoch = _parse_timestamp(end) if end is not None else None
        if start_epoch is None and end_epoch is None:
            return entries

        filtered: list[dict[str, Any]] = []
        for entry in entries:
            epoch = _parse_timestamp(entry.get("time"))
            if epoch is None:
                continue
            if start_epoch is not None and epoch < start_epoch:
                continue
            if end_epoch is not None and epoch > end_epoch:
                continue
            filtered.append(entry)
        return filtered
