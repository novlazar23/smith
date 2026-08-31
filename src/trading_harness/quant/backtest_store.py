"""Backtest-Storage für die Quant-Plattform (Phase 8, P8-2).

Verbindet ``BacktestEngine`` (P8-1) mit ``InfluxDBStore`` (P1-4):
das Ergebnis eines Backtest-Laufs wird als ein Punkt in das
``backtests``-Measurement geschrieben und per Flux-Query zurückgelesen.

Schemakontrakt (siehe ``tests/test_quant_backtest_integration.py``):

- Measurement: ``backtests``
- Tags: ``symbol``, ``exchange``
- Fields: ``total_trades``, ``winning_trades``, ``losing_trades``,
  ``win_rate``, ``total_pnl``, ``total_pnl_pct``, ``max_drawdown``,
  ``sharpe_ratio``, ``avg_trade_pnl``, ``profit_factor`` plus Kontext
  ``timeframe``

Aufrufkontrakte für ``run_and_store`` (Parallelearbeit P8-2/P8-3/P8-4):

1. ``(symbol, timeframe, candles, strategy, exchange="binance")`` —
   der Backtest wird mit ``engine.run`` ausgeführt (P8-4-Integration,
   P8-2-Skizze).
2. ``(symbol, timeframe, result, exchange=...)`` — ein bereits
   berechnetes ``BacktestResult`` wird nur persistiert (P8-3-API).

Semantik:

- Ein Punkt pro Backtest-Lauf (Lauf-Summary); die Equity-Kurve wird
  nicht persistiert (kein InfluxDB-Feld für unbeschränkt lange Serien).
- ``trades_stored`` zählt die Trades des Backtest-Laufs (unabhängig
  davon, ob der Store verfügbar war); ``stored`` ist True nur, wenn
  tatsächlich ein Punkt in einen verfügbaren Store geschrieben wurde.
- ``result`` hält das vollständige ``BacktestResult`` der Engine
  (Kontrakt 2: das übergebene Ergebnis).
- Ein Lauf ohne Trades erzeugt keinen Punkt: er trägt keine
  Handelsinformation (Muster: ``forward_outcomes_store``).
- Bei Kontrakt 2 (fremdberechnetes Ergebnis) trägt der Punkt den
  aktuellen UTC-Zeitstempel — es sind keine Kerzen übergeben.
- Engine-Fehler werden abgefangen und geloggt: der Lauf liefert ein
  leeres ``BacktestResult`` mit ``stored=False``, statt die Exception
  an den Aufrufer zu werfen.
- Store nicht verfügbar (``is_available=False``): kein Schreibversuch
  (``stored=False``), ``get_results`` liefert eine leere Liste.
- ``get_backtests`` ist ein Alias für ``get_results`` (P8-3-Vertrag).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, TypeGuard

from trading_harness.quant.backtesting import BacktestEngine, BacktestResult
from trading_harness.quant.influxdb_client import FieldDict, InfluxDBStore

logger = logging.getLogger(__name__)

_NANOSECONDS_PER_SECOND: int = 1_000_000_000

# Measurement-Name (Schema-Kontrakt, siehe Modul-Docstring und P8-4-Test).
BACKTESTS_MEASUREMENT: str = "backtests"

# Statistik-Fields eines Backtest-Laufs (Schema-Kontrakt).
BACKTEST_STAT_FIELDS: tuple[str, ...] = (
    "total_trades",
    "winning_trades",
    "losing_trades",
    "win_rate",
    "total_pnl",
    "total_pnl_pct",
    "max_drawdown",
    "sharpe_ratio",
    "avg_trade_pnl",
    "profit_factor",
)

# Ganzzahlige Statistik-Fields (werden beim Lesen als int zurückgegeben).
BACKTEST_INT_FIELDS: frozenset[str] = frozenset(
    {"total_trades", "winning_trades", "losing_trades"}
)

# Rückblickfenster für get_results ohne Startzeitpunkt (Flux-Dauer, keine Magie).
_DEFAULT_LOOKBACK: str = "-30d"


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


def _result_fields(result: BacktestResult, timeframe: str) -> FieldDict:
    """Mappt ein Backtest-Ergebnis auf seine InfluxDB-Fields (Kontext ``timeframe``)."""
    return {
        "total_trades": int(result.total_trades),
        "winning_trades": int(result.winning_trades),
        "losing_trades": int(result.losing_trades),
        "win_rate": float(result.win_rate),
        "total_pnl": float(result.total_pnl),
        "total_pnl_pct": float(result.total_pnl_pct),
        "max_drawdown": float(result.max_drawdown),
        "sharpe_ratio": float(result.sharpe_ratio),
        "avg_trade_pnl": float(result.avg_trade_pnl),
        "profit_factor": float(result.profit_factor),
        "timeframe": timeframe,
    }


def _empty_backtest_result(engine: BacktestEngine, symbol: str, timeframe: str) -> BacktestResult:
    """Leeres Engine-Ergebnis für abgefangene Engine-Fehler (Muster ``_empty_result``)."""
    return BacktestResult(
        symbol=symbol,
        timeframe=timeframe,
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        win_rate=0.0,
        total_pnl=0.0,
        total_pnl_pct=0.0,
        max_drawdown=0.0,
        sharpe_ratio=0.0,
        avg_trade_pnl=0.0,
        profit_factor=0.0,
        trades=[],
        equity_curve=[engine.initial_capital],
    )


def _compute_epoch(candles: list[dict]) -> int:
    """Zeitstempel des Backtest-Punkts (Epoch-Sekunden).

    Zeit der letzten Kerze mit parsebarem ``time``; bei keiner Kerze
    (oder ohne parsebare Zeitstempel) der aktuelle UTC-Zeitpunkt.
    """
    for candle in reversed(candles):
        epoch = _parse_timestamp(candle.get("time"))
        if epoch is not None:
            return epoch
    return int(time.time())


@dataclass(frozen=True)
class BacktestStoreResult:
    """Ergebnis eines Run+Store-Laufs.

    ``trades_stored`` ist die Anzahl der Trades des Backtest-Laufs
    (unabhängig davon, ob der Store verfügbar war); ``stored`` ist True
    nur, wenn tatsächlich ein Punkt in einen verfügbaren Store
    geschrieben wurde. ``result`` hält das vollständige
    ``BacktestResult`` (bei Kontrakt 2: das übergebene Ergebnis).
    """

    symbol: str
    timeframe: str
    trades_stored: int
    stored: bool
    result: BacktestResult | None = field(default=None, repr=False, compare=False)


class BacktestStore:
    """Schreibt Backtest-Ergebnisse in InfluxDB und liest sie zurück.

    - Ein Punkt pro Backtest-Lauf im ``backtests``-Measurement; Tags
      ``symbol`` / ``exchange``, Fields laut Schema-Kontrakt.
    - ``bucket`` wird aus der Store-Konfiguration gelesen
      (``store._bucket``), damit Queries ohne zusätzliche Konfiguration
      laufen (Muster: ``FeatureStore``).
    - Alle Store-Zugriffe laufen asynchron (``InfluxDBStore`` ist async).
    """

    def __init__(
        self,
        store: InfluxDBStore,
        engine: BacktestEngine | None = None,
    ) -> None:
        """Initialisiert den BacktestStore auf dem gegebenen Store (lazy, keine Verbindung)."""
        self._store = store
        self._engine = engine or BacktestEngine()
        self._bucket = str(getattr(store, "_bucket", "quant"))

    # ------------------------------------------------------------------
    # Schreibpfad
    # ------------------------------------------------------------------

    async def run_and_store(self, *args: Any, **kwargs: Any) -> BacktestStoreResult:
        """Run backtest and store results.

        Unterstützte Aufrufkontrakte (siehe Modul-Docstring):

        1. ``(symbol, timeframe, candles, strategy, exchange="binance")``
           — der Backtest wird mit ``engine.run`` ausgeführt.
        2. ``(symbol, timeframe, result, exchange=...)`` — ein bereits
           berechnetes ``BacktestResult`` wird nur persistiert.

        Der Rückgabewert ist ein ``BacktestStoreResult`` mit
        ``.symbol``/``.timeframe``/``.trades_stored``/``.stored``/``.result``.
        Ein Lauf ohne Trades erzeugt keinen Punkt. Bei Engine-Fehlern
        wird die Exception abgefangen und geloggt; es wird dann ein
        leeres ``BacktestResult`` mit ``stored=False`` zurückgegeben.
        Bei nicht verfügbarem Store (``is_available=False``) wird nicht
        geschrieben: ``stored=False`` (der Backtest läuft trotzdem).
        """
        symbol = kwargs.pop("symbol", None)
        timeframe = kwargs.pop("timeframe", None)
        exchange = kwargs.pop("exchange", None)
        if kwargs:
            raise TypeError(f"unexpected keyword arguments: {', '.join(sorted(kwargs))!r}")

        # Kontrakt 2 (P8-3): fremdberechnetes BacktestResult → nur persistieren.
        result = next((a for a in args if isinstance(a, BacktestResult)), None)
        if result is not None:
            rest = [a for a in args if not isinstance(a, BacktestResult)]
            if len(rest) > 2:
                raise TypeError("run_and_store() accepts at most (symbol, timeframe, result)")
            if symbol is None and rest:
                symbol = rest[0]
            if timeframe is None and len(rest) >= 2:
                timeframe = rest[1]
            return await self._persist_result(
                result,
                str(symbol or ""),
                str(timeframe or "1m"),
                str(exchange or "binance"),
            )

        # Kontrakt 1 (P8-4/P8-2): (symbol, timeframe, candles, strategy[, exchange])
        if len(args) < 4:
            raise TypeError(
                "run_and_store() requires (symbol, timeframe, candles, strategy[, exchange]) "
                "or (symbol, timeframe, result[, exchange=...])"
            )
        if len(args) > 5:
            raise TypeError("run_and_store() accepts at most 5 positional arguments")
        if not isinstance(args[0], str) or not isinstance(args[1], str):
            raise TypeError("run_and_store() symbol and timeframe must be strings")
        if symbol is None:
            symbol = args[0]
        if timeframe is None:
            timeframe = args[1]
        if exchange is None and len(args) == 5:
            exchange = args[4]
        raw_candles: object = args[2]
        if not isinstance(raw_candles, (list, tuple)):
            raise TypeError("candles must be a list of OHLCV dicts")
        candles: list[dict] = list(raw_candles)
        strategy = args[3]
        return await self._run_engine_and_store(
            candles,
            strategy,
            str(symbol),
            str(timeframe),
            str(exchange or "binance"),
        )

    # ------------------------------------------------------------------
    # Lesezugriff (Flux-Query über InfluxDBStore.query)
    # ------------------------------------------------------------------

    async def get_results(
        self,
        symbol: str,
        timeframe: str,
        start: str | None = None,
        end: str | None = None,
    ) -> list[dict]:
        """Query stored backtest results.

        Liest gespeicherte Backtest-Ergebnisse aus InfluxDB (aufsteigend).
        Jeder Eintrag enthält ``time`` (UTC-ISO-String), ``symbol``,
        ``exchange``, ``timeframe`` sowie die vorhandenen Statistik-Fields
        (Zählungen als int, Kennzahlen als float). ``start``/``end`` sind
        UTC-ISO-Zeichenketten (naive Zeichenketten sind UTC); ohne
        ``start`` gilt ein 30-Tage-Rückblick. Bei nicht verfügbarem Store
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
            entry: dict[str, Any] = {
                "time": time_iso,
                "symbol": record.get("symbol"),
                "exchange": record.get("exchange"),
                "timeframe": timeframe,
            }
            for name in BACKTEST_STAT_FIELDS:
                value = record.get(name)
                if _is_number(value):
                    entry[name] = int(value) if name in BACKTEST_INT_FIELDS else float(value)
            result.append(entry)
        return result

    async def get_backtests(
        self,
        symbol: str,
        timeframe: str,
        start: str | None = None,
        end: str | None = None,
    ) -> list[dict]:
        """Alias für ``get_results`` (P8-3-Endpunkt-Vertrag)."""
        return await self.get_results(symbol, timeframe, start, end)

    # ------------------------------------------------------------------
    # Private Hilfsmethoden
    # ------------------------------------------------------------------

    async def _run_engine_and_store(
        self,
        candles: list[dict],
        strategy: Callable[[list[dict], int], Any],
        symbol: str,
        timeframe: str,
        exchange: str,
    ) -> BacktestStoreResult:
        """Führt ``engine.run`` aus und persistiert das Ergebnis."""
        try:
            result = self._engine.run(candles, strategy, symbol=symbol, timeframe=timeframe)
        except Exception:
            logger.warning(
                "Backtest run failed for %s/%s — returning empty result",
                symbol,
                timeframe,
                exc_info=True,
            )
            result = _empty_backtest_result(self._engine, symbol, timeframe)
        return await self._persist_result(result, symbol, timeframe, exchange, _compute_epoch(candles))

    async def _persist_result(
        self,
        result: BacktestResult,
        symbol: str,
        timeframe: str,
        exchange: str,
        epoch: int | None = None,
    ) -> BacktestStoreResult:
        """Schreibt das Lauf-Ergebnis als einen Punkt (falls Store + Trades vorliegen)."""
        stored = False
        if result.total_trades > 0:
            # InfluxDBStore verbindet sich lazy. Ohne den Health-Check wäre
            # ``is_available`` nach jedem Prozessstart zunächst False und das
            # erste Backtest-Ergebnis würde stillschweigend verworfen.
            await self._store.health_check()
        if result.total_trades > 0 and self._store.is_available:
            timestamp = (epoch if epoch is not None else int(time.time())) * _NANOSECONDS_PER_SECOND
            await self._store.write_points(
                measurement=BACKTESTS_MEASUREMENT,
                tags={"symbol": symbol, "exchange": exchange},
                fields=_result_fields(result, timeframe),
                timestamp=timestamp,
            )
            stored = True
        return BacktestStoreResult(
            symbol=symbol,
            timeframe=timeframe,
            trades_stored=result.total_trades,
            stored=stored,
            result=result,
        )

    def _build_query(
        self,
        symbol: str,
        start: str | None,
        end: str | None,
    ) -> str:
        """Flux: Backtest-Punkte im Zeitraum, gepivotet, aufsteigend sortiert."""
        start_clause = f'"{_flux_escape(start)}"' if start is not None else _DEFAULT_LOOKBACK
        range_clause = f"range(start: {start_clause}"
        if end is not None:
            range_clause += f', stop: "{_flux_escape(end)}"'
        range_clause += ")"
        lines = [
            f'from(bucket: "{_flux_escape(self._bucket)}")',
            f"|> {range_clause}",
            f'|> filter(fn: (r) => r._measurement == "{BACKTESTS_MEASUREMENT}")',
            f'|> filter(fn: (r) => r["symbol"] == "{_flux_escape(symbol)}")',
            '|> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")',
            '|> sort(columns: ["_time"])',
        ]
        return "\n".join(lines)
