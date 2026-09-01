"""Kern-Logik des Orchestrator-Services.

Jeder Zyklus holt pro Instrument die letzten N Kerzen aus ClickHouse,
führt die ``OrchestratorPipeline`` (Analyse + Konsens + Audit) mit einem
frisch erstellten Agenten-Ensemble aus und persistiert die Entscheidung
in PostgreSQL (``shadow_decisions``). Der Agenten-Status ist über
``ORCHESTRATOR_AGENT_STATUS`` konfigurierbar (Default ``ACTIVE`` =
Realbetrieb; ``SHADOW`` = Beobachtungsmodus). Es findet **nie**
Order-Ausführung statt — unabhängig vom Feature-Flag
``live_trading_enabled``.
"""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from packages.agents import AnomalyAgent, ChartAgent, HistoricalAnalogyAgent
from packages.agents.base import AgentConfig, AgentType, BaseAgent
from packages.governance.feature_flags import feature_flags
from packages.orchestrator.pipeline import OrchestratorPipeline
from packages.orchestrator.second_round import RoundContext
from packages.persistence.clickhouse.engine import (
    ClickHouseConfig,
    ClickHouseEngine,
    create_ch_engine,
)
from packages.persistence.sqlalchemy.engine import DatabaseConfig, SQLAlchemyEngine
from packages.schemas.agent_report import AgentReport, AgentStatus
from sqlalchemy import text
from sqlalchemy.engine import Connection

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_SECONDS = 900.0
DEFAULT_INSTRUMENTS = "BTC/USDT,ETH/USDT"
DEFAULT_CANDLE_LIMIT = 200
DEFAULT_MIN_CANDLES = 30
DEFAULT_HORIZON = "15m"
DEFAULT_CANDLE_VENUE = "BINANCE_FUTURES"
DEFAULT_AGENT_STATUS = "ACTIVE"
HEARTBEAT_PATH = Path("/tmp/orchestrator_heartbeat")

INSERT_SHADOW_DECISION = text(
    """
    INSERT INTO shadow_decisions
        (run_id, instrument, decision, confidence, reason,
         first_round_count, second_round_count, latency_ms, errors, warnings)
    VALUES
        (:run_id, :instrument, :decision, :confidence, :reason,
         :first_round_count, :second_round_count, :latency_ms, :errors, :warnings)
    """
)


@dataclass(frozen=True)
class OrchestratorServiceConfig:
    """Laufzeit-Konfiguration des Orchestrator-Services."""

    interval_seconds: float = DEFAULT_INTERVAL_SECONDS
    instruments: tuple[str, ...] = ()
    candle_limit: int = DEFAULT_CANDLE_LIMIT
    min_candles: int = DEFAULT_MIN_CANDLES
    horizon: str = DEFAULT_HORIZON
    agent_status: str = DEFAULT_AGENT_STATUS
    heartbeat_path: Path = HEARTBEAT_PATH
    log_level: str = "INFO"


@dataclass(frozen=True)
class CandleWindow:
    """Ein OHLCV-Kerzenfenster (aufsteigend nach open_time)."""

    open: NDArray[np.float64]
    high: NDArray[np.float64]
    low: NDArray[np.float64]
    close: NDArray[np.float64]
    volume: NDArray[np.float64]


@dataclass(frozen=True)
class ShadowDecision:
    """Eine zu persistierende Shadow-Entscheidung."""

    run_id: str
    instrument: str
    decision: str
    confidence: float
    reason: str
    first_round_count: int
    second_round_count: int
    latency_ms: float
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class CandleProvider(Protocol):
    """Schnittstelle zum Lesen von Kerzen aus dem Zeitreihen-Speicher."""

    def fetch_candles(self, instrument: str, limit: int) -> CandleWindow | None:
        """Liefert die letzten ``limit`` Kerzen oder None bei Fehlschlag."""
        ...


class ClickHouseCandleProvider:
    """Liest OHLCV-Kerzen aus der ClickHouse-Tabelle ``candles``.

    Die Abfrage filtert zusätzlich auf eine Venue (Default: Env
    ``CANDLE_VENUE``, sonst ``BINANCE_FUTURES``), damit der Shadow-Zyklus
    nur Kerzen der konfigurierten Datenquelle liest.
    """

    def __init__(self, engine: ClickHouseEngine, venue: str | None = None) -> None:
        """Initialisiert den Provider mit einer ClickHouse-Engine.

        Args:
            engine: ClickHouse-Engine.
            venue: Venue-Filter für die Kerzenabfrage; Default ist der
                Env-Wert ``CANDLE_VENUE`` (sonst ``BINANCE_FUTURES``).
        """
        self._engine = engine
        self._venue = (
            venue if venue is not None else os.environ.get("CANDLE_VENUE", DEFAULT_CANDLE_VENUE)
        )

    def fetch_candles(self, instrument: str, limit: int) -> CandleWindow | None:
        """Holt die letzten ``limit`` Kerzen aufsteigend nach open_time.

        Args:
            instrument: Handelspaar (z.B. "BTC/USDT").
            limit: Maximale Anzahl der Kerzen.

        Returns:
            CandleWindow oder None, wenn keine Kerzen abrufbar sind.
        """
        escaped = self._escape(instrument)
        escaped_venue = self._escape(self._venue)
        query = (
            f"SELECT open, high, low, close, volume "
            f"FROM candles "
            f"WHERE instrument = '{escaped}' AND venue = '{escaped_venue}' "
            f"ORDER BY open_time DESC "
            f"LIMIT {int(limit)}"
        )
        names, rows = self._engine.query(query)
        if not rows:
            return None
        # DESC abgefragt (neueste zuerst) → umdrehen für aufsteigende Zeitfolge
        index = {name: i for i, name in enumerate(names)}
        order = [index[name] for name in ("open", "high", "low", "close", "volume")]
        reversed_rows = list(reversed(rows))
        return CandleWindow(
            open=np.array([row[order[0]] for row in reversed_rows], dtype=np.float64),
            high=np.array([row[order[1]] for row in reversed_rows], dtype=np.float64),
            low=np.array([row[order[2]] for row in reversed_rows], dtype=np.float64),
            close=np.array([row[order[3]] for row in reversed_rows], dtype=np.float64),
            volume=np.array([row[order[4]] for row in reversed_rows], dtype=np.float64),
        )

    @staticmethod
    def _escape(value: str) -> str:
        """Escapt einen String für ein ClickHouse-String-Literal."""
        return value.replace("\\", "\\\\").replace("'", "\\'")


class ContextualAgent:
    """Adapter, der einen BaseAgent an die Second-Round-Schnittstelle anpasst.

    Die Pipeline ruft im Second Round ``analyze_with_context(context, market_data)``
    auf; die getrackten Agenten implementieren nur ``analyze``. Dieser Adapter
    delegiert beide Aufrufe an die deterministische OHLCV-Analyse — der
    First-Round-Kontext ändert das Ergebnis der deterministischen Analyse nicht.
    """

    def __init__(self, agent: BaseAgent) -> None:
        """Umschließt einen bestehenden Analyse-Agenten."""
        self._agent = agent

    @property
    def agent_id(self) -> str:
        """ID des umschlossenen Agenten."""
        return self._agent.agent_id

    def analyze(self, data: dict[str, NDArray[np.float64]]) -> AgentReport:
        """First Round: Analyse der OHLCV-Rohdaten."""
        return self._agent.analyze(data)

    def analyze_with_context(
        self,
        context: RoundContext,
        market_data: dict[str, NDArray[np.float64]],
    ) -> AgentReport:
        """Second Round: delegiert an analyze (deterministische OHLCV-Analyse)."""
        del context
        return self._agent.analyze(market_data)


def build_ensemble(
    instrument: str,
    horizon: str,
    agent_status: AgentStatus = AgentStatus.SHADOW,
) -> list[ContextualAgent]:
    """Erzeugt frische Agenten für einen Zyklus.

    Ensemble: AnomalyAgent (OHLCV-Statistik), HistoricalAnalogyAgent
    (DTW-Analogien, nutzt das eigene Kerzenfenster als Historie) und
    ChartAgent (Swing-Pivots, S/R-Level, BOS/CHoCH) — alle drei sind
    ausschließlich auf ein OHLCV-Fenster angewiesen.

    Args:
        instrument: Kanonisches Instrument, z.B. ``"BTC/USDT"``.
        horizon: Analyse-Horizont, z.B. ``"15m"``.
        agent_status: Lebenszyklus-Status der Agenten. Default ``SHADOW``
            (konservativ für direkte Aufrufe); der Service übergibt den
            konfigurierten Status (Default ``ACTIVE`` = Realbetrieb).
    """
    specs: list[tuple[str, AgentType, type[BaseAgent]]] = [
        ("anomaly", AgentType.ANOMALY, AnomalyAgent),
        ("historical_analogy", AgentType.HISTORICAL_ANALOGY, HistoricalAnalogyAgent),
        ("chart", AgentType.CHART, ChartAgent),
    ]
    agents: list[ContextualAgent] = []
    for agent_id, agent_type, agent_cls in specs:
        config = AgentConfig(
            agent_id=agent_id,
            agent_type=agent_type,
            instrument=instrument,
            horizon=horizon,
            status=agent_status,
        )
        agents.append(ContextualAgent(agent_cls(config=config)))
    return agents


def build_market_data(window: CandleWindow) -> dict[str, NDArray[np.float64]]:
    """Baut das market_data-Dict aus einem Kerzenfenster."""
    return {
        "open": window.open,
        "high": window.high,
        "low": window.low,
        "close": window.close,
        "volume": window.volume,
    }


def make_run_id(instrument: str, moment: datetime | None = None) -> str:
    """Erzeugt eine eindeutige Run-ID pro (Zyklus, Instrument)."""
    stamp = (moment or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return f"orch-{stamp}-{instrument}"


def write_heartbeat(path: Path) -> None:
    """Schreibt die Epoche (Sekunden) in die Heartbeat-Datei."""
    path.write_text(f"{int(time.time())}\n", encoding="utf-8")


def persist_shadow_decision(conn: Connection, decision: ShadowDecision) -> None:
    """Persistiert eine Shadow-Entscheidung in ``shadow_decisions``.

    Args:
        conn: SQLAlchemy-Connection (oder Duck-typ-Äquivalent).
        decision: Zu persistierender Shadow-Entscheidungs-Wert.
    """
    conn.execute(
        INSERT_SHADOW_DECISION,
        {
            "run_id": decision.run_id,
            "instrument": decision.instrument,
            "decision": decision.decision,
            "confidence": decision.confidence,
            "reason": decision.reason,
            "first_round_count": decision.first_round_count,
            "second_round_count": decision.second_round_count,
            "latency_ms": decision.latency_ms,
            "errors": "\n".join(decision.errors) if decision.errors else None,
            "warnings": "\n".join(decision.warnings) if decision.warnings else None,
        },
    )
    conn.commit()


class OrchestratorService:
    """Führt die Shadow-Pipeline zyklisch für alle Instrumente aus."""

    def __init__(
        self,
        config: OrchestratorServiceConfig,
        provider: CandleProvider,
        db: SQLAlchemyEngine,
        pipeline_factory: Callable[[], OrchestratorPipeline] = OrchestratorPipeline,
    ) -> None:
        """Initialisiert den Service.

        Args:
            config: Laufzeit-Konfiguration (Instrumente, Intervall, ...).
            provider: Kerzen-Anbieter (ClickHouse).
            db: SQLAlchemy-Engine-Wrapper für die Persistenz.
            pipeline_factory: Factory für die Pipeline (Testbarkeit).
        """
        self._config = config
        self._provider = provider
        self._db = db
        self._pipeline_factory = pipeline_factory

    @property
    def config(self) -> OrchestratorServiceConfig:
        """Aktuelle Konfiguration."""
        return self._config

    def run_cycle(self) -> int:
        """Führt einen kompletten Shadow-Zyklus aus.

        Ein Fehler bei einem Instrument beendet den Zyklus nicht: Es wird
        (sofern möglich) eine Fehlerzeile persistiert und mit dem nächsten
        Instrument fortgefahren. Die Heartbeat-Datei wird nach jedem Zyklus
        geschrieben, unabhängig von Einzelfehlern.

        Returns:
            Anzahl der persistierten Entscheidungen (inklusive Fehlerzeilen).
        """
        live_enabled = feature_flags.is_enabled("live_trading_enabled")
        logger.info(
            "Shadow-Zyklus gestartet: %d Instrumente, live_trading_enabled=%s (keine Order-Ausführung)",
            len(self._config.instruments),
            live_enabled,
        )
        persisted = 0
        for instrument in self._config.instruments:
            try:
                persisted += self._run_instrument(instrument)
            except Exception as exc:
                logger.exception("Shadow-Zyklus für %s fehlgeschlagen: %s", instrument, exc)
                persisted += self._persist_error(instrument, exc)
        try:
            write_heartbeat(self._config.heartbeat_path)
        except OSError as exc:
            logger.warning("Heartbeat-Datei nicht schreibbar: %s", exc)
        return persisted

    def _run_instrument(self, instrument: str) -> int:
        """Führt die Pipeline für ein einzelnes Instrument aus.

        Returns:
            1, wenn eine Entscheidung persistiert wurde, sonst 0 (Skip).
        """
        window = self._provider.fetch_candles(instrument, self._config.candle_limit)
        if window is None or len(window.close) < self._config.min_candles:
            available = 0 if window is None else len(window.close)
            logger.warning(
                "Instrument %s übersprungen: %d Kerzen verfügbar, mindestens %d erforderlich",
                instrument,
                available,
                self._config.min_candles,
            )
            return 0

        market_data = build_market_data(window)
        run_id = make_run_id(instrument)
        agents = build_ensemble(
            instrument, self._config.horizon, AgentStatus[self._config.agent_status]
        )
        started = time.perf_counter()
        result = self._pipeline_factory().run(
            run_id=run_id,
            instrument=instrument,
            agents=agents,
            market_data=market_data,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0

        consensus = result.consensus
        decision = ShadowDecision(
            run_id=run_id,
            instrument=instrument,
            decision=result.decision,
            confidence=consensus.confidence if consensus is not None else 0.0,
            reason=consensus.reason if consensus is not None else "kein Konsens (Fehler im Zyklus)",
            first_round_count=len(result.first_round_reports),
            second_round_count=len(result.second_round_reports),
            latency_ms=latency_ms,
            errors=list(result.errors),
            warnings=list(result.warnings),
        )
        with self._db.engine.connect() as conn:
            persist_shadow_decision(conn, decision)
        logger.info(
            "Shadow-Cyklus fertig: %s -> %s (confidence=%.4f, %d/%d Agenten, %.0f ms)",
            instrument,
            decision.decision,
            decision.confidence,
            decision.first_round_count,
            len(agents),
            decision.latency_ms,
        )
        return 1

    def _persist_error(self, instrument: str, exc: BaseException) -> int:
        """Persistiert eine Fehlerzeile (decision='error'), falls möglich.

        Returns:
            1, wenn die Fehlerzeile persistiert wurde, sonst 0.
        """
        message = f"{type(exc).__name__}: {exc}"
        decision = ShadowDecision(
            run_id=make_run_id(instrument),
            instrument=instrument,
            decision="error",
            confidence=0.0,
            reason=message,
            first_round_count=0,
            second_round_count=0,
            latency_ms=0.0,
            errors=[message],
        )
        try:
            with self._db.engine.connect() as conn:
                persist_shadow_decision(conn, decision)
        except Exception as persist_exc:
            logger.error("Fehlerzeile für %s nicht persistierbar: %s", instrument, persist_exc)
            return 0
        return 1


def parse_instruments(raw: str) -> tuple[str, ...]:
    """Parst eine Kommagetrennte-Liste von Instrumenten."""
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def config_from_env() -> OrchestratorServiceConfig:
    """Baut die Service-Konfiguration aus Umgebungsvariablen.

    Umgebungsvariablen (Defaults in Klammern):
      ORCHESTRATOR_INTERVAL_SECONDS (900), ORCHESTRATOR_INSTRUMENTS
      (BTC/USDT,ETH/USDT), ORCHESTRATOR_CANDLE_LIMIT (200),
      ORCHESTRATOR_MIN_CANDLES (30), ORCHESTRATOR_HORIZON (15m),
      ORCHESTRATOR_AGENT_STATUS (ACTIVE), ORCHESTRATOR_HEARTBEAT
      (/tmp/orchestrator_heartbeat), LOG_LEVEL (INFO).
    """
    raw_instruments = os.environ.get("ORCHESTRATOR_INSTRUMENTS", DEFAULT_INSTRUMENTS)
    try:
        interval = float(
            os.environ.get("ORCHESTRATOR_INTERVAL_SECONDS", str(DEFAULT_INTERVAL_SECONDS))
        )
    except ValueError:
        logger.warning(
            "Ungültiges ORCHESTRATOR_INTERVAL_SECONDS → Default %.0f", DEFAULT_INTERVAL_SECONDS
        )
        interval = DEFAULT_INTERVAL_SECONDS
    try:
        candle_limit = int(os.environ.get("ORCHESTRATOR_CANDLE_LIMIT", str(DEFAULT_CANDLE_LIMIT)))
    except ValueError:
        candle_limit = DEFAULT_CANDLE_LIMIT
    try:
        min_candles = int(os.environ.get("ORCHESTRATOR_MIN_CANDLES", str(DEFAULT_MIN_CANDLES)))
    except ValueError:
        min_candles = DEFAULT_MIN_CANDLES
    heartbeat = Path(os.environ.get("ORCHESTRATOR_HEARTBEAT", str(HEARTBEAT_PATH)))
    raw_status = os.environ.get("ORCHESTRATOR_AGENT_STATUS", DEFAULT_AGENT_STATUS).strip().upper()
    if raw_status not in ("ACTIVE", "SHADOW"):
        logger.warning(
            "Ungültiges ORCHESTRATOR_AGENT_STATUS '%s' → Default '%s'",
            raw_status,
            DEFAULT_AGENT_STATUS,
        )
        raw_status = DEFAULT_AGENT_STATUS
    return OrchestratorServiceConfig(
        interval_seconds=interval,
        instruments=parse_instruments(raw_instruments),
        candle_limit=candle_limit,
        min_candles=min_candles,
        horizon=os.environ.get("ORCHESTRATOR_HORIZON", DEFAULT_HORIZON),
        agent_status=raw_status,
        heartbeat_path=heartbeat,
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
    )


def build_db_engine() -> SQLAlchemyEngine:
    """Erzeugt die PostgreSQL-Engine aus DB_*-Umgebungsvariablen."""
    return SQLAlchemyEngine(
        DatabaseConfig(
            host=os.environ.get("DB_HOST", "postgres"),
            port=int(os.environ.get("DB_PORT", "5432")),
            database=os.environ.get("DB_NAME", "trading"),
            user=os.environ.get("DB_USER", "orchestra"),
            password=os.environ.get("DB_PASSWORD", ""),
        )
    )


def build_ch_provider() -> ClickHouseCandleProvider:
    """Erzeugt den ClickHouse-Kerzen-Provider aus CH_*-Umgebungsvariablen.

    Der Venue-Filter stammt aus ``CANDLE_VENUE`` (Default
    ``BINANCE_FUTURES``) und wird beim Provider-Setup eingelesen.
    """
    engine = create_ch_engine(
        ClickHouseConfig(
            host=os.environ.get("CH_HOST", "clickhouse"),
            port=int(os.environ.get("CH_PORT", "8123")),
            database=os.environ.get("CH_DB", "trading_events"),
            user="orchestra",
            password=os.environ.get("CH_PASSWORD", ""),
        )
    )
    return ClickHouseCandleProvider(engine)


def build_service(
    config: OrchestratorServiceConfig | None = None,
    provider: CandleProvider | None = None,
    db: SQLAlchemyEngine | None = None,
) -> OrchestratorService:
    """Setzt den Service aus Env-Defaults und injizierten Abhängigkeiten zusammen."""
    return OrchestratorService(
        config if config is not None else config_from_env(),
        provider if provider is not None else build_ch_provider(),
        db if db is not None else build_db_engine(),
    )


def install_signal_handlers(stop_event: threading.Event) -> None:
    """Hängt SIGTERM/SIGINT an den Stop-Event (graceful Shutdown).

    Der laufende Zyklus fährt zu Ende; der Loop bricht danach ab.
    """

    def _handler(signum: int, frame: types.FrameType | None) -> None:
        del frame
        logger.info("Signal %d empfangen — laufender Zyklus wird zu Ende gefahren", signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def run_service(service: OrchestratorService, stop_flag: Callable[[], bool]) -> None:
    """Läuft der Zyklus-Loop: sofort erster Zyklus, dann im Intervall.

    Ein fehlgeschlagener Zyklus beendet den Loop nicht (Catch + Log).
    Der Loop bricht ab, sobald ``stop_flag`` True liefert (graceful:
    ein laufender Zyklus fährt zu Ende).

    Args:
        service: Der zu betreibende Service.
        stop_flag: Liefert True, wenn der Loop beendet werden soll (Signal-Hook).
    """
    while True:
        try:
            service.run_cycle()
        except Exception as exc:
            logger.exception("Zyklus fehlgeschlagen (Loop läuft weiter): %s", exc)
        if stop_flag():
            break
        # In kleinen Schritten warten, damit Stop-Signale zügig wirken
        waited = 0.0
        while waited < service.config.interval_seconds and not stop_flag():
            step = min(1.0, service.config.interval_seconds - waited)
            time.sleep(step)
            waited += step
