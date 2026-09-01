"""Kern-Logik des Demo-Traders (virtuelles Trading mit imaginärem Geld).

Jeder Zyklus holt pro Instrument die letzten N Kerzen aus ClickHouse
(Venue = ``CANDLE_VENUE``), führt die ``OrchestratorPipeline`` mit einem
frisch erstellten **ACTIVEn** Agenten-Ensemble aus (echte, gewichtete
Konsens-Entscheidungen) und mappt die Entscheidung auf Paper-Trades:

  - ``LONG_BIAS``  → Market-Buy (Menge = DEMO_TRADE_NOTIONAL / letzter Close)
  - ``SHORT_BIAS`` → Glattstellung der offenen Position (``close_position``)
  - ``NO_TRADE`` u. a. → kein Trade

Ausgeführt wird ausschließlich über den getrackten ``PaperExecutor``
(Slippage, Kommission, 10%-Positions-Limit) — es werden **nie** reale
Orders platziert. Jeder ausgeführte Trade landet in PostgreSQL
(``demo_trades``); nach jedem Zyklus wird der Account-Snapshot nach
``demo_account`` upgepusht.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import signal
import threading
import time
import types
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from apps.orchestrator_service.service import (
    CandleWindow,
    ContextualAgent,
    build_market_data,
    parse_instruments,
    write_heartbeat,
)
from packages.agents import AnomalyAgent, ChartAgent, HistoricalAnalogyAgent
from packages.agents.base import AgentConfig, AgentType, BaseAgent
from packages.consensus import ConsensusDecision
from packages.observability.mlflow_client import MLflowClient
from packages.orchestrator.pipeline import OrchestratorPipeline
from packages.paper import PaperAccount, PaperExecutor, Trade, TradeDirection
from packages.persistence.clickhouse.engine import (
    ClickHouseConfig,
    ClickHouseEngine,
    create_ch_engine,
)
from packages.persistence.sqlalchemy.engine import DatabaseConfig, SQLAlchemyEngine
from packages.schemas.agent_report import AgentStatus
from sqlalchemy import text
from sqlalchemy.engine import Connection

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_SECONDS = 300.0
DEFAULT_INSTRUMENTS = "BTC/USDT,ETH/USDT"
DEFAULT_INITIAL_CASH = 100000.0
DEFAULT_TRADE_NOTIONAL = 2000.0
DEFAULT_MIN_CONFIDENCE = 0.3
DEFAULT_CANDLE_VENUE = "BINANCE_FUTURES"
DEFAULT_CANDLE_LIMIT = 200
DEFAULT_MIN_CANDLES = 30
DEFAULT_HORIZON = "15m"
DEFAULT_ACCOUNT_ID = "demo"
HEARTBEAT_PATH = Path("/tmp/demo_trader_heartbeat")

ACTION_BUY = "BUY"
ACTION_SELL = "SELL"
ACTION_NONE = "NONE"

INSERT_DEMO_TRADE = text(
    """
    INSERT INTO demo_trades
        (trade_id, instrument, direction, quantity, price,
         filled_price, filled_quantity, commission, slippage, status)
    VALUES
        (:trade_id, :instrument, :direction, :quantity, :price,
         :filled_price, :filled_quantity, :commission, :slippage, :status)
    """
)

UPSERT_DEMO_ACCOUNT = text(
    """
    INSERT INTO demo_account
        (account_id, cash, equity, initial_cash, total_pnl,
         total_commission, total_trades, positions, updated_at)
    VALUES
        (:account_id, :cash, :equity, :initial_cash, :total_pnl,
         :total_commission, :total_trades, CAST(:positions AS jsonb), now())
    ON CONFLICT (account_id) DO UPDATE SET
        cash = EXCLUDED.cash,
        equity = EXCLUDED.equity,
        initial_cash = EXCLUDED.initial_cash,
        total_pnl = EXCLUDED.total_pnl,
        total_commission = EXCLUDED.total_commission,
        total_trades = EXCLUDED.total_trades,
        positions = EXCLUDED.positions,
        updated_at = now()
    """
)


@dataclass(frozen=True)
class DemoTraderConfig:
    """Laufzeit-Konfiguration des Demo-Traders."""

    interval_seconds: float = DEFAULT_INTERVAL_SECONDS
    instruments: tuple[str, ...] = ()
    initial_cash: float = DEFAULT_INITIAL_CASH
    trade_notional: float = DEFAULT_TRADE_NOTIONAL
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    candle_venue: str = DEFAULT_CANDLE_VENUE
    candle_limit: int = DEFAULT_CANDLE_LIMIT
    min_candles: int = DEFAULT_MIN_CANDLES
    horizon: str = DEFAULT_HORIZON
    account_id: str = DEFAULT_ACCOUNT_ID
    heartbeat_path: Path = HEARTBEAT_PATH
    log_level: str = "INFO"


@dataclass(frozen=True)
class DemoTradePlan:
    """Aus einer Konsens-Entscheidung abgeleiteter Paper-Trade-Plan."""

    action: str
    quantity: float
    price: float
    reason: str


class CandleSource(Protocol):
    """Schnittstelle zum Lesen von Kerzenfenstern aus dem Zeitreihen-Speicher."""

    def fetch_candles(self, instrument: str, limit: int) -> CandleWindow | None:
        """Liefert die letzten ``limit`` Kerzen oder None bei Fehlschlag."""
        ...


class DemoCandleProvider:
    """Liest OHLCV-Kerzen eines bestimmten Venues aus ``candles``.

    Anders als der Orchestrator-Provider wird hier zusätzlich nach
    ``venue`` gefiltert (``CANDLE_VENUE``), damit das Demo-Trading auf
    den Kerzen der konfigurierten Venue läuft.
    """

    def __init__(self, engine: ClickHouseEngine, venue: str) -> None:
        """Initialisiert den Provider mit einer ClickHouse-Engine und einem Venue."""
        self._engine = engine
        self._venue = venue

    def fetch_candles(self, instrument: str, limit: int) -> CandleWindow | None:
        """Holt die letzten ``limit`` Kerzen des Venues aufsteigend nach open_time.

        Args:
            instrument: Handelspaar (z.B. "BTC/USDT").
            limit: Maximale Anzahl der Kerzen.

        Returns:
            CandleWindow oder None, wenn keine Kerzen abrufbar sind.
        """
        escaped = self._escape(instrument)
        venue_escaped = self._escape(self._venue)
        query = (
            "SELECT open, high, low, close, volume "
            "FROM candles "
            f"WHERE instrument = '{escaped}' AND venue = '{venue_escaped}' "
            f"ORDER BY open_time DESC LIMIT {int(limit)}"
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


def build_active_ensemble(instrument: str, horizon: str) -> list[ContextualAgent]:
    """Erzeugt ein frisches ACTIVE-Ensemble für einen Zyklus.

    Dasselbe Agenten-Trio wie der Orchestrator-Service (AnomalyAgent,
    HistoricalAnalogyAgent, ChartAgent) — aber mit ``AgentStatus.ACTIVE``,
    damit der gewichtete Konsens echte Entscheidungen statt NO_TRADE
    ("No active agents (all shadow)") liefert.
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
            status=AgentStatus.ACTIVE,
        )
        agents.append(ContextualAgent(agent_cls(config=config)))
    return agents


def plan_trade(
    decision: str,
    confidence: float,
    min_confidence: float,
    latest_close: float,
    trade_notional: float,
) -> DemoTradePlan:
    """Mappt eine Konsens-Entscheidung auf einen Paper-Trade (long-only).

    Mapping inklusive Konfidenz-Gate:
      - ``confidence < min_confidence`` → kein Trade
        ("Signal unter Konfidenz-Gate → kein Trade")
      - ``LONG_BIAS``  → Market-Buy mit quantity = trade_notional / latest_close
      - ``SHORT_BIAS`` → Glattstellung der offenen Position (close_position)
      - sonstige Werte (``NO_TRADE``, ``RANGE``, ...) → kein Trade

    Args:
        decision: Konsens-Entscheidung (z.B. "LONG_BIAS").
        confidence: Konsens-Konfidenz (0..1).
        min_confidence: Konfidenz-Schwelle; ab ihr (inklusive) wird gehandelt.
        latest_close: Letzter Schlusskurs des Instruments (Markt-Preis).
        trade_notional: Ziel-Nominal pro Order in USDT.

    Returns:
        DemoTradePlan mit Aktion (BUY/SELL/NONE), Menge, Preis und Grund.
    """
    if confidence < min_confidence:
        return DemoTradePlan(
            action=ACTION_NONE,
            quantity=0.0,
            price=0.0,
            reason="Signal unter Konfidenz-Gate → kein Trade",
        )
    if decision == ConsensusDecision.LONG_BIAS.value:
        if latest_close <= 0:
            return DemoTradePlan(
                action=ACTION_NONE,
                quantity=0.0,
                price=0.0,
                reason="LONG_BIAS, aber ungültiger Schlusskurs → kein Trade",
            )
        return DemoTradePlan(
            action=ACTION_BUY,
            quantity=trade_notional / latest_close,
            price=latest_close,
            reason="LONG_BIAS → Market-Buy",
        )
    if decision == ConsensusDecision.SHORT_BIAS.value:
        return DemoTradePlan(
            action=ACTION_SELL,
            quantity=0.0,
            price=latest_close,
            reason="SHORT_BIAS → Position glattstellen",
        )
    return DemoTradePlan(
        action=ACTION_NONE,
        quantity=0.0,
        price=0.0,
        reason=f"{decision} → kein Trade",
    )


def build_account_snapshot(account: PaperAccount) -> dict[str, Any]:
    """Baut die ``demo_account``-Zeile aus dem Zustand des Paper-Accounts.

    Args:
        account: Der zu snapshotende PaperAccount.

    Returns:
        Dict mit account_id, cash, equity, initial_cash, total_pnl,
        total_commission, total_trades und positions (JSON-Liste aus
        instrument/quantity/avg_price/opened_at).
    """
    positions = [
        {
            "instrument": symbol,
            "quantity": position.quantity,
            "avg_price": position.avg_price,
            "opened_at": (
                position.opened_at.isoformat() if position.opened_at is not None else None
            ),
        }
        for symbol, position in account.positions.items()
    ]
    return {
        "account_id": account.account_id,
        "cash": account.cash,
        "equity": account.equity,
        "initial_cash": account.initial_cash,
        "total_pnl": account.total_pnl,
        "total_commission": account.total_commission,
        "total_trades": account.total_trades,
        "positions": positions,
    }


def persist_demo_trade(conn: Connection, trade: Trade) -> None:
    """Persistiert einen ausgeführten Paper-Trade in ``demo_trades``.

    Args:
        conn: SQLAlchemy-Connection (oder Duck-typ-Äquivalent).
        trade: Der ausgeführte Trade des PaperExecutors.
    """
    conn.execute(
        INSERT_DEMO_TRADE,
        {
            "trade_id": trade.trade_id,
            "instrument": trade.instrument,
            "direction": trade.direction.value,
            "quantity": trade.quantity,
            "price": trade.price,
            "filled_price": trade.filled_price,
            "filled_quantity": trade.filled_quantity,
            "commission": trade.commission,
            "slippage": trade.slippage,
            "status": trade.status,
        },
    )
    conn.commit()


def persist_account_snapshot(conn: Connection, snapshot: dict[str, Any]) -> None:
    """Upsertet den Account-Snapshot in ``demo_account`` (PKey account_id).

    Args:
        conn: SQLAlchemy-Connection (oder Duck-typ-Äquivalent).
        snapshot: Das von build_account_snapshot() erzeugte Snapshot-Dict.
    """
    conn.execute(
        UPSERT_DEMO_ACCOUNT,
        {
            "account_id": snapshot["account_id"],
            "cash": snapshot["cash"],
            "equity": snapshot["equity"],
            "initial_cash": snapshot["initial_cash"],
            "total_pnl": snapshot["total_pnl"],
            "total_commission": snapshot["total_commission"],
            "total_trades": snapshot["total_trades"],
            "positions": json.dumps(snapshot["positions"]),
        },
    )
    conn.commit()


def make_run_id(instrument: str, moment: datetime | None = None) -> str:
    """Erzeugt eine eindeutige Run-ID pro (Zyklus, Instrument)."""
    stamp = (moment or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return f"demo-{stamp}-{instrument}"


def log_cycle_to_mlflow(
    config: DemoTraderConfig,
    instrument: str,
    decision: str,
    confidence: float,
    plan: DemoTradePlan,
    trade: Trade | None,
    account: PaperAccount,
    latest_close: float,
    client_factory: Callable[[], Any] | None = None,
) -> None:
    """Recordet einen Demo-Zyklus als MLflow-Run (optional, nie fatal).

    Die Runs füllen den ML-Tab der zentralen Web-UI (Entscheidungen,
    Konfidenz und Konto-Verlauf pro Zyklus). Opt-in über
    ``MLFLOW_ENABLED=true`` (z. B. im Compose-Block); ein unerreichbarer
    MLflow-Server führt nur zu einer Warnung, nie zu einem Cycle-Abbruch.

    Umgebungsvariablen: MLFLOW_ENABLED (false), MLFLOW_TRACKING_URI
    (http://mlflow:5000), MLFLOW_EXPERIMENT_NAME (demo-trader).
    """
    if os.environ.get("MLFLOW_ENABLED", "false").strip().lower() not in ("1", "true", "yes", "on"):
        return
    factory = client_factory
    if factory is None:
        factory = lambda: MLflowClient(  # noqa: E731
            tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
            experiment_name=os.environ.get("MLFLOW_EXPERIMENT_NAME", "demo-trader"),
        )
    try:
        client = factory()
        run_id = client.start_run(
            run_name=f"{instrument}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}",
            tags={"instrument": instrument, "component": "demo-trader", "venue": config.candle_venue},
        )
        try:
            client.log_parameters(
                run_id,
                {
                    "min_confidence": config.min_confidence,
                    "trade_notional": config.trade_notional,
                    "horizon": config.horizon,
                    "decision": decision,
                    "action": plan.action,
                    "latest_close": latest_close,
                },
            )
            client.log_metrics(
                run_id,
                {
                    "confidence": confidence,
                    "equity": account.equity,
                    "cash": account.cash,
                    "total_pnl": account.total_pnl,
                    "total_trades": account.total_trades,
                    "trade_executed": 1.0 if trade is not None else 0.0,
                },
            )
            client.end_run(run_id, status="FINISHED")
        except Exception:
            with contextlib.suppress(Exception):
                client.end_run(run_id, status="FAILED")
            raise
    except Exception as exc:
        logger.warning("MLflow-Run-Aufzeichnung für %s fehlgeschlagen (nicht fatal): %s", instrument, exc)


class DemoTrader:
    """Führt den Demo-Zyklus (Analyse → Paper-Trade → Persistenz) aus.

    Der PaperExecutor-Account lebt in-Process über alle Zyklen hinweg;
    bei einem Neustart des Prozesses setzt er auf initial_cash zurück
    (die Trade-Historie bleibt in ``demo_trades`` erhalten).
    """

    def __init__(
        self,
        config: DemoTraderConfig,
        provider: CandleSource,
        db: SQLAlchemyEngine,
        executor: PaperExecutor,
        pipeline_factory: Callable[[], OrchestratorPipeline] = OrchestratorPipeline,
    ) -> None:
        """Initialisiert den Demo-Trader.

        Args:
            config: Laufzeit-Konfiguration (Instrumente, Intervall, ...).
            provider: Kerzen-Anbieter (ClickHouse, Venue gefiltert).
            db: SQLAlchemy-Engine-Wrapper für die Persistenz.
            executor: PaperExecutor (imaginäres Geld, Slippage, Kommission).
            pipeline_factory: Factory für die Pipeline (Testbarkeit).
        """
        self._config = config
        self._provider = provider
        self._db = db
        self._executor = executor
        self._pipeline_factory = pipeline_factory
        self._account = executor.create_account(config.account_id)

    @property
    def config(self) -> DemoTraderConfig:
        """Aktuelle Konfiguration."""
        return self._config

    @property
    def account(self) -> PaperAccount:
        """Der in-Process Paper-Account."""
        return self._account

    def run_cycle(self) -> int:
        """Führt einen kompletten Demo-Zyklus aus.

        Ein Fehler bei einem Instrument beendet den Zyklus nicht: Es wird
        geloggt und mit dem nächsten Instrument fortgefahren. Nach allen
        Instrumenten wird der Account-Snapshot upgepusht und die
        Heartbeat-Datei geschrieben — unabhängig von Einzelfehlern.

        Returns:
            Anzahl der ausgeführten Paper-Trades.
        """
        logger.info(
            "Demo-Zyklus gestartet: %d Instrumente, venue=%s, account=%s",
            len(self._config.instruments),
            self._config.candle_venue,
            self._config.account_id,
        )
        executed = 0
        for instrument in self._config.instruments:
            try:
                executed += self._run_instrument(instrument)
            except Exception as exc:
                logger.exception("Demo-Zyklus für %s fehlgeschlagen: %s", instrument, exc)
        self._persist_snapshot()
        try:
            write_heartbeat(self._config.heartbeat_path)
        except OSError as exc:
            logger.warning("Heartbeat-Datei nicht schreibbar: %s", exc)
        return executed

    def _run_instrument(self, instrument: str) -> int:
        """Führt Analyse und Paper-Trade für ein einzelnes Instrument aus.

        Returns:
            1, wenn ein Trade ausgeführt wurde, sonst 0 (Skip/no-op).
        """
        window = self._provider.fetch_candles(instrument, self._config.candle_limit)
        if window is None or len(window.close) < self._config.min_candles:
            available = 0 if window is None else len(window.close)
            logger.warning(
                "Instrument %s übersprungen: %d Kerzen (venue=%s) verfügbar, mindestens %d erforderlich",
                instrument,
                available,
                self._config.candle_venue,
                self._config.min_candles,
            )
            return 0

        latest_close = float(window.close[-1])
        market_data = build_market_data(window)
        run_id = make_run_id(instrument)
        agents = build_active_ensemble(instrument, self._config.horizon)
        result = self._pipeline_factory().run(
            run_id=run_id,
            instrument=instrument,
            agents=agents,
            market_data=market_data,
        )

        consensus = result.consensus
        decision = result.decision
        confidence = consensus.confidence if consensus is not None else 0.0
        plan = plan_trade(
            decision,
            confidence,
            self._config.min_confidence,
            latest_close,
            self._config.trade_notional,
        )

        trade: Trade | None = None
        if plan.action == ACTION_BUY:
            trade = self._executor.submit_order(
                self._account, instrument, TradeDirection.BUY, plan.quantity, plan.price
            )
        elif plan.action == ACTION_SELL:
            logger.info("%s: %s", instrument, plan.reason)
            trade = self._executor.close_position(self._account, instrument)
            if trade is None:
                logger.info("%s: keine offene Position → Glattstellung entfällt", instrument)
        else:
            logger.info("%s: %s", instrument, plan.reason)

        if trade is not None:
            with self._db.engine.connect() as conn:
                persist_demo_trade(conn, trade)
            executed = 1
        else:
            executed = 0

        trade_repr = (
            f"{trade.direction.value} {trade.filled_quantity}@{trade.filled_price}"
            if trade is not None
            else "—"
        )
        logger.info(
            "Demo-Zyklus: %s -> %s (confidence=%.4f, Trade=%s)",
            instrument,
            decision,
            confidence,
            trade_repr,
        )
        log_cycle_to_mlflow(
            self._config, instrument, decision, confidence, plan, trade, self._account, latest_close
        )
        return executed

    def _persist_snapshot(self) -> None:
        """Upsertet den Account-Snapshot in ``demo_account`` (nicht fatal)."""
        snapshot = build_account_snapshot(self._account)
        try:
            with self._db.engine.connect() as conn:
                persist_account_snapshot(conn, snapshot)
        except Exception as exc:
            logger.error("Account-Snapshot nicht persistierbar: %s", exc)


def config_from_env() -> DemoTraderConfig:
    """Baut die Demo-Trader-Konfiguration aus Umgebungsvariablen.

    Umgebungsvariablen (Defaults in Klammern):
      DEMO_INTERVAL_SECONDS (300), DEMO_INSTRUMENTS (BTC/USDT,ETH/USDT),
      DEMO_INITIAL_CASH (100000), DEMO_TRADE_NOTIONAL (2000),
      DEMO_MIN_CONFIDENCE (0.3), CANDLE_VENUE (BINANCE_FUTURES),
      DEMO_HEARTBEAT (/tmp/demo_trader_heartbeat), LOG_LEVEL (INFO).
    """
    raw_instruments = os.environ.get("DEMO_INSTRUMENTS", DEFAULT_INSTRUMENTS)
    try:
        interval = float(os.environ.get("DEMO_INTERVAL_SECONDS", str(DEFAULT_INTERVAL_SECONDS)))
    except ValueError:
        logger.warning(
            "Ungültiges DEMO_INTERVAL_SECONDS → Default %.0f", DEFAULT_INTERVAL_SECONDS
        )
        interval = DEFAULT_INTERVAL_SECONDS
    try:
        initial_cash = float(os.environ.get("DEMO_INITIAL_CASH", str(DEFAULT_INITIAL_CASH)))
    except ValueError:
        initial_cash = DEFAULT_INITIAL_CASH
    try:
        trade_notional = float(os.environ.get("DEMO_TRADE_NOTIONAL", str(DEFAULT_TRADE_NOTIONAL)))
    except ValueError:
        trade_notional = DEFAULT_TRADE_NOTIONAL
    try:
        min_confidence = float(os.environ.get("DEMO_MIN_CONFIDENCE", str(DEFAULT_MIN_CONFIDENCE)))
    except ValueError:
        min_confidence = DEFAULT_MIN_CONFIDENCE
    return DemoTraderConfig(
        interval_seconds=interval,
        instruments=parse_instruments(raw_instruments),
        initial_cash=initial_cash,
        trade_notional=trade_notional,
        min_confidence=min_confidence,
        candle_venue=os.environ.get("CANDLE_VENUE", DEFAULT_CANDLE_VENUE),
        heartbeat_path=Path(os.environ.get("DEMO_HEARTBEAT", str(HEARTBEAT_PATH))),
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


def build_ch_provider(venue: str) -> DemoCandleProvider:
    """Erzeugt den ClickHouse-Kerzen-Provider aus CH_*-Umgebungsvariablen."""
    engine = create_ch_engine(
        ClickHouseConfig(
            host=os.environ.get("CH_HOST", "clickhouse"),
            port=int(os.environ.get("CH_PORT", "8123")),
            database=os.environ.get("CH_DB", "trading_events"),
            user="orchestra",
            password=os.environ.get("CH_PASSWORD", ""),
        )
    )
    return DemoCandleProvider(engine, venue)


def build_trader(
    config: DemoTraderConfig | None = None,
    provider: CandleSource | None = None,
    db: SQLAlchemyEngine | None = None,
    executor: PaperExecutor | None = None,
) -> DemoTrader:
    """Setzt den Demo-Trader aus Env-Defaults und injizierten Abhängigkeiten zusammen."""
    cfg = config if config is not None else config_from_env()
    return DemoTrader(
        cfg,
        provider if provider is not None else build_ch_provider(cfg.candle_venue),
        db if db is not None else build_db_engine(),
        executor if executor is not None else PaperExecutor(initial_cash=cfg.initial_cash),
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


def run_service(trader: DemoTrader, stop_flag: Callable[[], bool]) -> None:
    """Läuft der Zyklus-Loop: sofort erster Zyklus, dann im Intervall.

    Ein fehlgeschlagener Zyklus beendet den Loop nicht (Catch + Log).
    Der Loop bricht ab, sobald ``stop_flag`` True liefert (graceful:
    ein laufender Zyklus fährt zu Ende).

    Args:
        trader: Der zu betreibende Demo-Trader.
        stop_flag: Liefert True, wenn der Loop beendet werden soll (Signal-Hook).
    """
    while True:
        try:
            trader.run_cycle()
        except Exception as exc:
            logger.exception("Zyklus fehlgeschlagen (Loop läuft weiter): %s", exc)
        if stop_flag():
            break
        # In kleinen Schritten warten, damit Stop-Signale zügig wirken
        waited = 0.0
        while waited < trader.config.interval_seconds and not stop_flag():
            step = min(1.0, trader.config.interval_seconds - waited)
            time.sleep(step)
            waited += step
