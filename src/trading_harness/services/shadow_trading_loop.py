"""Shadow Trading Loop (WI-ST-05, Spec ST.1–ST.15).

Kontinuierlicher, kontrollierbarer Shadow-Trading-Loop über Live-Marktdaten:

1. Pro Symbol und Iteration: Ticker -> deterministischer Snapshot (ST.14)
   -> ``TradingRun`` -> Analyse aller ``ACTIVE``/``CHAMPION``-Agenten (Z4)
   -> deterministische Signal-Aggregation (ST.6) -> deterministische
   Risk-Engine -> Entscheidung.
2. Freigegebene Trades laufen ausschließlich über den Paper-Stack
   (``ShadowExecutionBackend``) — niemals Live-Execution (Epic-B3, ST.3).
3. Mark-to-Market + Stop-/Target-Checks genau einmal pro Iteration (ST.10).
4. Persistenter Session-State inkl. Tages-Budget, Fehlerschutz (F6),
   Self-Stop bei Kill Switch (ST.11) und verletzten Flags (S2), Audit-Trail
   (ST.13) und deterministischen IDs (ST.14).

Dokumentierte Abweichung (ST.10): ``PortfolioTracker.update()`` erhält
``position_store.all()`` (offene + geschlossene Positionen) und nicht nur
die offenen Positionen, damit ``current_equity`` den realisierten PnL
abbildet (Akzeptanzkriterium ST.10). Der in der Portfolio-History
gespeicherte Shadow-State zeigt danach wieder nur offene Positionen.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any, Protocol

from trading_harness.config import Settings
from trading_harness.models import (
    AgentSignal,
    MarketSnapshot,
    OutcomeRecord,
    PortfolioState,
    RiskDecision,
    RunOutcome,
    RunState,
    ShadowEvaluationSummary,
    ShadowLoopStatus,
    ShadowSessionState,
    ShadowTradingRecord,
    ShadowTradingStatus,
    utcnow,
)
from trading_harness.services.agent_genome_store import AgentGenomeStore
from trading_harness.services.agent_runtime import AgentRuntime
from trading_harness.services.evaluation import _compute_classification_metrics
from trading_harness.services.kill_switch import KillSwitch
from trading_harness.services.orchestrator import TradingRunService
from trading_harness.services.risk_engine import RiskEngine
from trading_harness.services.shadow_decision import (
    aggregate_signals,
    build_trade_proposal,
    no_trade_reason,
)
from trading_harness.services.shadow_execution_backend import ShadowExecutionBackend
from trading_harness.services.shadow_trading_state import (
    MAX_CONSECUTIVE_ERRORS,
    ShadowTradingStateStore,
)
from trading_harness.services.snapshot_store import SnapshotStore

logger = logging.getLogger(__name__)

__all__ = [
    "CryptoMarketDataProvider",
    "MarketDataProvider",
    "ShadowTradingLoop",
    "ShadowTradingStartError",
    "compute_shadow_evaluation",
]


class ShadowTradingStartError(Exception):
    """``start()``-Wächterfehler mit maschinenlesbarem ``code`` (Spec ST.2).

    Codes: ``LIVE_EXECUTION_MUST_BE_DISABLED``, ``SHADOW_TRADING_DISABLED``,
    ``ALREADY_RUNNING``, ``NO_SYMBOLS_CONFIGURED``.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class MarketDataProvider(Protocol):
    """Synchroner Ticker-Provider: ``get_ticker(symbol) -> {"last": float, ...}``."""

    def get_ticker(self, symbol: str) -> dict[str, float]: ...


class _RouterLike(Protocol):
    """Struktureller Vertrag für den verdrahteten Crypto-Router (Spec ST.3).

    Bewusst strukturell (keine harte Import-Abhängigkeit der Crypto-
    Schicht): der Shadow-Loop bleibt dadurch von Live-Exchange-Imports
    getrennt; die API-Wiring (WI-ST-06) übergibt den echten Router.
    """

    def get_ticker(self, symbol: str) -> dict[str, float]: ...


class CryptoMarketDataProvider:
    """Produktions-Ticker-Quelle über den verdrahteten Crypto-Router (Spec ST.3)."""

    def __init__(self, router: _RouterLike) -> None:
        self._router = router

    def get_ticker(self, symbol: str) -> dict[str, float]:
        return self._router.get_ticker(symbol)


def _snapshot_id(symbol: str, timestamp: datetime, data: dict[str, Any]) -> str:
    """Deterministischer Snapshot-ID (ST.14).

    Spiegelbild von ``SnapshotStore._hash`` (gleiche Payload, gleiche
    Kanonisierung); Präfix ``snap-`` + 16 Hex-Zerlen. Gleiche
    (Symbol, Zeitstempel, Daten) ergeben den gleichen ID.
    """
    payload = {
        "symbol": symbol,
        "timestamp": timestamp.isoformat(),
        "data": data,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return f"snap-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


def _config_version(settings: Settings, risk_engine: RiskEngine) -> str:
    """Deterministische Config-Version: SHA-256 über Risk-Policy + Shadow-Flags."""
    payload = {
        "risk_policy": risk_engine.policy,
        "shadow_min_confidence": settings.shadow_min_confidence,
        "shadow_stop_loss_fraction": settings.shadow_stop_loss_fraction,
        "shadow_min_risk_reward": settings.shadow_min_risk_reward,
        "shadow_max_decisions_per_day": settings.shadow_max_decisions_per_day,
        "shadow_trading_enabled": settings.shadow_trading_enabled,
        "live_execution_enabled": settings.live_execution_enabled,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ShadowTradingLoop:
    """Kontinuierlicher Shadow-Trading-Loop (Spec ST.1–ST.15).

    Nur lesend gegenüber dem Markt, ausführend ausschließlich über den
    Paper-Stack. Lifecycle-Guards (ST.2), Budget (ST.4), Risiko (ST.6),
    Persistenz (ST.8), M2M (ST.10), Kill Switch (ST.11), Evaluation
    (ST.12), Audit (ST.13) und Determinismus (ST.14) sind Teil der Loop.
    """

    def __init__(
        self,
        market_data: MarketDataProvider,
        backend: ShadowExecutionBackend,
        agent_runtime: AgentRuntime,
        trading_run_service: TradingRunService,
        snapshot_store: SnapshotStore,
        risk_engine: RiskEngine,
        kill_switch: KillSwitch,
        agent_source: AgentGenomeStore,
        settings: Settings,
        state_store: ShadowTradingStateStore,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._market_data = market_data
        self._backend = backend
        self._agent_runtime = agent_runtime
        self._runs = trading_run_service
        self._snapshot_store = snapshot_store
        self._risk_engine = risk_engine
        self._kill_switch = kill_switch
        self._agent_source = agent_source
        self._settings = settings
        self._state_store = state_store
        self._clock = clock

        # Direktzugriff auf den Paper-Stack für M2M/Stop-Checks (ST.10).
        stack = backend.stack
        self._position_manager = stack.position_manager
        self._position_store = stack.position_store
        self._portfolio_tracker = stack.portfolio_tracker

        self._task: asyncio.Task[None] | None = None
        self._budget_announced_date: str = ""

    # ------------------------------------------------------------------
    # Öffentliche Schnittstelle (Spec §5.3)
    # ------------------------------------------------------------------

    def status(self) -> ShadowSessionState:
        """Defensive Deep Copy des Sessions-Zustands (ST.8)."""
        return self._state_store.status()

    async def start(self) -> None:
        """Startet den Loop nach Prüfung der Guards (Spec ST.2, S1)."""
        settings = self._settings
        if settings.live_execution_enabled:
            raise ShadowTradingStartError(
                "LIVE_EXECUTION_MUST_BE_DISABLED",
                "Shadow Trading startet nicht, solange Live-Execution aktiv ist",
            )
        if not settings.shadow_trading_enabled:
            raise ShadowTradingStartError(
                "SHADOW_TRADING_DISABLED",
                "Shadow Trading ist in der Konfiguration deaktiviert",
            )

        state = self._state_store.status()
        task = self._task
        if task is not None and not task.done():
            if state.status is ShadowLoopStatus.RUNNING:
                raise ShadowTradingStartError("ALREADY_RUNNING", "Shadow Trading läuft bereits")
            # Stales Task nach einem Self-Stop: kontrolliert beenden,
            # bevor ein zweiter Loop gestartet wird.
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._task = None

        if not settings.shadow_trading_symbols:
            raise ShadowTradingStartError(
                "NO_SYMBOLS_CONFIGURED",
                "Keine Shadow-Trading-Symbole konfiguriert",
            )

        updates: dict[str, Any] = {
            "status": "RUNNING",
            "started_at": self._clock(),
            "stopped_at": None,
            "restart_required": False,
            "symbols": list(settings.shadow_trading_symbols),
            "interval_seconds": settings.shadow_loop_interval_seconds,
        }
        if state.iteration_count == 0:
            updates["start_equity"] = settings.shadow_start_equity
            updates["current_equity"] = settings.shadow_start_equity
        self._state_store.update_state(updates)

        self._runs.audit(
            "SHADOW_LOOP_STARTED",
            "shadow_session",
            state.session_id,
            new_state="RUNNING",
            details={
                "flags": {
                    "shadow_trading_enabled": settings.shadow_trading_enabled,
                    "live_execution_enabled": settings.live_execution_enabled,
                },
                "symbols": list(settings.shadow_trading_symbols),
            },
        )
        logger.info(
            "SHADOW_LOOP_STARTED symbols=%s interval=%ss session=%s",
            settings.shadow_trading_symbols,
            settings.shadow_loop_interval_seconds,
            state.session_id,
        )
        self._task = asyncio.create_task(self.run())

    async def stop(self, reason: str = "API") -> None:
        """Stopp der Loop (Spec ST.2); idempotent, auch nach Self-Stop."""
        state = self._state_store.status()
        task = self._task
        self._task = None
        if task is not None and not task.done():
            self._state_store.update_state({"status": "STOPPING"})
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._state_store.update_state(
            {
                "status": "STOPPED",
                "stopped_at": self._clock(),
                "restart_required": True,
            }
        )
        self._state_store.save()
        self._runs.audit(
            "SHADOW_LOOP_STOPPED",
            "shadow_session",
            state.session_id,
            new_state="STOPPED",
            details={"reason": reason},
        )
        logger.info("SHADOW_LOOP_STOPPED reason=%s session=%s", reason, state.session_id)

    async def run(self) -> None:
        """Laufende Iterationen bis zum Stopp (Spec ST.1, ST.8, F6)."""
        while True:
            if self._state_store.status().status is not ShadowLoopStatus.RUNNING:
                break
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # F6: Iterationsfehler zählen, Loop läuft weiter
                logger.exception("Shadow-Loop: unerwarteter Iterationsfehler")
                if self._state_store.record_iteration_error(f"ITERATION_ERROR: {exc!r}"):
                    self._self_stop(
                        "SHADOW_LOOP_STOPPED_ERRORS",
                        self._state_store.status().session_id,
                        details={"last_error": f"ITERATION_ERROR: {exc!r}"},
                    )
            if self._state_store.status().status is not ShadowLoopStatus.RUNNING:
                break
            await asyncio.sleep(max(0, self._settings.shadow_loop_interval_seconds))

    # ------------------------------------------------------------------
    # Eine Iteration (Spec ST.1)
    # ------------------------------------------------------------------

    async def run_once(self) -> dict[str, Any]:
        """Eine vollständige Iteration über alle konfigurierten Symbole.

        Returns:
            ``{"iteration", "symbols": [...], "equity", "duration_ms"}`` plus —
            nur bei genau einem Symbol — die Top-Level-Felder
            ``snapshot_id``/``run_id``/``decision`` (Spec ST.1).
        """
        settings = self._settings
        perf_start = time.monotonic()
        now = self._clock()

        state = self._state_store.status()
        today = now.strftime("%Y-%m-%d")
        if state.budget_date != today:
            self._state_store.update_state({"budget_date": today, "decisions_today": 0})
            state = self._state_store.status()

        iteration = state.iteration_count + 1
        session_id = state.session_id
        symbols = list(settings.shadow_trading_symbols)
        base_decisions = state.decisions_today
        kill_active = self._kill_switch.is_active()
        flags_violated = settings.live_execution_enabled or not settings.shadow_trading_enabled
        config_version = _config_version(settings, self._risk_engine)

        results: list[dict[str, Any]] = []
        tickers: dict[str, float] = {}
        last_error: str | None = None
        had_error = False
        decisions_made = 0

        if kill_active:
            # ST.11: keine Entscheidung, kein M2M — der Loop stoppt sich selbst.
            for symbol in symbols:
                results.append(self._skip_entry(symbol, "SKIPPED_KILL_SWITCH"))
                self._audit_iteration(
                    iteration=iteration,
                    symbol=symbol,
                    snapshot_id=None,
                    run_id=None,
                    decision="SKIPPED_KILL_SWITCH",
                    risk_result=None,
                    execution_result=None,
                    session_id=session_id,
                )
            self._self_stop(
                "SHADOW_LOOP_STOPPED_KILL_SWITCH",
                session_id,
                details={"iteration": iteration},
            )
        elif flags_violated:
            # S2: Flags während des Laufs verletzt — M2M läuft noch, dann Self-Stop.
            for symbol in symbols:
                results.append(self._skip_entry(symbol, "SKIPPED_FLAGS"))
                self._audit_iteration(
                    iteration=iteration,
                    symbol=symbol,
                    snapshot_id=None,
                    run_id=None,
                    decision="SKIPPED_FLAGS",
                    risk_result=None,
                    execution_result=None,
                    session_id=session_id,
                )
            await self._mark_to_market(tickers)
            self._self_stop(
                "SHADOW_LOOP_STOPPED_FLAGS_CHANGED",
                session_id,
                details={
                    "iteration": iteration,
                    "flags": {
                        "shadow_trading_enabled": settings.shadow_trading_enabled,
                        "live_execution_enabled": settings.live_execution_enabled,
                    },
                },
            )
        else:
            for symbol in symbols:
                if base_decisions + decisions_made >= settings.shadow_max_decisions_per_day:
                    # ST.4: Budget erschöpft — keine neuen Entscheidungen,
                    # M2M/Stop-Checks laufen weiter, Loop bleibt RUNNING.
                    results.append(self._skip_entry(symbol, "SKIPPED_BUDGET"))
                    self._audit_iteration(
                        iteration=iteration,
                        symbol=symbol,
                        snapshot_id=None,
                        run_id=None,
                        decision="SKIPPED_BUDGET",
                        risk_result=None,
                        execution_result=None,
                        session_id=session_id,
                    )
                    if self._budget_announced_date != today:
                        self._budget_announced_date = today
                        self._runs.audit(
                            "SHADOW_LOOP_STOPPED_BUDGET_EXHAUSTED",
                            "shadow_session",
                            session_id,
                            details={
                                "iteration": iteration,
                                "decisions_today": base_decisions + decisions_made,
                                "max_decisions_per_day": settings.shadow_max_decisions_per_day,
                            },
                        )
                    continue
                entry, consumed, error = await self._process_symbol(
                    symbol,
                    iteration,
                    config_version=config_version,
                    tickers=tickers,
                )
                results.append(entry)
                if consumed:
                    decisions_made += 1
                if error is not None:
                    had_error = True
                    last_error = error
            await self._mark_to_market(tickers)

        # F6: Fehlerschutz über aufeinanderfolgende Iterationen.
        if had_error:
            if self._state_store.record_iteration_error(last_error or "UNKNOWN_ERROR"):
                self._self_stop(
                    "SHADOW_LOOP_STOPPED_ERRORS",
                    session_id,
                    details={"last_error": last_error},
                )
                logger.error(
                    "SHADOW_LOOP_STOPPED_ERRORS nach %d aufeinanderfolgenden Iterationsfehlern: %s",
                    MAX_CONSECUTIVE_ERRORS,
                    last_error,
                )
        else:
            self._state_store.record_iteration_success()

        self._state_store.update_state(
            {
                "iteration_count": iteration,
                "last_iteration_at": self._clock(),
                "decisions_today": base_decisions + decisions_made,
                "budget_date": today,
                "symbols": symbols,
            }
        )
        self._state_store.save()

        equity = self._state_store.status().current_equity
        duration_ms = (time.monotonic() - perf_start) * 1000.0
        # O1/ST.13: strukturiertes INFO-Log pro Symbol pro Iteration.
        for entry in results:
            logger.info(
                "SHADOW_ITERATION iteration=%s symbol=%s snapshot_id=%s run_id=%s "
                "decision=%s risk_result=%s execution_result=%s equity=%.2f duration_ms=%.3f",
                iteration,
                entry["symbol"],
                entry["snapshot_id"],
                entry["run_id"],
                entry["decision"],
                entry["risk_result"],
                entry["execution_result"],
                equity,
                duration_ms,
            )
        result: dict[str, Any] = {
            "iteration": iteration,
            "symbols": results,
            "equity": equity,
            "duration_ms": duration_ms,
        }
        if len(results) == 1:
            only = results[0]
            result["snapshot_id"] = only["snapshot_id"]
            result["run_id"] = only["run_id"]
            result["decision"] = only["decision"]
        return result

    # ------------------------------------------------------------------
    # Symbol-Pipeline (Spec ST.1, ST.6, ST.7, ST.13, ST.14)
    # ------------------------------------------------------------------

    async def _process_symbol(
        self,
        symbol: str,
        iteration: int,
        *,
        config_version: str,
        tickers: dict[str, float],
    ) -> tuple[dict[str, Any], bool, str | None]:
        """Entscheidungs-Pipeline für ein Symbol.

        Returns:
            ``(result_entry, consumed_budget, last_error)``; ``last_error``
            ist nicht ``None``, wenn die Iteration einen Fehler aufwies
            (F1: Ticker-Fehler zählen als Iterationsfehler).
        """
        state = self._state_store.status()
        session_id = state.session_id

        # (1) Ticker (synchroner Provider => off-thread)
        try:
            ticker_data = await asyncio.to_thread(self._market_data.get_ticker, symbol)
        except Exception as exc:  # noqa: BLE001 — F1: Ticker-Fehler müssen die Iteration überstehen
            logger.warning("SHADOW_TICKER_ERROR symbol=%s: %s", symbol, exc)
            entry = self._skip_entry(symbol, "SKIPPED_MARKET_DATA")
            self._audit_iteration(
                iteration=iteration,
                symbol=symbol,
                snapshot_id=None,
                run_id=None,
                decision="SKIPPED_MARKET_DATA",
                risk_result=None,
                execution_result=None,
                session_id=session_id,
            )
            return entry, False, f"TICKER_ERROR: {symbol}: {exc}"

        last = ticker_data.get("last") if isinstance(ticker_data, dict) else None
        if isinstance(last, bool) or not isinstance(last, (int, float)) or last <= 0:
            entry = self._skip_entry(symbol, "SKIPPED_MARKET_DATA")
            self._audit_iteration(
                iteration=iteration,
                symbol=symbol,
                snapshot_id=None,
                run_id=None,
                decision="SKIPPED_MARKET_DATA",
                risk_result=None,
                execution_result=None,
                session_id=session_id,
            )
            return entry, False, None
        last = float(last)
        tickers[symbol] = last

        run_id_holder: list[str | None] = [None]
        try:
            return await self._decide_symbol(
                symbol,
                iteration,
                last=last,
                ticker_data=ticker_data,
                session_id=session_id,
                run_id_holder=run_id_holder,
                config_version=config_version,
            )
        except Exception as exc:
            logger.exception("SHADOW_SYMBOL_ERROR symbol=%s", symbol)
            if run_id_holder[0] is not None:
                run = self._runs.get(run_id_holder[0])
                if run is not None and run.state not in (
                    RunState.COMPLETE,
                    RunState.FAILED,
                ):
                    self._runs.fail(run_id_holder[0], f"{type(exc).__name__}: {exc}")
            entry = {
                "symbol": symbol,
                "snapshot_id": None,
                "run_id": run_id_holder[0],
                "decision": "ERROR",
                "risk_result": None,
                "execution_result": None,
            }
            self._audit_iteration(
                iteration=iteration,
                symbol=symbol,
                snapshot_id=None,
                run_id=run_id_holder[0],
                decision="ERROR",
                risk_result=None,
                execution_result=None,
                session_id=session_id,
            )
            return entry, False, f"{symbol}: {type(exc).__name__}: {exc}"

    async def _decide_symbol(
        self,
        symbol: str,
        iteration: int,
        *,
        last: float,
        ticker_data: dict[str, Any],
        session_id: str,
        run_id_holder: list[str | None],
        config_version: str,
    ) -> tuple[dict[str, Any], bool, str | None]:
        """Snapshot -> Run -> Analyse -> Aggregation -> Risiko -> Execution."""
        settings = self._settings

        # (2) Deterministischer Snapshot + Run (ST.14)
        now = self._clock()
        data: dict[str, Any] = {**ticker_data, "ticker": last}
        snapshot_id = _snapshot_id(symbol, now, data)
        snapshot = MarketSnapshot(id=snapshot_id, symbol=symbol, timestamp=now, data=data)
        self._snapshot_store.add(snapshot)
        run_id = f"shadow-run-{snapshot.id}"
        decision_id = f"shadow-dec-{snapshot.id}-{symbol}"
        run_id_holder[0] = run_id

        self._runs.create(snapshot.id, run_id=run_id)
        self._runs.transition(run_id, RunState.DATA_READY, actor="shadow-loop")
        self._runs.transition(run_id, RunState.ANALYSIS_RUNNING, actor="shadow-loop")

        # (3) Qualifizierte Agenten (Z4) und Analyse
        agents = sorted(
            (
                agent
                for agent in self._agent_source.list_all()
                if agent.status in ("ACTIVE", "CHAMPION")
            ),
            key=lambda agent: agent.id,
        )
        signals: list[AgentSignal] = []
        for agent in agents:
            analysis = await self._agent_runtime.analyze(agent, snapshot, run_id=run_id)
            signals.append(analysis.signal)

        # MVP: kein separates Adversarial-Review-Modul; der State-
        # maschinen-Schritt bleibt als dokumentierter Platzhalter erhalten.
        self._runs.transition(run_id, RunState.ADVERSARIAL_REVIEW, actor="shadow-loop")

        # (4) Deterministische Aggregation (ST.6)
        aggregation = aggregate_signals(
            signals, symbol, min_confidence=settings.shadow_min_confidence
        )
        agent_ids = list(aggregation.agent_ids)
        self._runs.transition(run_id, RunState.CONSENSUS, actor="shadow-loop")

        if aggregation.direction == "NO_TRADE":
            reason = no_trade_reason(aggregation) or "NO_TRADE"
            self._runs.complete(run_id, RunOutcome.NO_TRADE, reason)
            record = ShadowTradingRecord(
                symbol=symbol,
                side="NONE",
                direction="NO_TRADE",
                status=ShadowTradingStatus.NO_TRADE,
                decision_id=decision_id,
                run_id=run_id,
                snapshot_id=snapshot_id,
                risk_reason=reason,
                agent_ids=agent_ids,
                quantity=0.0,
                entry_price=0.0,
                mark_price=last,
                config_version=config_version,
            )
            self._state_store.add_record(record)
            self._audit_iteration(
                iteration=iteration,
                symbol=symbol,
                snapshot_id=snapshot_id,
                run_id=run_id,
                decision="NO_TRADE",
                risk_result=reason,
                execution_result=None,
                session_id=session_id,
            )
            self._audit_decision(record, iteration, execution_result=None)
            return (
                self._entry(symbol, snapshot_id, run_id, "NO_TRADE", reason, None),
                False,
                None,
            )

        # (5) LONG/SHORT: deterministische Risk-Engine (ST.6)
        self._runs.transition(run_id, RunState.RISK_REVIEW, actor="shadow-loop")
        side = "BUY" if aggregation.direction == "LONG" else "SELL"
        policy = self._risk_engine.policy
        portfolio_view = self._sizing_view()
        preliminary = RiskDecision(
            approved=True,
            reason="PRELIMINARY",
            risk_fraction=float(policy.get("max_risk_per_trade", 0.0)),
            max_position_size=float("inf"),
        )
        preliminary_proposal = None
        try:
            preliminary_proposal = build_trade_proposal(
                aggregation,
                snapshot,
                portfolio_view,
                preliminary,
                stop_loss_fraction=settings.shadow_stop_loss_fraction,
                min_risk_reward=settings.shadow_min_risk_reward,
            )
        except ValueError as exc:
            risk_decision = RiskDecision(approved=False, reason=f"PROPOSAL_BUILD_ERROR: {exc}")
        else:
            try:
                risk_decision = self._risk_engine.evaluate(
                    preliminary_proposal, kill_switch=self._kill_switch.is_active()
                )
            except Exception as exc:
                logger.exception("SHADOW_RISK_ENGINE_ERROR symbol=%s", symbol)
                risk_decision = RiskDecision(
                    approved=False, reason=f"RISK_ENGINE_ERROR: {type(exc).__name__}"
                )

        if not risk_decision.approved:
            self._runs.complete(run_id, RunOutcome.WAIT, risk_decision.reason)
            record = ShadowTradingRecord(
                symbol=symbol,
                side=side,
                direction=aggregation.direction,
                status=ShadowTradingStatus.REJECTED,
                decision_id=decision_id,
                run_id=run_id,
                snapshot_id=snapshot_id,
                risk_reason=risk_decision.reason,
                agent_ids=agent_ids,
                quantity=(
                    preliminary_proposal.requested_quantity
                    if preliminary_proposal is not None
                    else 0.0
                ),
                entry_price=(
                    preliminary_proposal.entry_price if preliminary_proposal is not None else 0.0
                ),
                mark_price=last,
                config_version=config_version,
            )
            self._state_store.add_record(record)
            self._audit_iteration(
                iteration=iteration,
                symbol=symbol,
                snapshot_id=snapshot_id,
                run_id=run_id,
                decision="REJECTED",
                risk_result=risk_decision.reason,
                execution_result=None,
                session_id=session_id,
            )
            self._audit_decision(record, iteration, execution_result=None)
            return (
                self._entry(symbol, snapshot_id, run_id, "REJECTED", risk_decision.reason, None),
                True,
                None,
            )

        # (6) Freigegeben: finales Proposal + Paper-Execution (ST.7, ST.3)
        final_proposal = build_trade_proposal(
            aggregation,
            snapshot,
            portfolio_view,
            risk_decision,
            stop_loss_fraction=settings.shadow_stop_loss_fraction,
            min_risk_reward=settings.shadow_min_risk_reward,
        )
        self._runs.transition(run_id, RunState.DECISION, actor="shadow-loop")
        exec_result = await asyncio.to_thread(self._backend.execute, final_proposal)

        if exec_result.status == "FILLED":
            trade_id = exec_result.trade_id
            quantity = exec_result.quantity
            filled_price = exec_result.filled_price
            if trade_id is not None and quantity is not None and filled_price is not None:
                outcome = RunOutcome.LONG if side == "BUY" else RunOutcome.SHORT
                self._runs.complete(run_id, outcome, "FILLED")
                fees = self._lookup_fees(trade_id, quantity, filled_price)
                record = ShadowTradingRecord(
                    symbol=symbol,
                    side=side,
                    direction=aggregation.direction,
                    status=ShadowTradingStatus.FILLED,
                    decision_id=decision_id,
                    run_id=run_id,
                    snapshot_id=snapshot_id,
                    trade_id=trade_id,
                    risk_reason=risk_decision.reason,
                    agent_ids=agent_ids,
                    quantity=quantity,
                    entry_price=filled_price,
                    mark_price=last,
                    slippage=0.0,
                    commission=fees,
                    commission_rate=self._backend.stack.paper_exchange.fee_rate,
                    pnl_estimate=0.0,
                    config_version=config_version,
                )
                self._state_store.add_record(record)
                self._audit_iteration(
                    iteration=iteration,
                    symbol=symbol,
                    snapshot_id=snapshot_id,
                    run_id=run_id,
                    decision="TRADE",
                    risk_result="APPROVED",
                    execution_result="FILLED",
                    session_id=session_id,
                )
                self._audit_decision(record, iteration, execution_result="FILLED")
                self._runs.audit(
                    "SHADOW_FILL",
                    "paper_trade",
                    trade_id,
                    details={
                        "iteration": iteration,
                        "symbol": symbol,
                        "decision_id": decision_id,
                        "run_id": run_id,
                        "quantity": quantity,
                        "filled_price": filled_price,
                    },
                )
                return (
                    self._entry(symbol, snapshot_id, run_id, "TRADE", "APPROVED", "FILLED"),
                    True,
                    None,
                )
            # Fail-closed (Spec ST.3): ein FILLED ohne vollständige Fill-Daten ist
            # ein Protokollverstoß des Backends — auf ERROR herabstufen, damit der
            # gemeinsame Pfad unten protokolliert; niemals Fill-Daten erfinden.
            exec_result = exec_result.model_copy(
                update={"status": "ERROR", "reason": "INCOMPLETE_FILL_DATA"}
            )

        reason = exec_result.reason or "REJECTED"
        if exec_result.status == "ERROR" and not reason.startswith(
            ("EXECUTION_ERROR:", "UNEXPECTED_ADAPTER_STATUS:")
        ):
            reason = f"EXECUTION_ERROR: {reason}"
        self._runs.complete(run_id, RunOutcome.WAIT, reason)
        record = ShadowTradingRecord(
            symbol=symbol,
            side=side,
            direction=aggregation.direction,
            status=ShadowTradingStatus.REJECTED,
            decision_id=decision_id,
            run_id=run_id,
            snapshot_id=snapshot_id,
            risk_reason=reason,
            agent_ids=agent_ids,
            quantity=final_proposal.requested_quantity,
            entry_price=final_proposal.entry_price,
            mark_price=last,
            config_version=config_version,
        )
        self._state_store.add_record(record)
        execution_result = f"{exec_result.status}: {reason}"
        self._audit_iteration(
            iteration=iteration,
            symbol=symbol,
            snapshot_id=snapshot_id,
            run_id=run_id,
            decision="TRADE",
            risk_result="APPROVED",
            execution_result=execution_result,
            session_id=session_id,
        )
        self._audit_decision(record, iteration, execution_result=execution_result)
        return (
            self._entry(symbol, snapshot_id, run_id, "TRADE", "APPROVED", execution_result),
            True,
            None,
        )

    # ------------------------------------------------------------------
    # Mark-to-Market (Spec ST.10)
    # ------------------------------------------------------------------

    async def _mark_to_market(self, tickers: dict[str, float]) -> None:
        """Einmalig pro Iteration: Stop-/Target-Checks + Mark-to-Market.

        Siehe Modul-Docstring zur dokumentierten Abweichung (``all()`` statt
        ``get_open_positions()`` für den realisierten PnL).
        """
        open_positions = list(self._position_manager.get_open_positions())

        prices: dict[str, float] = dict(tickers)
        for pos in open_positions:
            if pos.symbol in prices:
                continue
            try:
                data = await asyncio.to_thread(self._market_data.get_ticker, pos.symbol)
                candidate = data.get("last") if isinstance(data, dict) else None
                if (
                    not isinstance(candidate, bool)
                    and isinstance(candidate, (int, float))
                    and candidate > 0
                ):
                    prices[pos.symbol] = float(candidate)
            except Exception:  # noqa: BLE001 — M2M: fehlendes Ticker-Update darf nur die Position überspringen
                logger.warning("SHADOW_M2M_TICKER_MISSING symbol=%s", pos.symbol)

        for pos in open_positions:
            price = prices.get(pos.symbol)
            if price is None:
                continue
            try:
                trigger = await asyncio.to_thread(
                    self._position_manager.check_stop_loss_target, pos.id, price
                )
            except Exception:
                logger.exception("SHADOW_M2M_STOP_CHECK_FAILED position=%s", pos.id)
                continue
            if trigger is not None:
                logger.info(
                    "SHADOW_POSITION_CLOSED position=%s symbol=%s reason=%s price=%s",
                    pos.id,
                    pos.symbol,
                    trigger["reason"],
                    trigger["price"],
                )
                self._runs.audit(
                    "SHADOW_POSITION_CLOSED",
                    "position",
                    pos.id,
                    details={
                        "symbol": pos.symbol,
                        "reason": trigger["reason"],
                        "price": trigger["price"],
                    },
                )

        all_positions = self._position_store.all()
        normalized = [
            pos.model_copy(update={"unrealized_pnl": 0.0}) if pos.status.value != "OPEN" else pos
            for pos in all_positions
        ]
        fresh = await asyncio.to_thread(self._portfolio_tracker.update, normalized)
        open_now = self._position_manager.get_open_positions()
        shadow_state = fresh.model_copy(
            update={
                "run_id": "shadow",
                "positions": {pos.symbol: pos.quantity for pos in open_now},
                "symbols": [pos.symbol for pos in open_now],
            }
        )
        self._state_store.add_portfolio_state(shadow_state)
        self._state_store.update_state(
            {
                "current_equity": fresh.current_equity,
                "open_positions": len(open_now),
            }
        )

    # ------------------------------------------------------------------
    # Hilfsfunktionen
    # ------------------------------------------------------------------

    def _sizing_view(self) -> PortfolioState:
        """Read-only Portfolio-Ansicht für die Sizing-Formel (ST.10)."""
        state = self._state_store.status()
        open_positions = self._position_manager.get_open_positions()
        return PortfolioState(
            run_id="shadow",
            start_equity=state.start_equity,
            current_equity=state.current_equity,
            positions={pos.symbol: pos.quantity for pos in open_positions},
            symbols=[pos.symbol for pos in open_positions],
        )

    def _lookup_fees(self, trade_id: str, quantity: float, price: float) -> float:
        """Commission aus dem Paper-Trade; Fallback ``|q*p| * fee_rate``."""
        fee_rate = self._backend.stack.paper_exchange.fee_rate
        try:
            trade = self._backend.stack.trade_store.get(trade_id)
            if trade is not None:
                return float(trade.fees)
        except Exception:
            logger.exception("SHADOW_FEE_LOOKUP_FAILED trade_id=%s", trade_id)
        return abs(quantity * price) * fee_rate

    def _self_stop(self, action: str, session_id: str, details: dict[str, Any]) -> None:
        """Self-Stop der Loop (ST.4/ST.11, S2, F6)."""
        self._state_store.update_state(
            {
                "status": "STOPPED",
                "stopped_at": self._clock(),
                "restart_required": True,
            }
        )
        self._state_store.save()
        self._runs.audit(action, "shadow_session", session_id, new_state="STOPPED", details=details)
        logger.warning("SHADOW_LOOP_SELF_STOP %s session=%s", action, session_id)

    def _audit_iteration(
        self,
        *,
        iteration: int,
        symbol: str,
        snapshot_id: str | None,
        run_id: str | None,
        decision: str,
        risk_result: str | None,
        execution_result: str | None,
        session_id: str,
    ) -> None:
        """ST.13: pro Symbol pro Iteration ein SHADOW_ITERATION-Eintrag."""
        if run_id is not None:
            entity_type, entity_id = "trading_run", run_id
        else:
            entity_type, entity_id = "shadow_session", session_id
        self._runs.audit(
            "SHADOW_ITERATION",
            entity_type,
            entity_id,
            details={
                "iteration": iteration,
                "symbol": symbol,
                "snapshot_id": snapshot_id,
                "decision": decision,
                "risk_result": risk_result,
                "execution_result": execution_result,
            },
        )

    def _audit_decision(
        self, record: ShadowTradingRecord, iteration: int, *, execution_result: str | None
    ) -> None:
        """ST.13: pro Entscheidungs-Record ein SHADOW_DECISION-Eintrag."""
        self._runs.audit(
            "SHADOW_DECISION",
            "shadow_decision",
            record.decision_id,
            details={
                "iteration": iteration,
                "symbol": record.symbol,
                "snapshot_id": record.snapshot_id,
                "run_id": record.run_id,
                "status": record.status.value,
                "risk_reason": record.risk_reason,
                "trade_id": record.trade_id,
                "execution_result": execution_result,
            },
        )

    @staticmethod
    def _entry(
        symbol: str,
        snapshot_id: str,
        run_id: str,
        decision: str,
        risk_result: str,
        execution_result: str | None,
    ) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "snapshot_id": snapshot_id,
            "run_id": run_id,
            "decision": decision,
            "risk_result": risk_result,
            "execution_result": execution_result,
        }

    @staticmethod
    def _skip_entry(symbol: str, decision: str) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "snapshot_id": None,
            "run_id": None,
            "decision": decision,
            "risk_result": None,
            "execution_result": None,
        }


# ---------------------------------------------------------------------------
# Phase-2-Evaluation (Spec ST.12/O4) — reines Modul-Level-Helper
# ---------------------------------------------------------------------------


def compute_shadow_evaluation(
    records: list[ShadowTradingRecord],
    portfolio_history: list[PortfolioState],
) -> ShadowEvaluationSummary:
    """Berechnet Phase-2-Metriken über akkumulierte Shadow-Records (ST.12).

    Dokumentiertes MVP-Mapping auf ``OutcomeRecord``:

    - ``direction_actual`` aus dem Vorzeichen von ``pnl_estimate``
      (``>0`` LONG, ``<0`` SHORT, ``==0`` NO_TRADE — offene FILLED-Records
      sind damit erst nach Settle "correct/incorrect" bewertbar),
    - ``confidence_predicted`` konstant 0.5 (Shadow-Records speichern kein
      per-Trade-Confidence; Brier ist damit für jede nicht-leere Menge 0.25),
    - ``realized_pnl`` = ``pnl_estimate``,
    - ``max_drawdown`` = Maximum über ``portfolio_history``.

    Deterministisch: gleiche Records/History => identische Metriken.
    """
    outcomes: list[OutcomeRecord] = []
    for record in records:
        if record.pnl_estimate > 0:
            actual = "LONG"
        elif record.pnl_estimate < 0:
            actual = "SHORT"
        else:
            actual = "NO_TRADE"
        outcomes.append(
            OutcomeRecord(
                prediction_id=record.id,
                agent_id=record.agent_ids[0] if record.agent_ids else "",
                run_id=record.run_id,
                snapshot_id=record.snapshot_id,
                symbol=record.symbol,
                direction_predicted=record.direction,
                direction_actual=actual,
                confidence_predicted=0.5,
                entry_price=record.entry_price,
                exit_price=record.mark_price,
                realized_pnl=record.pnl_estimate,
            )
        )
    metrics = _compute_classification_metrics(outcomes)
    return ShadowEvaluationSummary(
        brier_score=metrics.brier_score,
        directional_accuracy=metrics.directional_accuracy,
        expectancy=metrics.expectancy,
        max_drawdown=max((state.max_drawdown for state in portfolio_history), default=0.0),
        sample_size=len(records),
    )
