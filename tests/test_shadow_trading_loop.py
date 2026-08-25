"""Tests für den Shadow Trading Loop (WI-ST-05, Spec ST.1–ST.15).

Abgedeckt: vollständige Entscheidungs-Kette (Ticker -> Snapshot -> Run ->
Agenten -> Aggregation -> Risk Engine -> Paper-Execution -> Records ->
Mark-to-Market -> Audit), strikte Isolation von Live-Execution-Endpunkten,
Event-Loop-Blocking-Schutz, Tages-Budget inkl. UTC-Rollover,
NO_TRADE-Semantik, Stop-Loss-Handling in Iterationen, Kill-Switch- und
Flags-Self-Stops, Fehlerschutz bei aufeinanderfolgenden Iterationsfehlern,
State-Größenbegrenzung, Phase-2-Evaluation, Audit-Trail und Determinismus
(ST.14).

Alle Tests sind deterministisch: feste Uhr (``T0``), scriptbare
Ticker-/Agenten-Fakes, echte übrige Kollaboratoren (Paper-Stack, Risk
Engine, Kill Switch, State Store).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import time
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from trading_harness.config import Settings
from trading_harness.models import (
    AgentAnalysisResult,
    AgentGenome,
    AgentSignal,
    AgentStatus,
    PaperPosition,
    PaperPositionStatus,
    PortfolioState,
    RunOutcome,
    RunState,
    ShadowLoopStatus,
    ShadowTradingRecord,
    ShadowTradingStatus,
)
from trading_harness.services.agent_genome_store import AgentGenomeStore
from trading_harness.services.crypto_exchange_adapter import (
    BinanceExchangeAdapter,
    BitgetExchangeAdapter,
    BybitExchangeAdapter,
    CoinbaseExchangeAdapter,
)
from trading_harness.services.kill_switch import KillSwitch
from trading_harness.services.live_execution_service import LiveExecutionService
from trading_harness.services.orchestrator import TradingRunService
from trading_harness.services.paper_execution_stack import build_paper_execution_stack
from trading_harness.services.risk_engine import RiskEngine
from trading_harness.services.shadow_execution_backend import (
    ShadowExecutionBackend,
    ShadowExecutionResult,
)
from trading_harness.services.shadow_trading_loop import (
    ShadowTradingLoop,
    ShadowTradingStartError,
    compute_shadow_evaluation,
)
from trading_harness.services.shadow_trading_state import (
    MAX_CONSECUTIVE_ERRORS,
    ShadowTradingStateStore,
)
from trading_harness.services.snapshot_store import SnapshotStore

pytestmark = pytest.mark.asyncio

T0 = datetime(2026, 8, 20, 10, 0, 0, tzinfo=UTC)
AGENT_ID = "agent-tech-fixed"

# Spiegelt config/risk-policy.yaml (erlaubt BTCUSDT + ETHUSDT).
POLICY: dict[str, object] = {
    "allowed_symbols": ["BTCUSDT", "ETHUSDT"],
    "max_risk_per_trade": 0.005,
    "max_daily_loss": 0.02,
    "max_portfolio_risk": 0.04,
    "max_leverage": 2.0,
    "max_positions": 5,
    "minimum_risk_reward": 1.8,
    "max_slippage_bps": 20,
}


class MutableClock:
    """Deterministische, manuell vorlaufende Uhr für den Loop."""

    def __init__(self, start: datetime) -> None:
        self._now = start

    def __call__(self) -> datetime:
        return self._now

    def advance(self, *, days: int = 0, seconds: float = 0.0) -> None:
        self._now = self._now + timedelta(days=days, seconds=seconds)


class ScriptedTickerProvider:
    """Synchroner Ticker-Stub mit optionaler Verzögerung und Fehlerinjection."""

    def __init__(
        self,
        prices: dict[str, float],
        *,
        delay_seconds: float = 0.0,
        fail: Exception | None = None,
    ) -> None:
        self._prices = dict(prices)
        self.delay_seconds = delay_seconds
        self.fail = fail

    def set_price(self, symbol: str, price: float) -> None:
        self._prices[symbol] = price

    def get_ticker(self, symbol: str) -> dict[str, float]:
        if self.fail is not None:
            raise self.fail
        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)
        return {"last": self._prices[symbol]}


class FakeAgentRuntime:
    """Deterministischer Agenten-Stub: Script-Einträge pro analyze()-Aufruf.

    Das Script wird nach dem letzten Eintrag gehalten (Repeat), damit
    mehr Iterationen als Skript-Einträge laufen können.
    """

    def __init__(self, script: list[tuple[str, float]]) -> None:
        self._script = list(script)
        self._index = 0
        self.calls: list[tuple[str, str]] = []

    def reset_script(self, script: list[tuple[str, float]]) -> None:
        self._script = list(script)
        self._index = 0

    def _next(self) -> tuple[str, float]:
        entry = self._script[min(self._index, len(self._script) - 1)]
        self._index += 1
        return entry

    async def analyze(self, agent, snapshot, run_id=None):
        direction, confidence = self._next()
        self.calls.append((agent.id, snapshot.id))
        run_id = run_id or "run-fake"
        signal = AgentSignal(
            run_id=run_id,
            agent_id=agent.id,
            snapshot_id=snapshot.id,
            category=agent.category,
            direction=direction,
            confidence=confidence,
            reasoning="scripted fake signal",
        )
        return AgentAnalysisResult(
            run_id=run_id,
            agent_id=agent.id,
            signal=signal,
            prompt_version="test",
            model_profile="local-main",
        )


def build_env(root: Path) -> SimpleNamespace:
    """Baue eine vollständige, isolierte Loop-Umgebung unter ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        _env_file=None,
        shadow_trading_enabled=True,
        shadow_trading_symbols=["BTCUSDT"],
        shadow_loop_interval_seconds=900,
    )
    policy = dict(POLICY)
    stack = build_paper_execution_stack(db=None)
    backend = ShadowExecutionBackend(stack)
    provider = ScriptedTickerProvider({"BTCUSDT": 100.0, "ETHUSDT": 2000.0})
    runtime = FakeAgentRuntime([("LONG", 0.9)])
    agents = AgentGenomeStore()
    agents.add(AgentGenome(id=AGENT_ID, category="technical", status=AgentStatus.ACTIVE))
    state_store = ShadowTradingStateStore(root / "shadow_trading_state.json")
    kill = KillSwitch(db_path=str(root / "kill_switch.json"))
    clock = MutableClock(T0)
    loop = ShadowTradingLoop(
        market_data=provider,
        backend=backend,
        agent_runtime=runtime,
        trading_run_service=TradingRunService(),
        snapshot_store=SnapshotStore(),
        risk_engine=RiskEngine(policy),
        kill_switch=kill,
        agent_source=agents,
        settings=settings,
        state_store=state_store,
        clock=clock,
    )
    return SimpleNamespace(
        root=root,
        settings=settings,
        policy=policy,
        stack=stack,
        backend=backend,
        provider=provider,
        runtime=runtime,
        agents=agents,
        state_store=state_store,
        kill=kill,
        clock=clock,
        loop=loop,
        runs=loop._runs,
        snapshots=loop._snapshot_store,
    )


@pytest.fixture
def env(tmp_path) -> SimpleNamespace:
    return build_env(tmp_path / "env")

async def test_shadow_loop_iteration_full_chain(env) -> None:
    """ST.1/ST.3/ST.7/ST.13/ST.14: vollständige Kette Ticker -> ... -> Fill."""
    result = await env.loop.run_once()

    assert result["iteration"] == 1
    assert len(result["symbols"]) == 1
    entry = result["symbols"][0]
    sid = result["snapshot_id"]
    run_id = result["run_id"]
    assert re.fullmatch(r"snap-[0-9a-f]{16}", sid) is not None
    assert run_id == f"shadow-run-{sid}"
    assert entry == {
        "symbol": "BTCUSDT",
        "snapshot_id": sid,
        "run_id": run_id,
        "decision": "TRADE",
        "risk_result": "APPROVED",
        "execution_result": "FILLED",
    }
    assert result["decision"] == "TRADE"
    assert result["equity"] == 100000.0
    assert result["duration_ms"] >= 0

    snapshot = env.snapshots.get(sid)
    assert snapshot is not None
    assert snapshot.data == {"last": 100.0, "ticker": 100.0}
    assert snapshot.content_hash

    run = env.runs.get(run_id)
    assert run is not None
    assert run.state is RunState.COMPLETE
    assert run.outcome is RunOutcome.LONG
    assert run.outcome_reason == "FILLED"

    fills = env.stack.trade_store.all()
    assert len(fills) == 1
    trade = fills[0]
    decision_id = f"shadow-dec-{sid}-BTCUSDT"
    assert trade.trade_id == decision_id
    assert trade.status.value == "FILLED"
    assert trade.actual_quantity == 200.0  # requested 250 * fill_rate 0.8
    assert trade.actual_price == 100.0
    assert trade.fees == pytest.approx(20.0)

    positions = env.stack.position_store.all()
    assert len(positions) == 1
    pos = positions[0]
    assert pos.trade_id == decision_id
    assert pos.quantity == 200.0
    assert pos.entry_price == 100.0
    assert pos.stop_price == pytest.approx(95.0)
    assert pos.target_price == pytest.approx(105.0)

    records = env.state_store.records()
    assert len(records) == 1
    record = records[0]
    assert record.decision_id == decision_id
    assert record.status is ShadowTradingStatus.FILLED
    assert record.side == "BUY"
    assert record.direction == "LONG"
    assert record.trade_id == decision_id
    assert record.risk_reason == "APPROVED"
    assert record.quantity == 200.0
    assert record.entry_price == 100.0
    assert record.mark_price == 100.0
    assert record.commission == pytest.approx(20.0)
    assert record.commission_rate == pytest.approx(0.001)
    assert record.agent_ids == [AGENT_ID]
    assert len(record.config_version) == 64

    state = env.loop.status()
    assert state.iteration_count == 1
    assert state.decisions_today == 1
    assert state.budget_date == "2026-08-20"
    assert state.current_equity == 100000.0
    assert state.open_positions == 1
    assert state.start_equity == 100000.0
    assert state.symbols == ["BTCUSDT"]

    history = env.state_store.portfolio_history()
    assert len(history) == 1
    assert history[0].run_id == "shadow"
    assert history[0].current_equity == 100000.0
    assert history[0].positions == {"BTCUSDT": 200.0}

    # ST.13: Run-Lifecycle und SHADOW_ITERATION haengen an der run_id ...
    run_audit = env.runs.get_audit_log(run_id)
    assert [a.action for a in run_audit] == [
        "create",
        "transition",
        "transition",
        "transition",
        "transition",
        "transition",
        "transition",
        "complete",
        "SHADOW_ITERATION",
    ]
    # ... Decision/Fill dagegen an decision_id/trade_id -> globaler Log.
    all_audit = env.runs.get_audit_log()
    decision_entry = next(a for a in all_audit if a.action == "SHADOW_DECISION")
    assert decision_entry.entity_id == decision_id
    assert decision_entry.details["status"] == "FILLED"
    fill_entry = next(a for a in all_audit if a.action == "SHADOW_FILL")
    assert fill_entry.entity_type == "paper_trade"
    assert fill_entry.details["quantity"] == 200.0


async def test_loop_never_reaches_trade_endpoints(env) -> None:
    """Epic-B3/ST.3: FILLED-Trades laufen nur über den Paper-Stack."""
    # Kein autospec: start()-Ergebnisse sind auf diesem Python-Build nackte
    # Funktionen ohne .stop(); gestoppt wird ueber das Patch-Objekt selbst.
    patchers = [
        patch.object(klass, "submit_order")
        for klass in (
            LiveExecutionService,
            BinanceExchangeAdapter,
            BitgetExchangeAdapter,
            BybitExchangeAdapter,
            CoinbaseExchangeAdapter,
        )
    ]
    patchers += [
        patch.object(LiveExecutionService, name)
        for name in ("get_order_status", "cancel_order")
    ]

    try:
        spies = [p.start() for p in patchers]
        result = await env.loop.run_once()
    finally:
        for p in patchers:
            p.stop()

    assert result["decision"] == "TRADE"
    assert result["symbols"][0]["execution_result"] == "FILLED"
    for spy in spies:
        spy.assert_not_called()
    assert len(env.stack.trade_store.all()) == 1


async def test_slow_ticker_does_not_block_event_loop(env) -> None:
    """ST.1: synchroner (vernetzter) Ticker-Provider läuft off-thread."""
    env.settings.shadow_trading_symbols = ["BTCUSDT", "ETHUSDT"]
    env.settings.shadow_loop_interval_seconds = 0
    env.provider.delay_seconds = 0.5
    env.state_store.update_state({"status": "RUNNING"})

    ticks: list[float] = []

    async def heartbeat() -> None:
        while True:
            ticks.append(time.monotonic())
            await asyncio.sleep(0.05)

    hb = asyncio.create_task(heartbeat())
    try:
        await asyncio.wait_for(env.loop.start(), timeout=5)
        assert env.loop.status().status is ShadowLoopStatus.RUNNING
        first = await asyncio.wait_for(
            env.loop.run_once(), timeout=15
        )  # erste Iteration: 2 x 0.5 s off-thread
        assert first["iteration"] == 1
        assert len(ticks) >= 3
        assert max(b - a for a, b in pairwise(ticks)) < 0.2
        assert env.loop.status().iteration_count >= 1
    finally:
        with contextlib.suppress(asyncio.CancelledError):
            hb.cancel()
            await asyncio.gather(hb, return_exceptions=True)
        with contextlib.suppress(asyncio.CancelledError):
            if env.loop._task is not None:
                env.loop._task.cancel()
                await asyncio.gather(env.loop._task, return_exceptions=True)


async def test_start_rejected_when_disabled(env) -> None:
    """ST.2: Shadow Trading startet nicht ohne Flag."""
    env.settings.shadow_trading_enabled = False
    with pytest.raises(ShadowTradingStartError) as exc:
        await env.loop.start()
    assert exc.value.code == "SHADOW_TRADING_DISABLED"
    assert env.loop.status().status is ShadowLoopStatus.STOPPED
    assert env.loop._task is None


async def test_start_rejected_when_live_enabled(env) -> None:
    """ST.2/S1: Live-Execution hat absolute Vorrang-Sperre (Guard-Reihenfolge)."""
    env.settings.live_execution_enabled = True
    with pytest.raises(ShadowTradingStartError) as exc:
        await env.loop.start()
    assert exc.value.code == "LIVE_EXECUTION_MUST_BE_DISABLED"
    assert env.loop.status().status is ShadowLoopStatus.STOPPED


async def test_start_rejected_when_no_symbols(env) -> None:
    """ST.2: ohne konfigurierte Symbole kein Start."""
    env.settings.shadow_trading_symbols = []
    with pytest.raises(ShadowTradingStartError) as exc:
        await env.loop.start()
    assert exc.value.code == "NO_SYMBOLS_CONFIGURED"


async def test_start_rejected_when_already_running(env) -> None:
    """ST.2: zweiter start() während RUNNING und aktivem Task wird abgewiesen."""
    env.state_store.update_state({"status": "RUNNING"})
    async def _idle() -> None:
        await asyncio.sleep(3600)

    task: asyncio.Task[None] = asyncio.create_task(_idle())
    env.loop._task = task
    try:
        with pytest.raises(ShadowTradingStartError) as exc:
            await env.loop.start()
        assert exc.value.code == "ALREADY_RUNNING"
        assert not task.done()
    finally:
        with contextlib.suppress(asyncio.CancelledError):
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_concurrent_start_single_winner(env) -> None:
    """ST.2: zwei parallele start()-Aufrufe (ein Event Loop) -> genau ein Winner.

    Der Guard ist auf einem Event Loop atomar (kein await zwischen Check und
    Set), daher muss der zweite Aufruf ALREADY_RUNNING erhalten — es darf
    keinen Double-Start geben (ein Task, ein SHADOW_LOOP_STARTED-Audit).
    """
    outcomes = await asyncio.gather(
        env.loop.start(),
        env.loop.start(),
        return_exceptions=True,
    )
    try:
        errors = [o for o in outcomes if isinstance(o, ShadowTradingStartError)]
        assert len(errors) == 1
        assert errors[0].code == "ALREADY_RUNNING"
        assert sum(1 for o in outcomes if not isinstance(o, BaseException)) == 1
        assert env.loop.status().status is ShadowLoopStatus.RUNNING
        assert env.loop._task is not None
        assert not env.loop._task.done()
        started = [
            a for a in env.runs.get_audit_log() if a.action == "SHADOW_LOOP_STARTED"
        ]
        assert len(started) == 1
    finally:
        with contextlib.suppress(asyncio.CancelledError):
            if env.loop._task is not None:
                env.loop._task.cancel()
                await asyncio.gather(env.loop._task, return_exceptions=True)


async def test_start_cancels_stale_task(env) -> None:
    """ST.2: Stale Task nach Self-Stop wird vor dem Neustart kontrolliert beendet."""
    env.state_store.update_state({"status": "STOPPED", "restart_required": True})
    started = asyncio.Event()

    async def _stale() -> None:
        started.set()
        await asyncio.sleep(3600)

    stale: asyncio.Task[None] = asyncio.create_task(_stale())
    env.loop._task = stale
    try:
        await started.wait()
        await env.loop.start()
    finally:
        with contextlib.suppress(asyncio.CancelledError):
            if env.loop._task is not None:
                env.loop._task.cancel()
                await asyncio.gather(env.loop._task, return_exceptions=True)
    assert stale.cancelled() or stale.done()
    assert env.loop.status().status is ShadowLoopStatus.RUNNING
    assert env.loop._task is not stale


async def test_stop_marks_restart_required(env) -> None:
    """ST.2: stop() ist idempotent und markiert restart_required."""
    await env.loop.stop("test")
    state = env.loop.status()
    assert state.status is ShadowLoopStatus.STOPPED
    assert state.restart_required is True
    assert state.stopped_at == T0
    await env.loop.stop("test")  # idempotent
    assert env.loop.status().restart_required is True
    audit = env.runs.get_audit_log()
    stopped = [a for a in audit if a.action == "SHADOW_LOOP_STOPPED"]
    assert len(stopped) == 2
    assert stopped[0].details == {"reason": "test"}


async def test_single_symbol_return_has_top_level_keys(env) -> None:
    """ST.1: genau ein Symbol -> Top-Level snapshot_id/run_id/decision."""
    result = await env.loop.run_once()
    for key in ("snapshot_id", "run_id", "decision"):
        assert key in result
    assert result["symbols"][0]["snapshot_id"] == result["snapshot_id"]
    assert result["symbols"][0]["run_id"] == result["run_id"]
    assert result["symbols"][0]["decision"] == result["decision"]


async def test_multi_symbol_return_has_symbols_list(env) -> None:
    """ST.1: mehrere Symbole -> nur symbols-Liste, keine Top-Level-Keys."""
    env.settings.shadow_trading_symbols = ["BTCUSDT", "ETHUSDT"]
    result = await env.loop.run_once()
    assert result["iteration"] == 1
    assert len(result["symbols"]) == 2
    assert [e["symbol"] for e in result["symbols"]] == ["BTCUSDT", "ETHUSDT"]
    for key in ("snapshot_id", "run_id", "decision"):
        assert key not in result
    assert result["equity"] == 100000.0
    assert env.loop.status().decisions_today == 2


async def test_budget_exhaustion_skips_decisions(env) -> None:
    """ST.4: Tages-Budget erschöpft -> SKIPPED_BUDGET, Loop bleibt RUNNING."""
    env.settings.shadow_trading_symbols = ["BTCUSDT", "ETHUSDT"]
    env.settings.shadow_max_decisions_per_day = 2
    env.state_store.update_state({"status": "RUNNING", "budget_date": "2026-08-19"})

    first = await env.loop.run_once()
    assert [e["decision"] for e in first["symbols"]] == ["TRADE", "TRADE"]
    assert env.loop.status().decisions_today == 2
    assert env.loop.status().status is ShadowLoopStatus.RUNNING

    env.clock.advance(seconds=3600)
    second = await env.loop.run_once()
    assert [e["decision"] for e in second["symbols"]] == ["SKIPPED_BUDGET"] * 2
    assert env.loop.status().decisions_today == 2
    assert env.loop.status().status is ShadowLoopStatus.RUNNING
    assert env.loop.status().iteration_count == 2

    exhausted = [
        a
        for a in env.runs.get_audit_log()
        if a.action == "SHADOW_LOOP_STOPPED_BUDGET_EXHAUSTED"
    ]
    assert len(exhausted) == 1
    assert exhausted[0].details["decisions_today"] == 2
    assert exhausted[0].details["max_decisions_per_day"] == 2
    assert exhausted[0].details["iteration"] == 2


async def test_budget_resets_on_day_rollover(env) -> None:
    """ST.4: neues UTC-Tag -> Budget wird zurückgesetzt."""
    env.settings.shadow_trading_symbols = ["BTCUSDT", "ETHUSDT"]
    env.settings.shadow_max_decisions_per_day = 2
    env.state_store.update_state({"status": "RUNNING"})

    await env.loop.run_once()
    assert env.loop.status().budget_date == "2026-08-20"
    assert env.loop.status().decisions_today == 2

    env.clock.advance(seconds=3600)
    await env.loop.run_once()  # beide SKIPPED_BUDGET
    env.clock.advance(days=1)
    third = await env.loop.run_once()
    assert [e["decision"] for e in third["symbols"]] == ["TRADE", "TRADE"]
    assert env.loop.status().budget_date == "2026-08-21"
    assert env.loop.status().decisions_today == 2


async def test_no_trade_does_not_consume_budget(env) -> None:
    """ST.4/ST.6: NO_TRADE ist eine Entscheidung ohne Budget-Konsum."""
    env.runtime.reset_script(
        [("LONG", 0.9), ("NO_TRADE", 0.9), ("NO_TRADE", 0.9)]
    )
    env.settings.shadow_max_decisions_per_day = 2
    first = await env.loop.run_once()
    assert first["decision"] == "TRADE"
    assert env.loop.status().decisions_today == 1

    env.clock.advance(seconds=3600)
    second = await env.loop.run_once()
    assert second["decision"] == "NO_TRADE"
    assert env.loop.status().decisions_today == 1

    # Das Script wiederholt nach dem letzten Eintrag NO_TRADE; fuer die dritte
    # Iteration wird es daher explizit auf LONG zurueckgesetzt.
    env.runtime.reset_script([("LONG", 0.9)])
    env.clock.advance(seconds=3600)
    third = await env.loop.run_once()
    assert third["decision"] == "TRADE"
    assert env.loop.status().decisions_today == 2


async def test_mark_to_market_continues_when_budget_exhausted(env) -> None:
    """ST.10/ST.4: M2M und Stop-Checks laufen auch ohne Budget weiter."""
    env.settings.shadow_max_decisions_per_day = 1
    env.state_store.update_state({"status": "RUNNING"})
    await env.loop.run_once()  # FILLED, offene Position, Budget (1) erschöpft
    assert env.loop.status().decisions_today == 1
    assert len(env.state_store.portfolio_history()) == 1

    env.clock.advance(seconds=3600)
    result = await env.loop.run_once()
    assert result["symbols"][0]["decision"] == "SKIPPED_BUDGET"
    assert len(env.state_store.portfolio_history()) == 2
    assert env.loop.status().status is ShadowLoopStatus.RUNNING


async def test_mark_to_market_skips_out_of_scope_symbol(env, caplog) -> None:
    """ST.10-Härten: Out-of-Scope-Positionen (Symbol außer Konfig-Set) lösen
    keine Ticker-Abfrage aus; es wird nur gewarnt, die Position bleibt offen.
    """
    env.settings.shadow_trading_symbols = ["BTCUSDT"]
    # Offene Position für ein Symbol, das das konfigurierte Set verlassen hat.
    pos = PaperPosition(
        trade_id="shadow-dec-legacy-ETHUSDT",
        run_id="shadow-run-legacy",
        symbol="ETHUSDT",
        side="LONG",
        entry_price=2000.0,
        quantity=10.0,
    )
    env.stack.position_store.add(pos)
    caplog.set_level(
        logging.WARNING, logger="trading_harness.services.shadow_trading_loop"
    )

    with patch.object(env.provider, "get_ticker", wraps=env.provider.get_ticker) as spy:
        await env.loop._mark_to_market({})

    spy.assert_not_called()
    assert "SHADOW_M2M_SYMBOL_OUT_OF_SCOPE symbol=ETHUSDT" in caplog.text
    assert f"pos_id={pos.id}" in caplog.text
    # Kein Crash: die Position bleibt offen und M2M hat den State aktualisiert.
    assert env.stack.position_store.all()[0].status is PaperPositionStatus.OPEN
    assert len(env.state_store.portfolio_history()) == 1
    assert env.loop.status().open_positions == 1


# --- CHUNK3 ---


def _eval_record(direction: str, pnl: float) -> ShadowTradingRecord:
    """Synthetischer FILLED-Record für reine Evaluations-Assertions (ST.12)."""
    return ShadowTradingRecord(
        symbol="BTCUSDT",
        side="BUY",
        direction=direction,
        status=ShadowTradingStatus.FILLED,
        decision_id=f"dec-{direction.lower()}-{pnl}",
        run_id="run-eval",
        snapshot_id="snap-eval",
        risk_reason="APPROVED",
        agent_ids=[AGENT_ID],
        quantity=10.0,
        entry_price=100.0,
        mark_price=100.0,
        pnl_estimate=pnl,
        config_version="c" * 64,
    )


def _record_fingerprint(record: ShadowTradingRecord) -> dict[str, object]:
    """Deterministischer Vergleichs-Fingerabdruck ohne ID/Timestamp (ST.14)."""
    data = record.model_dump()
    data.pop("id", None)
    data.pop("timestamp", None)
    return data


async def test_no_trade_record_semantics(env) -> None:
    """ST.6/E2: NO_TRADE-Records tragen genau einen der zwei Grund-Strings."""
    # Fall A: keine qualifizierenden Agenten -> NO_ACTIVE_CHAMPIONS.
    env.loop._agent_source = AgentGenomeStore()
    first = await env.loop.run_once()
    assert first["decision"] == "NO_TRADE"
    assert first["symbols"][0]["risk_result"] == "NO_ACTIVE_CHAMPIONS"
    records = env.state_store.records()
    assert len(records) == 1
    record = records[0]
    assert record.side == "NONE"
    assert record.direction == "NO_TRADE"
    assert record.status is ShadowTradingStatus.NO_TRADE
    assert record.risk_reason == "NO_ACTIVE_CHAMPIONS"
    assert record.agent_ids == []
    assert record.quantity == 0.0
    assert record.entry_price == 0.0
    assert record.mark_price == pytest.approx(100.0)
    assert env.loop.status().decisions_today == 0

    # Fall B: Signal unter min_confidence (0.4 < 0.6) -> BELOW_MIN_CONFIDENCE.
    env.loop._agent_source = env.agents
    env.runtime.reset_script([("LONG", 0.4)])
    env.clock.advance(seconds=3600)
    second = await env.loop.run_once()
    assert second["decision"] == "NO_TRADE"
    assert second["symbols"][0]["risk_result"] == "BELOW_MIN_CONFIDENCE"
    records = env.state_store.records()
    assert len(records) == 2
    assert records[1].risk_reason == "BELOW_MIN_CONFIDENCE"
    assert records[1].agent_ids == [AGENT_ID]
    state = env.loop.status()
    assert state.decisions_today == 0
    assert state.iteration_count == 2


async def test_risk_rejected_symbol_record(env) -> None:
    """ST.6: Risk-Engine-Ablehnung erzeugt REJECTED-Record und WAIT-Run."""
    env.settings.shadow_trading_symbols = ["DOGEUSDT"]
    env.provider.set_price("DOGEUSDT", 100.0)
    result = await env.loop.run_once()

    assert result["decision"] == "REJECTED"
    assert result["symbols"][0]["risk_result"] == "SYMBOL_NOT_ALLOWED"
    assert result["symbols"][0]["execution_result"] is None

    record = env.state_store.records()[0]
    assert record.status is ShadowTradingStatus.REJECTED
    assert record.side == "BUY"
    assert record.direction == "LONG"
    assert record.risk_reason == "SYMBOL_NOT_ALLOWED"
    # Preliminary-Sizing: (100000 * 0.005) / (100 * 0.02) = 250.
    assert record.quantity == pytest.approx(250.0)
    assert record.entry_price == pytest.approx(100.0)
    assert record.mark_price == pytest.approx(100.0)
    assert record.trade_id is None

    run = env.runs.get(result["run_id"])
    assert run.state is RunState.COMPLETE
    assert run.outcome is RunOutcome.WAIT
    assert run.outcome_reason == "SYMBOL_NOT_ALLOWED"

    assert env.stack.trade_store.all() == []
    assert env.stack.position_store.all() == []
    assert env.loop.status().decisions_today == 1


async def test_stop_loss_closes_position_in_iteration(env) -> None:
    """ST.10: Stop-Loss-Treffer schliesst die Position in derselben Iteration."""
    await env.loop.run_once()  # Iteration 1: FILLED, Position offen (Stop 95 / Target 105)
    env.runtime.reset_script([("NO_TRADE", 0.9)])
    env.provider.set_price("BTCUSDT", 90.0)
    env.clock.advance(seconds=3600)

    result = await env.loop.run_once()

    assert result["decision"] == "NO_TRADE"
    positions = env.stack.position_store.all()
    assert len(positions) == 1
    pos = positions[0]
    assert pos.status is PaperPositionStatus.STOPPED_OUT
    assert pos.close_reason == "STOP_LOSS"
    assert pos.close_price == pytest.approx(90.0)
    # (90 - 100) * 200 - 20 Open-Gebuehr = -2020.
    assert pos.realized_pnl == pytest.approx(-2020.0)

    state = env.loop.status()
    assert state.current_equity == pytest.approx(97980.0)
    assert state.open_positions == 0

    history = env.state_store.portfolio_history()
    assert len(history) == 2
    assert history[1].positions == {}

    records = env.state_store.records()
    assert len(records) == 2
    assert records[1].status is ShadowTradingStatus.NO_TRADE
    assert records[1].mark_price == pytest.approx(90.0)

    closed = [
        a for a in env.runs.get_audit_log() if a.action == "SHADOW_POSITION_CLOSED"
    ]
    assert len(closed) == 1
    assert closed[0].details == {"symbol": "BTCUSDT", "reason": "STOP_LOSS", "price": 90.0}


async def test_target_hit_closes_position_in_iteration(env) -> None:
    """ST.10: Target-Treffer schliesst die Position mit realisiertem Gewinn."""
    await env.loop.run_once()
    env.runtime.reset_script([("NO_TRADE", 0.9)])
    env.provider.set_price("BTCUSDT", 110.0)
    env.clock.advance(seconds=3600)

    await env.loop.run_once()

    pos = env.stack.position_store.all()[0]
    assert pos.status is PaperPositionStatus.TARGET_HIT
    assert pos.close_reason == "TARGET_HIT"
    assert pos.close_price == pytest.approx(110.0)
    # (110 - 100) * 200 - 20 Open-Gebuehr = +1980.
    assert pos.realized_pnl == pytest.approx(1980.0)
    state = env.loop.status()
    assert state.current_equity == pytest.approx(101980.0)
    assert state.open_positions == 0
    closed = [
        a for a in env.runs.get_audit_log() if a.action == "SHADOW_POSITION_CLOSED"
    ]
    assert len(closed) == 1
    assert closed[0].details == {"symbol": "BTCUSDT", "reason": "TARGET_HIT", "price": 110.0}


async def test_kill_switch_stops_loop_within_one_iteration(env) -> None:
    """ST.11: aktiver Kill Switch -> Skip ohne M2M, Self-Stop in Iteration 1."""
    env.kill.activate()

    result = await env.loop.run_once()

    assert result["symbols"][0]["decision"] == "SKIPPED_KILL_SWITCH"
    state = env.loop.status()
    assert state.status is ShadowLoopStatus.STOPPED
    assert state.restart_required is True
    assert state.stopped_at == T0
    assert state.iteration_count == 1
    assert state.decisions_today == 0
    assert env.state_store.records() == []
    assert env.state_store.portfolio_history() == []
    stopped = [
        a
        for a in env.runs.get_audit_log()
        if a.action == "SHADOW_LOOP_STOPPED_KILL_SWITCH"
    ]
    assert len(stopped) == 1
    assert stopped[0].details == {"iteration": 1}


async def test_flags_change_midrun_stops_loop(env) -> None:
    """S2: verletzte Flags mid-run -> M2M laeuft weiter, dann Self-Stop."""
    await env.loop.run_once()  # Iteration 1: TRADE, offene Position
    env.settings.live_execution_enabled = True
    env.clock.advance(seconds=3600)

    second = await env.loop.run_once()

    assert second["decision"] == "SKIPPED_FLAGS"
    state = env.loop.status()
    assert state.status is ShadowLoopStatus.STOPPED
    assert state.restart_required is True
    assert state.stopped_at == T0 + timedelta(seconds=3600)
    assert state.decisions_today == 1
    # M2M ist vor dem Self-Stop noch gelaufen: History hat beide Iterationen,
    # die Position blieb unangetastet offen.
    assert len(env.state_store.portfolio_history()) == 2
    assert env.stack.position_store.all()[0].status is PaperPositionStatus.OPEN
    stopped = [
        a
        for a in env.runs.get_audit_log()
        if a.action == "SHADOW_LOOP_STOPPED_FLAGS_CHANGED"
    ]
    assert len(stopped) == 1
    assert stopped[0].details["iteration"] == 2
    assert stopped[0].details["flags"] == {
        "shadow_trading_enabled": True,
        "live_execution_enabled": True,
    }


async def test_phase2_evaluation_metrics() -> None:
    """ST.12/O4: Phase-2-Metriken ueber akkumulierte Shadow-Records."""
    records = [
        _eval_record("LONG", 10.0),
        _eval_record("SHORT", -5.0),
        _eval_record("SHORT", -7.0),
        _eval_record("LONG", 0.0),
    ]
    history = [
        PortfolioState(run_id="shadow", max_drawdown=0.05),
        PortfolioState(run_id="shadow", max_drawdown=0.12),
    ]

    summary = compute_shadow_evaluation(records, history)

    assert summary.sample_size == 4
    # LONG/LONG korrekt, SHORT/SHORT zweimal korrekt, LONG vs. NO_TRADE falsch.
    assert summary.directional_accuracy == pytest.approx(0.75)
    # Confidence 0.5 -> Brier fuer jede nicht-leere Menge 0.25.
    assert summary.brier_score == pytest.approx(0.25)
    # Win-Rate 0.25: (0.25 * 10) - (0.75 * 6) = -2.0.
    assert summary.expectancy == pytest.approx(-2.0)
    assert summary.max_drawdown == pytest.approx(0.12)


async def test_loop_writes_logs_and_audit_entries(env, caplog) -> None:
    """O1/ST.13: strukturiertes INFO-Log plus vollstaendige Audit-Kette."""
    caplog.set_level(logging.INFO, logger="trading_harness.services.shadow_trading_loop")
    result = await env.loop.run_once()

    assert "SHADOW_ITERATION iteration=1 symbol=BTCUSDT" in caplog.text
    assert "decision=TRADE risk_result=APPROVED execution_result=FILLED" in caplog.text
    assert "equity=100000.00" in caplog.text

    run_audit = env.runs.get_audit_log(result["run_id"])
    assert "SHADOW_ITERATION" in [a.action for a in run_audit]
    iteration_entry = next(a for a in run_audit if a.action == "SHADOW_ITERATION")
    assert iteration_entry.details["iteration"] == 1
    assert iteration_entry.details["symbol"] == "BTCUSDT"
    assert iteration_entry.details["decision"] == "TRADE"
    assert iteration_entry.details["risk_result"] == "APPROVED"
    assert iteration_entry.details["execution_result"] == "FILLED"
    all_audit = env.runs.get_audit_log()
    decision_entry = next(a for a in all_audit if a.action == "SHADOW_DECISION")
    assert decision_entry.details["status"] == "FILLED"
    assert decision_entry.details["risk_reason"] == "APPROVED"
    fill_entry = next(a for a in all_audit if a.action == "SHADOW_FILL")
    assert fill_entry.entity_type == "paper_trade"
    assert fill_entry.details["quantity"] == 200.0
    assert fill_entry.details["filled_price"] == pytest.approx(100.0)


async def test_loop_deterministic_with_fixed_inputs(tmp_path) -> None:
    """ST.14: identische Eingaben -> identische IDs, Records und Session-State."""
    env1 = build_env(tmp_path / "env1")
    env2 = build_env(tmp_path / "env2")

    first = await env1.loop.run_once()
    second = await env2.loop.run_once()

    assert first["snapshot_id"] == second["snapshot_id"]
    assert first["run_id"] == second["run_id"]
    assert first["decision"] == second["decision"]

    rec1 = env1.state_store.records()[0]
    rec2 = env2.state_store.records()[0]
    assert _record_fingerprint(rec1) == _record_fingerprint(rec2)

    hist1 = env1.state_store.portfolio_history()
    hist2 = env2.state_store.portfolio_history()
    assert hist1[0].positions == {"BTCUSDT": 200.0}
    assert hist2[0].positions == hist1[0].positions

    for state in (env1.loop.status(), env2.loop.status()):
        assert state.iteration_count == 1
        assert state.decisions_today == 1
        assert state.budget_date == "2026-08-20"
        assert state.current_equity == pytest.approx(100000.0)


async def test_loop_stops_after_consecutive_errors(env) -> None:
    """F6: nach MAX_CONSECUTIVE_ERRORS Iterationsfehlern stoppt sich der Loop selbst."""
    env.state_store.update_state({"status": "RUNNING"})
    env.provider.fail = RuntimeError("ticker down")

    for _ in range(MAX_CONSECUTIVE_ERRORS):
        result = await env.loop.run_once()
        assert result["decision"] == "SKIPPED_MARKET_DATA"

    state = env.loop.status()
    assert state.status is ShadowLoopStatus.STOPPED
    assert state.restart_required is True
    assert state.error_count == MAX_CONSECUTIVE_ERRORS
    assert state.last_error == "TICKER_ERROR: BTCUSDT: ticker down"
    assert state.iteration_count == MAX_CONSECUTIVE_ERRORS
    assert state.decisions_today == 0
    # Keine Entscheidung erzeugt Records; M2M lief in jeder Iteration weiter.
    assert env.state_store.records() == []
    assert len(env.state_store.portfolio_history()) == MAX_CONSECUTIVE_ERRORS
    stopped = [
        a
        for a in env.runs.get_audit_log()
        if a.action == "SHADOW_LOOP_STOPPED_ERRORS"
    ]
    assert len(stopped) == 1
    assert stopped[0].details == {"last_error": "TICKER_ERROR: BTCUSDT: ticker down"}


async def test_state_file_size_bounded(env, monkeypatch) -> None:
    """ST.15: Record-/History-Puffer bleiben begrenzt; Lifetime-Aggregate laufen weiter."""
    await env.loop.run_once()  # echter Durchlauf: 1 FILLED-Record + 1 Portfolio-State
    monkeypatch.setattr("trading_harness.services.shadow_trading_state.MAX_RECORDS", 5)
    monkeypatch.setattr(
        "trading_harness.services.shadow_trading_state.MAX_PORTFOLIO_HISTORY", 10
    )
    monkeypatch.setattr(
        "trading_harness.services.shadow_trading_state._RECENT_PORTFOLIO_EXACT", 4
    )

    real = env.state_store.records()[0]
    env.state_store.add_records([real] * 12)  # 13 gesamt -> nur die neuesten 5 bleiben
    assert len(env.state_store.records()) == 5
    summary = env.state_store.records_summary()
    assert summary["total_records"] == 13
    assert summary["trimmed_count"] == 8
    assert summary["by_status"] == {"FILLED": 13}

    env.state_store.add_portfolio_states([PortfolioState(run_id="shadow")] * 25)
    assert len(env.state_store.portfolio_history()) == 10

    doc = json.loads(env.state_store.state_path.read_text(encoding="utf-8"))
    assert len(doc["records"]) == 5
    assert len(doc["portfolio_history"]) == 10
    assert doc["records_summary"]["total_records"] == 13


async def test_incomplete_fill_data_fail_closed(env) -> None:
    """ST.3/S3: FILLED ohne vollstaendige Fill-Daten wird fail-closed zu REJECTED."""
    incomplete = ShadowExecutionResult(
        trade_id=None,
        status="FILLED",
        filled_price=100.0,
        quantity=200.0,
        reason=None,
    )
    with patch.object(ShadowExecutionBackend, "execute", return_value=incomplete):
        result = await env.loop.run_once()

    assert result["decision"] == "TRADE"
    assert result["symbols"][0]["risk_result"] == "APPROVED"
    assert (
        result["symbols"][0]["execution_result"]
        == "ERROR: EXECUTION_ERROR: INCOMPLETE_FILL_DATA"
    )

    record = env.state_store.records()[0]
    assert record.status is ShadowTradingStatus.REJECTED
    assert record.risk_reason == "EXECUTION_ERROR: INCOMPLETE_FILL_DATA"
    assert record.trade_id is None
    # Fail-closed protokolliert das unverbrauchte Preliminary-Proposal (250 @ 100).
    assert record.quantity == pytest.approx(250.0)
    assert record.entry_price == pytest.approx(100.0)

    run = env.runs.get(result["run_id"])
    assert run.state is RunState.COMPLETE
    assert run.outcome is RunOutcome.WAIT
    assert run.outcome_reason == "EXECUTION_ERROR: INCOMPLETE_FILL_DATA"

    assert env.stack.trade_store.all() == []
    assert env.stack.position_store.all() == []
    assert env.loop.status().decisions_today == 1
