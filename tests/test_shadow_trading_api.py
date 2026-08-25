"""Tests für die Shadow-Trading-API und das Lifespan-Wiring (WI-ST-06)."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from trading_harness.config import Settings
from trading_harness.main import app
from trading_harness.models import (
    AgentAnalysisResult,
    AgentGenome,
    AgentSignal,
    AgentStatus,
)
from trading_harness.services.agent_genome_store import AgentGenomeStore
from trading_harness.services.kill_switch import KillSwitch
from trading_harness.services.orchestrator import TradingRunService
from trading_harness.services.paper_execution_stack import build_paper_execution_stack
from trading_harness.services.risk_engine import RiskEngine
from trading_harness.services.shadow_execution_backend import ShadowExecutionBackend
from trading_harness.services.shadow_trading_service import ShadowTradingService
from trading_harness.services.snapshot_store import SnapshotStore

T0 = datetime(2026, 8, 24, 12, 0, 0, tzinfo=UTC)
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
    """Deterministische, manuell vorlaufende Uhr (Spiegel aus test_shadow_trading_loop)."""

    def __init__(self, start: datetime) -> None:
        self._now = start

    def __call__(self) -> datetime:
        return self._now

    def advance(self, *, days: int = 0, seconds: float = 0.0) -> None:
        self._now = self._now + timedelta(days=days, seconds=seconds)


class ScriptedTickerRouter:
    """Duck-typed Crypto-Router: get_ticker liefert ``{"last": price}``."""

    def __init__(self, prices: dict[str, float]) -> None:
        self._prices = dict(prices)

    def get_ticker(self, symbol: str) -> dict[str, float]:
        return {"last": self._prices[symbol]}


class FakeAgentRuntime:
    """Deterministischer Agenten-Stub: Script-Einträge pro analyze()-Aufruf.

    Das Script wird nach dem letzten Eintrag gehalten (Repeat), damit
    mehr Agenten als Skript-Einträge bedient werden können.
    """

    def __init__(self, script: list[tuple[str, float]]) -> None:
        self._script = list(script)
        self._index = 0

    async def analyze(self, agent, snapshot, run_id=None):
        direction, confidence = self._script[min(self._index, len(self._script) - 1)]
        self._index += 1
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


def build_env(root: Path, **setting_overrides: Any) -> SimpleNamespace:
    """Baue eine isolierte ShadowTradingService-Umgebung unter ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    settings_kwargs: dict[str, Any] = {
        "_env_file": None,
        "shadow_trading_enabled": True,
        "shadow_trading_symbols": ["BTCUSDT", "ETHUSDT"],
        "shadow_loop_interval_seconds": 900,
        "shadow_state_path": str(root / "shadow_trading_state.json"),
    }
    settings_kwargs.update(setting_overrides)
    settings = Settings(**settings_kwargs)
    provider = ScriptedTickerRouter({"BTCUSDT": 100.0, "ETHUSDT": 2000.0})
    runtime = FakeAgentRuntime([("LONG", 0.9)])
    agents = AgentGenomeStore()
    agents.add(AgentGenome(id=AGENT_ID, category="technical", status=AgentStatus.ACTIVE))
    kill = KillSwitch(db_path=str(root / "kill_switch.json"))
    clock = MutableClock(T0)
    service = ShadowTradingService(
        settings,
        backend=ShadowExecutionBackend(build_paper_execution_stack(db=None)),
        crypto_router=provider,
        trading_run_service=TradingRunService(),
        snapshot_store=SnapshotStore(),
        risk_engine=RiskEngine(dict(POLICY)),
        kill_switch=kill,
        agent_source=agents,
        agent_runtime=runtime,
        clock=clock,
    )
    return SimpleNamespace(
        root=root,
        settings=settings,
        provider=provider,
        runtime=runtime,
        kill=kill,
        clock=clock,
        service=service,
    )


@pytest.fixture()
def api_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Isolierte Service-Instanz hinter dem echten App-Router."""
    from trading_harness.api import routes

    env = build_env(tmp_path / "api-env")
    monkeypatch.setattr(routes, "shadow_trading_service", env.service)
    return SimpleNamespace(env=env, client=TestClient(app))


def patched_client(monkeypatch: pytest.MonkeyPatch, env: SimpleNamespace) -> TestClient:
    """Ersetze den API-Singleton durch eine Test-Umgebung."""
    from trading_harness.api import routes

    monkeypatch.setattr(routes, "shadow_trading_service", env.service)
    return TestClient(app)


def lifecycle_actions(env: SimpleNamespace) -> list[str]:
    """Audit-Aktionen der aktuellen Session (entity_id = session_id)."""
    runs = env.service._loop._runs
    session_id = env.service.status().session_id
    return [entry.action for entry in runs.get_audit_log(entity_id=session_id)]


class TestShadowStartEndpoint:
    """POST /shadow-trading/start (Spec ST.2): Guards als HTTP-Codes."""

    def test_start_returns_running_session(self, api_env):
        with api_env.client:
            response = api_env.client.post("/shadow-trading/start")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "RUNNING"
        assert body["symbols"] == ["BTCUSDT", "ETHUSDT"]
        assert body["interval_seconds"] == 900
        assert body["restart_required"] is False
        assert re.fullmatch(r"shadow-[0-9a-f-]{36}", body["session_id"]) is not None
        assert "SHADOW_LOOP_STARTED" in lifecycle_actions(api_env.env)

    def test_start_twice_returns_conflict(self, api_env):
        with api_env.client:
            first = api_env.client.post("/shadow-trading/start")
            second = api_env.client.post("/shadow-trading/start")

        assert first.status_code == 200
        assert second.status_code == 409
        assert second.json()["detail"]["code"] == "ALREADY_RUNNING"

    def test_start_blocked_when_live_execution_enabled(self, tmp_path, monkeypatch):
        env = build_env(tmp_path / "live", live_execution_enabled=True)
        client = patched_client(monkeypatch, env)
        with client:
            response = client.post("/shadow-trading/start")

        assert response.status_code == 403
        assert response.json()["detail"]["code"] == "LIVE_EXECUTION_MUST_BE_DISABLED"

    def test_start_blocked_when_shadow_disabled(self, tmp_path, monkeypatch):
        env = build_env(tmp_path / "off", shadow_trading_enabled=False)
        client = patched_client(monkeypatch, env)
        with client:
            response = client.post("/shadow-trading/start")

        assert response.status_code == 403
        assert response.json()["detail"]["code"] == "SHADOW_TRADING_DISABLED"

    def test_start_blocked_without_symbols(self, tmp_path, monkeypatch):
        env = build_env(tmp_path / "empty", shadow_trading_symbols=[])
        client = patched_client(monkeypatch, env)
        with client:
            response = client.post("/shadow-trading/start")

        assert response.status_code == 403
        assert response.json()["detail"]["code"] == "NO_SYMBOLS_CONFIGURED"


class TestShadowStopEndpoint:
    """POST /shadow-trading/stop: graceful, idempotent, restart_required."""

    def test_stop_is_idempotent_and_sets_restart_required(self, api_env):
        with api_env.client:
            before = api_env.client.post("/shadow-trading/stop").json()
            started = api_env.client.post("/shadow-trading/start")
            after = api_env.client.post("/shadow-trading/stop").json()
            again = api_env.client.post("/shadow-trading/stop")

        assert started.status_code == 200
        assert before["status"] == "STOPPED"
        assert before["restart_required"] is True
        assert after["status"] == "STOPPED"
        assert again.status_code == 200
        assert "SHADOW_LOOP_STOPPED" in lifecycle_actions(api_env.env)


def test_status_endpoint_reflects_fresh_session(api_env):
    """GET /shadow-trading/status ohne Interaktion: Persistenter Frischzustand."""
    with api_env.client:
        response = api_env.client.get("/shadow-trading/status")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "STOPPED"
    assert body["iteration_count"] == 0
    assert body["decisions_today"] == 0
    assert body["current_equity"] == 100000.0
    assert body["open_positions"] == 0


def test_run_once_executes_full_chain(tmp_path, monkeypatch):
    """POST /shadow-trading/run-once: Ticker -> Agents -> Risiko -> Fill -> Budget."""
    env = build_env(tmp_path / "chain", shadow_trading_symbols=["BTCUSDT"])
    client = patched_client(monkeypatch, env)
    with client:
        response = client.post("/shadow-trading/run-once")
        status = client.get("/shadow-trading/status").json()
        records = client.get("/shadow-trading/records").json()

    assert response.status_code == 200
    data = response.json()
    sid = data["snapshot_id"]
    assert re.fullmatch(r"snap-[0-9a-f]{16}", sid) is not None
    assert data["run_id"] == f"shadow-run-{sid}"
    assert data["decision"] == "TRADE"
    entry = data["symbols"][0]
    assert entry["symbol"] == "BTCUSDT"
    assert entry["execution_result"] == "FILLED"
    assert status["iteration_count"] == 1
    assert status["decisions_today"] == 1
    assert len(records) == 1
    assert records[0]["status"] == "FILLED"
    assert records[0]["side"] == "BUY"


def test_records_endpoint_filters_symbol_status_limit(api_env):
    """GET /shadow-trading/records: Filter symbol/status, limit = neueste N."""
    with api_env.client:
        api_env.client.post("/shadow-trading/run-once")
        all_records = api_env.client.get("/shadow-trading/records").json()
        eth_only = api_env.client.get(
            "/shadow-trading/records", params={"symbol": "ETHUSDT"}
        ).json()
        filled = api_env.client.get(
            "/shadow-trading/records", params={"status": "filled"}
        ).json()
        limited = api_env.client.get("/shadow-trading/records", params={"limit": 1}).json()
        invalid = api_env.client.get("/shadow-trading/records", params={"limit": 0})

    assert [r["symbol"] for r in all_records] == ["BTCUSDT", "ETHUSDT"]
    assert [r["symbol"] for r in eth_only] == ["ETHUSDT"]
    assert len(filled) == 2
    assert [r["symbol"] for r in limited] == ["ETHUSDT"]
    assert invalid.status_code == 422


def test_portfolio_endpoint_reports_equity_and_history(api_env):
    """GET /shadow-trading/portfolio: Equity plus Positions-Historie nach M2M."""
    with api_env.client:
        initial = api_env.client.get("/shadow-trading/portfolio").json()
        api_env.client.post("/shadow-trading/run-once")
        api_env.client.post("/shadow-trading/run-once")
        after = api_env.client.get("/shadow-trading/portfolio").json()

    assert initial["current_equity"] == 100000.0
    assert initial["history"] == []
    assert after["start_equity"] == 100000.0
    assert after["current_equity"] > 0
    assert len(after["history"]) >= 2
    final_positions = after["history"][-1]["positions"]
    assert (
        final_positions.get("BTCUSDT") is not None
        or final_positions.get("ETHUSDT") is not None
    )


class TestLifespanWiring:
    """FastAPI-Lifespan (Z2): kein Autostart, graceful Shutdown."""

    def test_lifespan_shutdown_stops_running_loop(self, api_env):
        with api_env.client:
            started = api_env.client.post("/shadow-trading/start").json()
            assert started["status"] == "RUNNING"

        state = api_env.env.service.status()
        assert state.status.value == "STOPPED"
        assert state.restart_required is True

    def test_lifespan_does_not_autostart_loop(self, api_env):
        with api_env.client:
            pass

        state = api_env.env.service.status()
        assert state.status.value == "STOPPED"


def test_all_six_endpoints_exist_with_auth_dependencies():
    """Alle sechs Endpunkte vorhanden; Trade-Key schreibt, Read-Key liest."""
    from trading_harness.api import routes as routes_module

    # FastAPI inkludiert Router lazy (_IncludedRouter); Pfade stehen auf dem Router.
    by_path = {
        getattr(route, "path", ""): route for route in routes_module.router.routes
    }
    expected = {
        "/shadow-trading/start",
        "/shadow-trading/stop",
        "/shadow-trading/run-once",
        "/shadow-trading/status",
        "/shadow-trading/records",
        "/shadow-trading/portfolio",
    }
    assert expected <= set(by_path)
    trade_paths = {"/shadow-trading/start", "/shadow-trading/stop", "/shadow-trading/run-once"}
    for path in sorted(trade_paths):
        deps = {d.call.__name__ for d in by_path[path].dependant.dependencies}
        assert "require_trade_key" in deps, path
    for path in sorted(expected - trade_paths):
        deps = {d.call.__name__ for d in by_path[path].dependant.dependencies}
        assert "require_read_key" in deps, path
