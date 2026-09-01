"""Gemeinsame Fakes und Helfer für die Demo-Trader-Tests.

Alle Abhängigkeiten (Pipeline, Candle-Provider, DB) werden injiziert —
es gibt keine Netzwerk- oder Datenbank-Zugriffe.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from apps.demo_trader.service import DemoTrader, DemoTraderConfig
from apps.orchestrator_service.service import CandleWindow
from packages.consensus import ConsensusDecision, ConsensusResult
from packages.orchestrator.pipeline import OrchestratorPipelineResult
from packages.paper import PaperExecutor

BTC = "BTC/USDT"
ETH = "ETH/USDT"


class FakeConnection:
    """Duck-typed Ersatz für eine SQLAlchemy-Connection (rekordiert Aufrufe)."""

    def __init__(self) -> None:
        self.executed: list[tuple[object, dict]] = []
        self.commits = 0
        self.fail_commit = False

    def execute(self, statement: object, parameters: dict) -> None:
        self.executed.append((statement, parameters))

    def commit(self) -> None:
        if self.fail_commit:
            raise RuntimeError("commit failed")
        self.commits += 1


class FakeConnectionContext:
    """Kontextmanager, der eine FakeConnection yieldet."""

    def __init__(self, conn: FakeConnection) -> None:
        self._conn = conn

    def __enter__(self) -> FakeConnection:
        return self._conn

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False


class FakeDBEngine:
    """Stellvertreter für die SQLAlchemy-Engine (connect → ContextManager)."""

    def __init__(self, conn: FakeConnection) -> None:
        self._conn = conn

    def connect(self) -> FakeConnectionContext:
        return FakeConnectionContext(self._conn)


class FakeDB:
    """Stellvertreter für SQLAlchemyEngine (hat .engine)."""

    def __init__(self, conn: FakeConnection) -> None:
        self.engine = FakeDBEngine(conn)


class StubCandleSource:
    """Stellvertreter für den Candle-Provider (konfigurierbare Fenster)."""

    def __init__(self, windows: dict[str, CandleWindow | None]) -> None:
        self.windows = windows
        self.requests: list[tuple[str, int]] = []
        self.fail_for: tuple[str, ...] = ()

    def fetch_candles(self, instrument: str, limit: int) -> CandleWindow | None:
        self.requests.append((instrument, limit))
        if instrument in self.fail_for:
            raise RuntimeError(f"clickhouse unavailable for {instrument}")
        return self.windows.get(instrument)


class StubPipeline:
    """Stellvertreter für OrchestratorPipeline (vorgefertigte Results)."""

    def __init__(self, results: dict[str, OrchestratorPipelineResult]) -> None:
        self.results = results
        self.calls: list[dict] = []

    def run(
        self,
        run_id: str,
        instrument: str,
        agents: list,
        market_data: dict,
    ) -> OrchestratorPipelineResult:
        self.calls.append(
            {"run_id": run_id, "instrument": instrument, "agents": agents, "market_data": market_data}
        )
        if instrument in self.results:
            return self.results[instrument]
        raise RuntimeError(f"stub pipeline failure for {instrument}")


def make_ohlcv(n: int, start_price: float = 100.0) -> CandleWindow:
    """Erzeugt ein deterministisches OHLCV-Kerzenfenster mit n Kerzen."""
    close = np.linspace(start_price, start_price + n, n)
    return CandleWindow(
        open=close - 0.1,
        high=close + 0.5,
        low=close - 0.5,
        close=close,
        volume=np.full(n, 1000.0),
    )


def make_result(
    decision: str = "NO_TRADE",
    confidence: float = 0.0,
    reason: str = "demo",
    first: int = 3,
    second: int = 3,
) -> OrchestratorPipelineResult:
    """Baut ein OrchestratorPipelineResult mit Konsens."""
    consensus = ConsensusResult(
        decision=ConsensusDecision(decision),
        vote_distribution={},
        agent_weights={},
        agent_agreements=[],
        agent_disagreements=[],
        confidence=confidence,
        reason=reason,
    )
    return OrchestratorPipelineResult(
        decision=decision,
        consensus=consensus,
        first_round_reports=[object() for _ in range(first)],
        seal_records=[],
        second_round_reports=[object() for _ in range(second)],
    )


def make_trader(
    config: DemoTraderConfig,
    provider: StubCandleSource,
    conn: FakeConnection,
    pipeline: StubPipeline,
    executor: PaperExecutor | None = None,
) -> DemoTrader:
    """Baut einen DemoTrader mit injizierten Stubs."""
    return DemoTrader(
        config=config,
        provider=provider,
        db=FakeDB(conn),
        executor=executor if executor is not None else PaperExecutor(initial_cash=config.initial_cash),
        pipeline_factory=lambda: pipeline,
    )


@pytest.fixture
def fake_conn() -> FakeConnection:
    return FakeConnection()


@pytest.fixture
def stub_provider() -> StubCandleSource:
    return StubCandleSource({})


@pytest.fixture
def stub_pipeline() -> StubPipeline:
    return StubPipeline({})


@pytest.fixture
def config(tmp_path: Path) -> DemoTraderConfig:
    return DemoTraderConfig(
        interval_seconds=300.0,
        instruments=(BTC, ETH),
        initial_cash=100000.0,
        trade_notional=2000.0,
        min_confidence=0.4,
        candle_venue="DUMMY_EXCHANGE",
        candle_limit=200,
        min_candles=30,
        horizon="15m",
        account_id="demo",
        heartbeat_path=tmp_path / "demo-heartbeat",
        log_level="INFO",
    )
