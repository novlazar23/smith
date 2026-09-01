"""Gemeinsame Fakes und Helfer für die Orchestrator-Service-Tests.

Alle Abhängigkeiten (Pipeline, Candle-Provider, DB) werden injiziert —
es gibt keine Netzwerk- oder Datenbank-Zugriffe.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from apps.orchestrator_service.service import (
    CandleWindow,
    OrchestratorService,
    OrchestratorServiceConfig,
)
from packages.consensus import ConsensusDecision, ConsensusResult
from packages.orchestrator.pipeline import OrchestratorPipelineResult


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


class StubProvider:
    """Stellvertreter für die CandleProvider (konfigurierbare Fenster)."""

    def __init__(self, windows: dict[str, CandleWindow | None]) -> None:
        self.windows = windows
        self.requests: list[tuple[str, int]] = []

    def fetch_candles(self, instrument: str, limit: int) -> CandleWindow | None:
        self.requests.append((instrument, limit))
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
    reason: str = "shadow",
    first: int = 3,
    second: int = 3,
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
) -> OrchestratorPipelineResult:
    """Baut ein OrchestratorPipelineResult mit Konsens."""
    consensus = ConsensusResult(
        decision=ConsensusDecision.NO_TRADE,
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
        errors=errors or [],
        warnings=warnings or [],
    )


@pytest.fixture
def fake_conn() -> FakeConnection:
    return FakeConnection()


@pytest.fixture
def fake_db(fake_conn: FakeConnection) -> FakeDB:
    return FakeDB(fake_conn)


@pytest.fixture
def stub_provider() -> StubProvider:
    return StubProvider({})


@pytest.fixture
def stub_pipeline() -> StubPipeline:
    return StubPipeline({})


@pytest.fixture
def config(tmp_path: Path) -> OrchestratorServiceConfig:
    return OrchestratorServiceConfig(
        interval_seconds=900.0,
        instruments=("BTC/USDT", "ETH/USDT"),
        candle_limit=200,
        min_candles=30,
        horizon="15m",
        heartbeat_path=tmp_path / "heartbeat",
        log_level="INFO",
    )


@pytest.fixture
def service(
    config: OrchestratorServiceConfig,
    stub_provider: StubProvider,
    fake_db: FakeDB,
    stub_pipeline: StubPipeline,
) -> OrchestratorService:
    return OrchestratorService(
        config=config,
        provider=stub_provider,
        db=fake_db,
        pipeline_factory=lambda: stub_pipeline,
    )
