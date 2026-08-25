"""API-Fassade für den Shadow-Trading-Loop (WI-ST-06).

Assembliert den ``ShadowTradingLoop`` aus den verdrahteten Singletons der
API-Schicht und kapselt die Operationen der ``/shadow-trading/*``-Endpunkte.
Kein Autostart (Z2): Der Loop läuft ausschließlich nach explizitem
``start()``; ``shutdown()`` stoppt einen laufenden Loop beim App-Shutdown.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol

from pydantic import BaseModel

from trading_harness.config import Settings
from trading_harness.models import (
    PortfolioState,
    ShadowLoopStatus,
    ShadowSessionState,
    ShadowTradingRecord,
)
from trading_harness.services.agent_genome_store import AgentGenomeStore
from trading_harness.services.agent_runtime import AgentRuntime
from trading_harness.services.kill_switch import KillSwitch
from trading_harness.services.orchestrator import TradingRunService
from trading_harness.services.persisted_snapshot_store import PersistedSnapshotStore
from trading_harness.services.risk_engine import RiskEngine
from trading_harness.services.shadow_execution_backend import ShadowExecutionBackend
from trading_harness.services.shadow_trading_loop import (
    CryptoMarketDataProvider,
    ShadowTradingLoop,
)
from trading_harness.services.shadow_trading_state import ShadowTradingStateStore
from trading_harness.services.snapshot_store import SnapshotStore


class _CryptoRouterLike(Protocol):
    """Struktureller Vertrag für den Crypto-Router (Spiegel ``_RouterLike``)."""

    def get_ticker(self, symbol: str) -> dict[str, float]: ...


def _default_clock() -> datetime:
    return datetime.now(UTC)


class ShadowPortfolioReport(BaseModel):
    """Antwortmodell für GET /shadow-trading/portfolio."""

    current_equity: float
    start_equity: float
    history: list[PortfolioState]


class ShadowTradingService:
    """Assembliert und kapselt den Shadow-Loop hinter einer schmalen Fassade.

    Alle Operationen delegieren 1:1 an den Loop; die Fassade fügt keine
    Geschäftslogik hinzu, sondern nur Assemblierung (Wiring) sowie
    Lese-Filter für Records und Portfolio.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        backend: ShadowExecutionBackend,
        crypto_router: _CryptoRouterLike,
        trading_run_service: TradingRunService,
        snapshot_store: SnapshotStore | PersistedSnapshotStore,
        risk_engine: RiskEngine,
        kill_switch: KillSwitch,
        agent_source: AgentGenomeStore,
        agent_runtime: AgentRuntime,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._clock = clock or _default_clock
        self._state_store = ShadowTradingStateStore(settings.shadow_state_path)
        provider = CryptoMarketDataProvider(router=crypto_router)
        self._loop = ShadowTradingLoop(
            market_data=provider,
            backend=backend,
            agent_runtime=agent_runtime,
            trading_run_service=trading_run_service,
            snapshot_store=snapshot_store,
            risk_engine=risk_engine,
            kill_switch=kill_switch,
            agent_source=agent_source,
            settings=settings,
            state_store=self._state_store,
            clock=self._clock,
        )

    @property
    def loop(self) -> ShadowTradingLoop:
        """Direktzugriff auf den Loop (Test-Injektion von Stubs)."""
        return self._loop

    async def start(self) -> ShadowSessionState:
        """Startet den Loop nach Prüfung der Guards; liefert den Session-State."""
        await self._loop.start()
        return self._loop.status()

    async def stop(self, reason: str = "API") -> ShadowSessionState:
        """Stoppt den Loop graceful und idempotent; liefert den End-State."""
        await self._loop.stop(reason=reason)
        return self._loop.status()

    async def run_once(self) -> dict[str, object]:
        """Führt genau eine Iteration aus (ST.1), unabhängig vom Loop-Zustand."""
        result: dict[str, object] = await self._loop.run_once()
        return result

    def status(self) -> ShadowSessionState:
        """Aktueller persistenter Session-State (ST.8)."""
        return self._loop.status()

    def records(
        self,
        *,
        symbol: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[ShadowTradingRecord]:
        """Entscheidungs-Records chronologisch aufsteigend (ST.7).

        ``symbol`` filtert exakt, ``status`` case-insensitive gegen den
        Enum-Wert; ``limit`` liefert die NEUESTEN N Einträge (Reihenfolge
        bleibt aufsteigend).
        """
        records = self._state_store.records()
        if symbol is not None:
            records = [record for record in records if record.symbol == symbol]
        if status is not None:
            wanted = status.upper()
            records = [record for record in records if record.status.value == wanted]
        if limit is not None:
            # limit=0 muss leer sein; [-0:] würde die ganze Liste liefern.
            records = records[-limit:] if limit > 0 else []
        return records

    def portfolio(self, *, limit: int | None = None) -> ShadowPortfolioReport:
        """Equity-Report mit M2M-Historie (chronologisch, limit = neueste N)."""
        state = self._loop.status()
        history = self._state_store.portfolio_history()
        if limit is not None:
            history = history[-limit:] if limit > 0 else []
        return ShadowPortfolioReport(
            current_equity=state.current_equity,
            start_equity=state.start_equity,
            history=history,
        )

    async def shutdown(self) -> None:
        """Lifespan-Shutdown: stoppt ausschließlich einen LAUFENDEN Loop (Z2)."""
        if self._loop.status().status is ShadowLoopStatus.RUNNING:
            await self._loop.stop(reason="SHUTDOWN")
