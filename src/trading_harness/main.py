"""FastAPI-Einstiegspunkt mit Graceful-Shutdown für Shadow-Trading (WI-ST-06).

Bindet die Haupt-Router (``routes``) sowie die Quant-Plattform-Router
(``quant_routes``, Prefix ``/quant``) ein, damit die Quant-API im
Docker-/Produktivbetrieb erreichbar ist. Der Lifespan initialisiert die
PostgreSQL-Persistenz, übernimmt optional den verschlüsselten Runtime-State
des System-Handoffs und startet bei Bedarf den autonomen Shadow-Loop.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from trading_harness.config import get_settings
from trading_harness.services.state_handoff import HandoffCoordinator

logger = logging.getLogger(__name__)
settings = get_settings()


def _handoff_coordinator() -> HandoffCoordinator | None:
    if not settings.state_handoff_enabled:
        return None
    password = settings.state_handoff_password.get_secret_value()
    if not password or not settings.state_node_id:
        raise RuntimeError(
            "STATE_HANDOFF_PASSWORD and STATE_NODE_ID are required when state handoff is enabled"
        )
    return HandoffCoordinator(
        settings.state_data_dir,
        settings.state_bundle_path,
        password,
        settings.state_node_id,
        lease_seconds=settings.state_handoff_lease_seconds,
    )


# Restore before importing API singletons so every persistent store sees the transferred state.
handoff_coordinator = _handoff_coordinator()
if handoff_coordinator is not None and handoff_coordinator.bundle_path.exists():
    handoff_coordinator.hand_on()

from trading_harness.api import quant_routes, routes


def _initialize_database() -> None:
    """Erzwingt Pool- und Schema-Initialisierung beim App-Start.

    Ohne diese Initialisierung bleibt ``Database.is_available`` False, solange kein
    Direktzugriff auf ``execute()`` erfolgt — alle Stores würden dauerhaft ihren
    In-Memory-Fallback nutzen, obwohl PostgreSQL erreichbar ist.
    """
    ready = routes.initialize_database()
    if ready:
        logger.info("PostgreSQL verbunden, persistente Stores aktiv")
    else:
        logger.warning("PostgreSQL nicht erreichbar — Stores laufen im In-Memory-Fallback")


async def _sync_handoff() -> None:
    assert handoff_coordinator is not None
    while True:
        await asyncio.sleep(max(1, settings.state_handoff_sync_seconds))
        handoff_coordinator.renew()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    sync_task: asyncio.Task[None] | None = None
    _initialize_database()
    if handoff_coordinator is not None:
        # A missing initial bundle becomes a leased encrypted generation immediately.
        if not handoff_coordinator.bundle_path.exists():
            handoff_coordinator.hand_off()
            handoff_coordinator.hand_on()
        sync_task = asyncio.create_task(_sync_handoff())
    if settings.autonomous_shadow_enabled:
        routes.seed_champion_population()
        await routes.shadow_trading_service.start()
    try:
        yield
    finally:
        await routes.shadow_trading_service.shutdown()
        if sync_task is not None:
            sync_task.cancel()
            await asyncio.gather(sync_task, return_exceptions=True)
        if handoff_coordinator is not None:
            handoff_coordinator.hand_off()


app = FastAPI(
    title="Evolutionary Trading Harness",
    version="0.1.0",
    description="Shadow-first evolutionary multi-agent trading research service",
    lifespan=lifespan,
)
app.include_router(routes.router)
app.include_router(quant_routes.router)