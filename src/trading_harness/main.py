"""FastAPI-Einstiegspunkt mit Graceful-Shutdown für Shadow-Trading (WI-ST-06).

Bindet die Haupt-Router (``routes``) sowie die Quant-Plattform-Router
(``quant_routes``, Prefix ``/quant``) ein, damit die Quant-API im
Docker-/Produktivbetrieb erreichbar ist.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from trading_harness.api import quant_routes, routes

logger = logging.getLogger(__name__)


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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Z2: kein Autostart beim App-Start; beim Shutdown wird ein laufender
    # Shadow-Loop kontrolliert gestoppt.
    _initialize_database()
    yield
    await routes.shadow_trading_service.shutdown()


app = FastAPI(
    title="Evolutionary Trading Harness",
    version="0.1.0",
    description="Shadow-first evolutionary multi-agent trading research service",
    lifespan=lifespan,
)
app.include_router(routes.router)
app.include_router(quant_routes.router)
