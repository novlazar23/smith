"""FastAPI-Einstiegspunkt mit Graceful-Shutdown für Shadow-Trading (WI-ST-06)."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from trading_harness.api import quant_routes, routes


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Z2: kein Autostart beim App-Start; beim Shutdown wird ein laufender
    # Shadow-Loop kontrolliert gestoppt.
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
