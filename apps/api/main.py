"""FastAPI REST API für die Trading-Orchestra Analyse-Fähigkeiten.

Dieses Modul stellt die FastAPI-Anwendung bereit, die die Analyse-Fähigkeiten
des Trading-Systems über eine REST-Schnittstelle exponiert. FastAPI und pydantic
sind optionale Abhängigkeiten — bei Nicht-Verfügbarkeit gibt die API einen
grazilen Fehler zurück.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

# FastAPI import — optional dependency
try:
    from fastapi import FastAPI, HTTPException, status
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None  # type: ignore[assignment,misc]
    HTTPException = None  # type: ignore[assignment,misc]
    BaseModel = None  # type: ignore[assignment,misc]
    Field = None  # type: ignore[assignment]
    FASTAPI_AVAILABLE = False

from apps.api.endpoints import (
    analyze_endpoint,
    status_endpoint,
)

logger = logging.getLogger(__name__)

VERSION = "0.1.0"


if FASTAPI_AVAILABLE:

    class AnalyzeRequest(BaseModel):
        """Anfrage-Modell für die Analyse-Endpunktes."""

        instrument: str = Field(..., min_length=1, max_length=50)
        horizons: list[str] = Field(default_factory=lambda: ["1m", "5m", "15m"])
        strategy: dict[str, object] = Field(default_factory=dict)

    class StatusResponse(BaseModel):
        """Antwort-Modell für den Status-Endpunkt."""

        version: str
        status: str
        uptime_seconds: float
        modules: dict[str, str]
        timestamp: str

    def _check_module_available(module_name: str) -> str:
        """Prüft, ob ein Modul importiert werden kann."""
        try:
            __import__(module_name, fromlist=[""])
            return "ready"
        except ImportError:
            return "unavailable"


def create_app() -> FastAPI:
    """Erstellt die FastAPI-Anwendung.

    Setzt Middleware (CORS), mountet Routen und gibt die App zurück.
    """
    if not FASTAPI_AVAILABLE:
        raise RuntimeError(
            "FastAPI ist nicht verfügbar. Installieren Sie fastapi und pydantic."
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Lifecycle-Handler für die Anwendung."""
        logger.info("Trading Orchestra API starting up")
        yield
        logger.info("Trading Orchestra API shutting down")

    app = FastAPI(
        title="Trading Orchestra API",
        description="REST API für die Trading-Analysefähigkeiten",
        version=VERSION,
        lifespan=lifespan,
    )

    # CORS Middleware
    try:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    except ImportError:
        pass

    # Routes
    @app.post("/analyze", status_code=status.HTTP_200_OK)
    async def post_analyze(request: AnalyzeRequest) -> JSONResponse:
        """Akzeptiert eine Analyse-Anfrage und gibt das Ergebnis zurück."""
        result = await analyze_endpoint(request)
        return JSONResponse(content=result, status_code=status.HTTP_200_OK)

    @app.get("/status")
    async def get_status() -> JSONResponse:
        """Gibt Systemstatus mit Verfügbarkeit der Module zurück."""
        status_data = await status_endpoint()
        return JSONResponse(content=status_data, status_code=status.HTTP_200_OK)

    @app.get("/health")
    async def get_health() -> JSONResponse:
        """Einfacher Health-Check."""
        return JSONResponse(
            content={"status": "healthy", "timestamp": datetime.now(UTC).isoformat()},
            status_code=status.HTTP_200_OK,
        )

    return app
