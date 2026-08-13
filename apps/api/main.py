"""FastAPI REST API für die Trading-Orchestra Analyse-Fähigkeiten.

Dieses Modul stellt die FastAPI-Anwendung bereit, die die Analyse-Fähigkeiten
des Trading-Systems über eine REST-Schnittstelle exponiert. FastAPI und pydantic
sind optionale Abhängigkeiten — bei Nicht-Verfügbarkeit gibt die API einen
gracialen Fehler zurück.
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
    batch_analysis_endpoint,
    status_endpoint,
)
from apps.api.middleware import (
    create_auth_middleware,
    create_rate_limit_middleware,
)

# Live-Signal Router — optional, behind feature flag
try:
    from apps.api.routers import live_signal
except ImportError:
    live_signal = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

VERSION = "0.1.0"


if FASTAPI_AVAILABLE:

    class AnalyzeRequest(BaseModel):  # type: ignore[misc]
        """Anfrage-Modell für die Analyse-Endpunktes."""

        instrument: str = Field(..., min_length=1, max_length=50)  # type: ignore[call-overload]
        horizons: list[str] = Field(default_factory=lambda: ["1m", "5m", "15m"])  # type: ignore[call-overload]
        strategy: dict[str, object] = Field(default_factory=dict)  # type: ignore[call-overload]

    class StatusResponse(BaseModel):  # type: ignore[misc]
        """Antwort-Modell für den Status-Endpunkt."""

        version: str
        status: str
        uptime_seconds: float
        modules: dict[str, str]
        timestamp: str

    class BatchAnalyzeRequest(BaseModel):  # type: ignore[misc]
        """Anfrage-Modell für die Batch-Analyse."""

        instruments: list[str] = Field(  # type: ignore[call-overload]
            ..., min_length=1, max_length=20,
            description="Liste der zu analysierenden Instrumente.",
        )
        horizons: list[str] = Field(  # type: ignore[call-overload]
            default_factory=lambda: ["15m", "4h", "1d"],
            description="Zeitrahmen für die Analyse.",
        )
        strategy: dict[str, object] = Field(  # type: ignore[call-overload]
            default_factory=dict,
            description="Analyse-Strategie-Parameter.",
        )

    def _check_module_available(module_name: str) -> str:
        """Prüft, ob ein Modul importiert werden kann."""
        try:
            __import__(module_name, fromlist=[""])
            return "ready"
        except ImportError:
            return "unavailable"


def create_app() -> FastAPI:  # type: ignore[return-value, valid-type]
    """Erstellt die FastAPI-Anwendung.

    Setzt Middleware (CORS), mountet Routen und gibt die App zurück.
    """
    if not FASTAPI_AVAILABLE:
        raise RuntimeError(  # type: ignore[call-arg]
            "FastAPI ist nicht verfügbar. Installieren Sie fastapi und pydantic."
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # type: ignore[type-arg]
        """Lifecycle-Handler für die Anwendung."""
        logger.info("Trading Orchestra API starting up")
        yield
        logger.info("Trading Orchestra API shutting down")

    app = FastAPI(  # type: ignore[call-overload]
        title="Trading Orchestra API",
        description="REST API für die Trading-Analysefähigkeiten",
        version=VERSION,
        lifespan=lifespan,
    )

    # CORS Middleware — origins from env var (prod) or defaults (dev)
    try:
        import os

        from fastapi.middleware.cors import CORSMiddleware

        origins = os.environ.get(
            "API_CORS_ORIGINS",
            "http://localhost:3000,http://localhost:8080",
        ).split(",")

        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )
    except ImportError:
        pass

    # Rate Limiting & Auth Middleware
    rate_middleware = create_rate_limit_middleware()
    app.middleware("http")(rate_middleware)

    auth_middleware = create_auth_middleware()
    if auth_middleware is not None:
        app.middleware("http")(auth_middleware)

    # Routes
    @app.post("/analyze", status_code=status.HTTP_200_OK)  # type: ignore[assignment]
    async def post_analyze(request: AnalyzeRequest) -> JSONResponse:  # type: ignore[return-value]
        """Akzeptiert eine Analyse-Anfrage und gibt das Ergebnis zurück."""
        result = await analyze_endpoint(request)
        return JSONResponse(content=result, status_code=status.HTTP_200_OK)  # type: ignore[assignment, arg-type, call-arg]

    @app.post("/v1/analysis-runs/batch", status_code=status.HTTP_200_OK)  # type: ignore[assignment]
    async def post_batch_analyze(  # type: ignore[return-value]
        request: BatchAnalyzeRequest,
    ) -> JSONResponse:
        """Akzeptiert eine Batch-Analyse-Anfrage und gibt das Ergebnis zurück."""
        result = await batch_analysis_endpoint(request)
        return JSONResponse(content=result, status_code=status.HTTP_200_OK)  # type: ignore[assignment, arg-type, call-arg]

    @app.get("/status")
    async def get_status() -> JSONResponse:  # type: ignore[return-value]
        """Gibt Systemstatus mit Verfügbarkeit der Module zurück."""
        status_data = await status_endpoint()
        return JSONResponse(content=status_data, status_code=status.HTTP_200_OK)  # type: ignore[assignment, arg-type, var-annotated]  # type: ignore[assignment]

    @app.get("/health")
    async def get_health() -> JSONResponse:  # type: ignore[return-value]
        """Einfacher Health-Check."""
        return JSONResponse(  # type: ignore[possibly-unbound, call-overload]
            content={"status": "healthy", "timestamp": datetime.now(UTC).isoformat()},
            status_code=status.HTTP_200_OK,  # type: ignore[assignment, arg-type]
        )

    # Live-Signal Router — nur verfügbar wenn Router importiert werden konnte
    if live_signal is not None:
        app.include_router(live_signal.router)

    return app