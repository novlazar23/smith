"""Tests für die apps/api-Modul — AnalyzeRequest, StatusResponse, create_app und Endpunkte."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

    from apps.api.main import AnalyzeRequest, StatusResponse


@pytest.fixture(autouse=True)
def _clear_api_module() -> Generator[None, None, None]:
    """Sorgt dafür, dass das api-Modul bei jedem Test neu importiert wird."""
    yield
    # Nach jedem Test das Modul aus dem Cache entfernen,
    # damit nachfolgende Tests mit sauberen Imports starten.
    for key in list(sys.modules.keys()):
        if key.startswith("apps.api"):
            del sys.modules[key]


@pytest.fixture
def fresh_app() -> Callable[[], Any]:
    """Importiert apps.api.main mit sauberem Zustand (FastAPI verfügbar)."""
    from apps.api.main import create_app

    return create_app


@pytest.fixture
def fresh_analyze_request() -> type[AnalyzeRequest]:
    """Liefert das AnalyzeRequest-Modell."""
    from apps.api.main import AnalyzeRequest

    return AnalyzeRequest


@pytest.fixture
def fresh_status_response() -> type[StatusResponse]:
    """Liefert das StatusResponse-Modell."""
    from apps.api.main import StatusResponse

    return StatusResponse


class TestAnalyzeRequest:
    """Testet das AnalyzeRequest Pydantic-Modell."""

    def test_valid_request(self, fresh_analyze_request: type[Any]) -> None:
        """Valid request created."""
        req = fresh_analyze_request(instrument="BTC/USD")
        assert req.instrument == "BTC/USD"

    def test_request_with_horizons(self, fresh_analyze_request: type[Any]) -> None:
        """Request mit benutzerdefinierten Horizons."""
        req = fresh_analyze_request(
            instrument="ETH/USD",
            horizons=["5m", "15m", "1h"],
        )
        assert req.horizons == ["5m", "15m", "1h"]

    def test_request_with_strategy(self, fresh_analyze_request: type[Any]) -> None:
        """Request mit Strategie-Parametern."""
        req = fresh_analyze_request(
            instrument="BTC/USD",
            strategy={"risk_level": "conservative"},
        )
        assert req.strategy == {"risk_level": "conservative"}

    def test_empty_instrument_raises(self, fresh_analyze_request: type[Any]) -> None:
        """ValueError on empty instrument."""
        with pytest.raises(Exception, match="string_too_short"):
            fresh_analyze_request(instrument="")

    def test_too_long_instrument_raises(self, fresh_analyze_request: type[Any]) -> None:
        """ValueError on instrument > 50 chars."""
        long_name = "A" * 51
        with pytest.raises(Exception, match="string_too_long"):
            fresh_analyze_request(instrument=long_name)

    def test_default_horizons(self, fresh_analyze_request: type[Any]) -> None:
        """Default horizons list."""
        req = fresh_analyze_request(instrument="BTC/USD")
        assert req.horizons == ["1m", "5m", "15m"]

    def test_strategy_default(self, fresh_analyze_request: type[Any]) -> None:
        """Empty strategy by default."""
        req = fresh_analyze_request(instrument="BTC/USD")
        assert req.strategy == {}


class TestStatusResponse:
    """Testet das StatusResponse Pydantic-Modell."""

    def test_all_fields(self, fresh_status_response: type[Any]) -> None:
        """Status response has all fields."""
        now_str = datetime.now(UTC).isoformat()
        resp = fresh_status_response(
            version="0.1.0",
            status="running",
            uptime_seconds=123.4,
            modules={"regime": "ready"},
            timestamp=now_str,
        )
        assert resp.version == "0.1.0"
        assert resp.status == "running"
        assert resp.uptime_seconds == 123.4
        assert resp.modules == {"regime": "ready"}
        assert resp.timestamp == now_str

    def test_empty_modules(self, fresh_status_response: type[Any]) -> None:
        """StatusResponse mit leerem modules-Dict."""
        now_str = datetime.now(UTC).isoformat()
        resp = fresh_status_response(
            version="0.1.0",
            status="idle",
            uptime_seconds=0.0,
            modules={},
            timestamp=now_str,
        )
        assert resp.modules == {}


class TestCreateApp:
    """Testet die create_app()-Funktion."""

    def test_returns_app(self, fresh_app: Callable[[], Any]) -> None:
        """create_app returns FastAPI instance."""
        app = fresh_app()
        assert app is not None
        assert hasattr(app, "routes")

    def test_routes_defined(self, fresh_app: Callable[[], Any]) -> None:
        """App has /analyze, /status, /health routes."""
        app = fresh_app()
        route_paths = {route.path for route in app.routes}
        assert "/analyze" in route_paths
        assert "/status" in route_paths
        assert "/health" in route_paths


class TestHealthEndpoint:
    """Testet den Health-Endpunkt."""

    def test_health_endpoint(self, fresh_app: Callable[[], Any]) -> None:
        """Returns healthy status."""
        from fastapi.testclient import TestClient

        app = fresh_app()
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "healthy"
            assert "timestamp" in data


class TestStatusEndpoint:
    """Testet den Status-Endpunkt."""

    def test_status_endpoint_includes_modules(self, fresh_app: Callable[[], Any]) -> None:
        """Status includes module dict."""
        from fastapi.testclient import TestClient

        app = fresh_app()
        with TestClient(app) as client:
            resp = client.get("/status")
            assert resp.status_code == 200
            data = resp.json()
            assert "modules" in data
            assert isinstance(data["modules"], dict)
            assert "version" in data
            assert "status" in data
            assert "uptime_seconds" in data
            assert "timestamp" in data

    def test_version_format(self, fresh_app: Callable[[], Any]) -> None:
        """Version is non-empty string."""
        from fastapi.testclient import TestClient

        app = fresh_app()
        with TestClient(app) as client:
            resp = client.get("/status")
            data = resp.json()
            assert isinstance(data["version"], str)
            assert len(data["version"]) > 0

    def test_uptime_calculation(self, fresh_app: Callable[[], Any]) -> None:
        """Uptime increases over time."""
        import time

        from fastapi.testclient import TestClient

        app = fresh_app()
        with TestClient(app) as client:
            resp1 = client.get("/status")
            data1 = resp1.json()
            time.sleep(0.05)
            resp2 = client.get("/status")
            data2 = resp2.json()
            # Die Zeitdifferenz zwischen zwei Aufrufen sollte messbar sein
            assert data2["uptime_seconds"] >= data1["uptime_seconds"]


class TestAPIEndpointProcessing:
    """Testet die Analyse-Endpunkt-Funktionalität."""

    def test_analyze_endpoint_processing(self, fresh_app: Callable[[], Any]) -> None:
        """Returns processing status."""
        from fastapi.testclient import TestClient

        app = fresh_app()
        with TestClient(app) as client:
            resp = client.post(
                "/analyze",
                json={
                    "instrument": "BTC/USD",
                    "horizons": ["1m", "5m"],
                    "strategy": {"type": "trend"},
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["instrument"] == "BTC/USD"
            assert data["status"] == "processing"

    def test_analyze_endpoint_includes_timestamp(self, fresh_app: Callable[[], Any]) -> None:
        """Timestamp in response."""
        from fastapi.testclient import TestClient

        app = fresh_app()
        with TestClient(app) as client:
            resp = client.post(
                "/analyze",
                json={"instrument": "ETH/USD"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "timestamp" in data
            assert isinstance(data["timestamp"], str)
            assert len(data["timestamp"]) > 0

    def test_analyze_includes_horizons(self, fresh_app: Callable[[], Any]) -> None:
        """Horizons appear in response."""
        from fastapi.testclient import TestClient

        app = fresh_app()
        with TestClient(app) as client:
            resp = client.post(
                "/analyze",
                json={
                    "instrument": "XAU/USD",
                    "horizons": ["15m", "1h"],
                },
            )
            data = resp.json()
            assert data["horizons"] == ["15m", "1h"]

    def test_analyze_includes_analysis_id(self, fresh_app: Callable[[], Any]) -> None:
        """Analysis ID in response."""
        from fastapi.testclient import TestClient

        app = fresh_app()
        with TestClient(app) as client:
            resp = client.post(
                "/analyze",
                json={"instrument": "BTC/USD"},
            )
            data = resp.json()
            assert "analysis_id" in data
            assert isinstance(data["analysis_id"], str)


class TestGracefulDegradation:
    """Testet das graceful Handling wenn FastAPI nicht verfügbar ist."""

    def test_api_no_fastapi_graceful(self) -> None:
        """Graceful handling when FastAPI unavailable."""
        # Block imports of fastapi and pydantic
        blocked = {"fastapi", "fastapi.*", "pydantic", "pydantic.*"}
        original_modules = {}
        for key in list(sys.modules.keys()):
            if any(key.startswith(b) for b in blocked):
                original_modules[key] = sys.modules.pop(key)

        # Remove api module from cache
        for key in list(sys.modules.keys()):
            if key.startswith("apps.api"):
                del sys.modules[key]

        try:
            with patch.dict(sys.modules, {"fastapi": None, "pydantic": None}):
                # Import the module — it should set FASTAPI_AVAILABLE=False
                mod = importlib.import_module("apps.api.main")
                assert hasattr(mod, "FASTAPI_AVAILABLE")
                assert mod.FASTAPI_AVAILABLE is False
        finally:
            # Restore original modules
            sys.modules.update(original_modules)
