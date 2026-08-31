"""Tests für den realen /status-Endpunkt und den Prometheus-/metrics-Endpunkt."""

from __future__ import annotations

import sys
from collections.abc import Callable, Generator
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _clear_api_module() -> Generator[None, None, None]:
    """Entfernt das api-Modul aus dem Cache, damit jeder Test sauber startet."""
    yield
    for key in list(sys.modules.keys()):
        if key.startswith("apps.api"):
            del sys.modules[key]


@pytest.fixture
def fresh_app() -> Callable[[], Any]:
    """Importiert apps.api.main mit sauberem Zustand (FastAPI verfügbar)."""
    from apps.api.main import create_app

    return create_app


def _make_client(fresh_app: Callable[[], Any]) -> Any:
    """Erstellt einen TestClient für eine frische App."""
    from fastapi.testclient import TestClient

    return TestClient(fresh_app())


def _healthy_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patcht alle Status-Probes auf gesunden Zustand."""
    import apps.api.endpoints as endpoints

    monkeypatch.setattr(
        endpoints, "_probe_postgres", lambda: {"connected": True, "latency_ms": 1.5}
    )
    monkeypatch.setattr(
        endpoints, "_probe_clickhouse", lambda: {"connected": True, "latency_ms": 2.5}
    )
    monkeypatch.setattr(endpoints, "_probe_redpanda", lambda: True)
    monkeypatch.setattr(endpoints, "_count_candles", lambda: 12345)
    monkeypatch.setattr(endpoints, "_count_news_events", lambda: 42)


def _sum_metric(text: str, name: str) -> float:
    """Summiert alle Label-Instanzen einer Metrik im Prometheus-Textformat."""
    total = 0.0
    prefix = f"{name}{{"
    for line in text.splitlines():
        if line.startswith(prefix):
            total += float(line.rsplit(" ", 1)[1])
    return total


class TestStatusEndpointProbes:
    """Testet den echten /status-Endpunkt mit gepatchten Probes."""

    def test_all_probes_healthy_running(self, fresh_app: Callable[[], Any], monkeypatch: pytest.MonkeyPatch) -> None:
        """Alle Probes gesund → status running, alte und neue Keys vorhanden."""
        _healthy_probes(monkeypatch)
        with _make_client(fresh_app) as client:
            resp = client.get("/status")
        assert resp.status_code == 200
        data = resp.json()

        # Alte Keys bleiben erhalten (abwärtskompatibel)
        for key in ("version", "status", "uptime_seconds", "modules", "timestamp"):
            assert key in data
        assert data["status"] == "running"
        assert data["uptime_seconds"] >= 0.0

        # Module werden gemeldet
        assert isinstance(data["modules"], dict)
        assert len(data["modules"]) > 0
        assert set(data["modules"].values()) <= {"ready", "unavailable"}

        # Datenbank-Probes
        assert data["database"]["postgres"] == {"connected": True, "latency_ms": 1.5}
        assert data["database"]["clickhouse"] == {"connected": True, "latency_ms": 2.5}

        # Streaming
        assert data["streaming"]["redpanda"] == {"connected": True}
        assert data["streaming"]["candles_1h"] == 12345
        assert data["streaming"]["news_events_total"] == 42

        # Feature-Flags: Live-Trading muss deaktiviert sein
        assert data["feature_flags"]["live_trading_enabled"] is False

    def test_pg_down_degraded(self, fresh_app: Callable[[], Any], monkeypatch: pytest.MonkeyPatch) -> None:
        """PostgreSQL down → status degraded, trotzdem HTTP 200."""
        _healthy_probes(monkeypatch)
        import apps.api.endpoints as endpoints

        monkeypatch.setattr(
            endpoints, "_probe_postgres", lambda: {"connected": False, "latency_ms": None}
        )
        with _make_client(fresh_app) as client:
            resp = client.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["database"]["postgres"] == {"connected": False, "latency_ms": None}

    def test_probe_raises_treated_as_down(self, fresh_app: Callable[[], Any], monkeypatch: pytest.MonkeyPatch) -> None:
        """Probe wirft unerwartete Exception → down, keine 500."""
        _healthy_probes(monkeypatch)
        import apps.api.endpoints as endpoints

        def boom() -> dict[str, Any]:
            raise RuntimeError("probe explosion")

        monkeypatch.setattr(endpoints, "_probe_clickhouse", boom)
        with _make_client(fresh_app) as client:
            resp = client.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["database"]["clickhouse"] == {"connected": False, "latency_ms": None}


class TestMetricsEndpoint:
    """Testet den Prometheus-/metrics-Endpunkt."""

    def test_metrics_endpoint_counter_increments(self, fresh_app: Callable[[], Any]) -> None:
        """GET /metrics → 200, http_requests_total vorhanden, Zähler steigt."""
        with _make_client(fresh_app) as client:
            r1 = client.get("/metrics")
            assert r1.status_code == 200
            assert "http_requests_total" in r1.text
            total1 = _sum_metric(r1.text, "http_requests_total")

            r2 = client.get("/metrics")
            assert r2.status_code == 200
            total2 = _sum_metric(r2.text, "http_requests_total")
        assert total2 > total1
