"""Tests für den Proxy-Router (apps.api.routers.proxy).

Die fünf internen Upstreams werden nicht real kontaktiert: ``client_factory``
wird per monkeypatch durch eine Fake-Fabrik ersetzt, deren Client die
Anfragen (URL, Methode, Body, Header) aufzeichnet und eine vorbereitete
httpx-Antwort liefert (oder die konfigurierte Ausnahme wirft).
"""

from __future__ import annotations

import asyncio
import gzip
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import pytest
from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Generator


@pytest.fixture(autouse=True)
def _clear_api_module() -> Generator[None, None, None]:
    """Entfernt apps.api-Module nach jedem Test aus dem sys.modules-Cache."""
    yield
    for key in list(sys.modules.keys()):
        if key.startswith("apps.api"):
            del sys.modules[key]


@pytest.fixture
def proxy_module() -> Any:
    """Importiert das Proxy-Router-Modul mit sauberem Zustand."""
    from apps.api.routers import proxy

    return proxy


@pytest.fixture
def fresh_app() -> Callable[[], Any]:
    """Liefert create_app mit sauberem Zustand (FastAPI verfügbar)."""
    from apps.api.main import create_app

    return create_app


class _StaticAsyncStream(httpx.AsyncByteStream):
    """Asynchroner Byte-Stream mit festem Inhalt für Test-Antworten."""

    def __init__(self, content: bytes) -> None:
        self._content = content

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._content

    async def aclose(self) -> None:
        return None


class FakeUpstreamClient:
    """Ersatz für httpx.AsyncClient — zeichnet die Anfragen auf, liefert die
    vorbereiteten Antworten (optional in Reihenfolge) oder wirft die
    konfigurierte Ausnahme."""

    def __init__(
        self,
        response: httpx.Response,
        exc: Exception | None = None,
        responses: list[httpx.Response] | None = None,
    ) -> None:
        self.response = response
        self.exc = exc
        self._queue: list[httpx.Response] = list(responses) if responses is not None else []
        self.request: httpx.Request | None = None
        self.requests: list[httpx.Request] = []
        self.body: bytes = b""
        self.closed: bool = False

    async def send(self, request: httpx.Request, **kwargs: object) -> httpx.Response:
        self.request = request
        self.requests.append(request)
        self.body = await request.aread()
        if self.exc is not None:
            raise self.exc
        return self._queue.pop(0) if self._queue else self.response

    async def aclose(self) -> None:
        self.closed = True


def _canned_response(
    status_code: int = 200,
    content: bytes = b"ok",
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """Erzeugt eine gestreamte httpx-Antwort mit festem Inhalt."""
    return httpx.Response(
        status_code=status_code,
        headers=headers or {},
        stream=_StaticAsyncStream(content),
    )


def _frame_denying_headers() -> dict[str, str]:
    """Antwort-Header, wie sie Grafana und MinIO für das Embedding senden."""
    return {
        "content-type": "text/html; charset=utf-8",
        "x-frame-options": "DENY",
        "content-security-policy": "frame-ancestors 'self'",
        "x-upstream": "kept",
        "set-cookie": "grafana_session=abc123; Path=/; HttpOnly",
    }


def _install_fake(
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    response: httpx.Response,
    exc: Exception | None = None,
    responses: list[httpx.Response] | None = None,
) -> FakeUpstreamClient:
    """Ersetzt client_factory durch eine Fake-Fabrik und liefert den Client."""
    fake = FakeUpstreamClient(response, exc=exc, responses=responses)
    monkeypatch.setattr(module, "client_factory", lambda: fake)
    return fake


class TestProxyUpstreamSelection:
    """Testet die Zuordnung von Proxy-Namen zu den internen Upstreams."""

    @pytest.mark.parametrize(
        ("name", "expected_url"),
        [
            ("grafana", "http://grafana:3000/proxy/grafana/"),
            ("minio", "http://minio:9001/"),
            ("prometheus", "http://prometheus:9090/"),
            ("alertmanager", "http://alertmanager:9093/"),
            ("mlflow", "http://mlflow:5000/"),
        ],
    )
    def test_upstream_selected_per_name(
        self,
        fresh_app: Callable[[], Any],
        proxy_module: Any,
        monkeypatch: pytest.MonkeyPatch,
        name: str,
        expected_url: str,
    ) -> None:
        """Jeder Name wird an den korrekten Upstream weitergeleitet.

        Grafana erhält den Subpfad zurück (GF_SERVER_SERVE_FROM_SUB_PATH),
        alle anderen Ziele die Upstream-Wurzel „/".
        """
        fake = _install_fake(monkeypatch, proxy_module, _canned_response())
        with TestClient(fresh_app()) as client:
            resp = client.get(f"/proxy/{name}/")
        assert resp.status_code == 200
        assert fake.request is not None
        assert str(fake.request.url) == expected_url

    def test_name_without_trailing_slash_maps_to_root(
        self,
        fresh_app: Callable[[], Any],
        proxy_module: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """/proxy/{name} ohne Slash wird an die Upstream-Wurzel „/" geleitet."""
        fake = _install_fake(monkeypatch, proxy_module, _canned_response())
        with TestClient(fresh_app()) as client:
            resp = client.get("/proxy/mlflow")
        assert resp.status_code == 200
        assert fake.request is not None
        assert str(fake.request.url) == "http://mlflow:5000/"


class TestProxyPathAndQuery:
    """Testet die Durchreichung von Pfad und Query-String an das Upstream."""

    def test_path_and_query_passthrough(
        self,
        fresh_app: Callable[[], Any],
        proxy_module: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Pfad und Query-String landen unverändert an der Upstream-URL."""
        fake = _install_fake(monkeypatch, proxy_module, _canned_response())
        with TestClient(fresh_app()) as client:
            resp = client.get("/proxy/prometheus/graph?h=1&panel=42")
        assert resp.status_code == 200
        assert fake.request is not None
        assert str(fake.request.url) == "http://prometheus:9090/graph?h=1&panel=42"

    def test_deep_path_passthrough(
        self,
        fresh_app: Callable[[], Any],
        proxy_module: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Mehrfach verschachtelte Pfade werden vollständig durchgereicht."""
        fake = _install_fake(monkeypatch, proxy_module, _canned_response())
        with TestClient(fresh_app()) as client:
            resp = client.get("/proxy/grafana/dashboards/db/myboard?orgId=1")
        assert resp.status_code == 200
        assert fake.request is not None
        expected = "http://grafana:3000/proxy/grafana/dashboards/db/myboard?orgId=1"
        assert str(fake.request.url) == expected


class TestProxyForwarding:
    """Testet die Weiterleitung von Methode, Body und Request-Headern."""

    def test_method_and_body_forwarded(
        self,
        fresh_app: Callable[[], Any],
        proxy_module: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """POST mit Body und Header wird an den Upstream weitergeleitet."""
        payload = b'{"query": "SELECT 1"}'
        fake = _install_fake(monkeypatch, proxy_module, _canned_response())
        with TestClient(fresh_app()) as client:
            resp = client.post(
                "/proxy/grafana/api/ds/query",
                content=payload,
                headers={"content-type": "application/json", "x-trace": "abc123"},
            )
        assert resp.status_code == 200
        assert fake.request is not None
        assert fake.request.method == "POST"
        assert fake.body == payload
        assert fake.request.headers["x-trace"] == "abc123"

    def test_host_header_set_to_upstream_netloc(
        self,
        fresh_app: Callable[[], Any],
        proxy_module: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Der Host-Header wird auf das Upstream-Netloc gesetzt."""
        fake = _install_fake(monkeypatch, proxy_module, _canned_response())
        with TestClient(fresh_app()) as client:
            client.get("/proxy/minio/buckets")
        assert fake.request is not None
        assert fake.request.headers["host"] == "minio:9001"

    def test_hop_by_hop_request_headers_dropped(
        self,
        fresh_app: Callable[[], Any],
        proxy_module: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Hop-by-hop-Header (connection, proxy-*) werden nicht weitergeleitet."""
        fake = _install_fake(monkeypatch, proxy_module, _canned_response())
        with TestClient(fresh_app()) as client:
            client.get(
                "/proxy/prometheus/api/v1/status",
                headers={"connection": "keep-alive", "proxy-authorization": "nope"},
            )
        assert fake.request is not None
        assert "connection" not in fake.request.headers
        assert "proxy-authorization" not in fake.request.headers
        assert fake.request.headers["host"] == "prometheus:9090"

    def test_upstream_status_code_forwarded(
        self,
        fresh_app: Callable[[], Any],
        proxy_module: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Der Upstream-Statuscode wird 1:1 an den Client zurückgegeben."""
        fake = _install_fake(monkeypatch, proxy_module, _canned_response(503, b"down"))
        with TestClient(fresh_app()) as client:
            resp = client.get("/proxy/alertmanager/")
        assert fake.request is not None
        assert resp.status_code == 503
        assert resp.text == "down"


class TestProxyHeaderStripping:
    """Testet die Entfernung ziel-spezifischer Response-Header."""

    def test_grafana_strips_frame_headers(
        self,
        fresh_app: Callable[[], Any],
        proxy_module: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Grafana-Antworten verlieren X-Frame-Options und CSP, behalten den Rest."""
        fake = _install_fake(
            monkeypatch, proxy_module, _canned_response(200, b"<html></html>", _frame_denying_headers())
        )
        with TestClient(fresh_app()) as client:
            resp = client.get("/proxy/grafana/")
        assert fake.request is not None
        assert resp.status_code == 200
        assert resp.text == "<html></html>"
        assert "x-frame-options" not in resp.headers
        assert "content-security-policy" not in resp.headers
        assert resp.headers["x-upstream"] == "kept"
        assert resp.headers["set-cookie"] == "grafana_session=abc123; Path=/; HttpOnly"
        assert resp.headers["content-type"] == "text/html; charset=utf-8"

    def test_minio_strips_frame_headers(
        self,
        fresh_app: Callable[[], Any],
        proxy_module: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """MinIO-Antworten verlieren X-Frame-Options und CSP."""
        fake = _install_fake(
            monkeypatch, proxy_module, _canned_response(200, b"<html></html>", _frame_denying_headers())
        )
        with TestClient(fresh_app()) as client:
            resp = client.get("/proxy/minio/")
        assert fake.request is not None
        assert resp.status_code == 200
        assert "x-frame-options" not in resp.headers
        assert "content-security-policy" not in resp.headers
        assert resp.headers["set-cookie"] == "grafana_session=abc123; Path=/; HttpOnly"

    def test_prometheus_keeps_frame_headers(
        self,
        fresh_app: Callable[[], Any],
        proxy_module: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Ohne Strip-Liste bleiben X-Frame-Options und CSP erhalten."""
        fake = _install_fake(
            monkeypatch, proxy_module, _canned_response(200, b"ok", _frame_denying_headers())
        )
        with TestClient(fresh_app()) as client:
            resp = client.get("/proxy/prometheus/")
        assert fake.request is not None
        assert resp.status_code == 200
        assert resp.headers["x-frame-options"] == "DENY"
        assert resp.headers["content-security-policy"] == "frame-ancestors 'self'"


class TestProxyEncodingHeaders:
    """httpx dekomprimiert transparent — Encoding-Header dürfen den Client nicht erreichen."""

    def test_content_encoding_and_length_stripped(
        self,
        fresh_app: Callable[[], Any],
        proxy_module: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """content-encoding/content-length des Upstream werden entfernt, der
        dekomprimierte Body kommt unverändert an (sonst
        net::ERR_CONTENT_LENGTH_MISMATCH / „Failed to fetch" im Browser)."""
        payload = b'{"status":"success"}'
        _install_fake(
            monkeypatch,
            proxy_module,
            _canned_response(
                200,
                gzip.compress(payload),
                {"content-type": "application/json", "content-encoding": "gzip", "content-length": "15"},
            ),
        )
        with TestClient(fresh_app()) as client:
            resp = client.get("/proxy/prometheus/api/v1/query")
        assert resp.status_code == 200
        assert resp.text == payload.decode("utf-8")
        assert "content-encoding" not in resp.headers
        assert resp.headers.get("content-length") != "15"


class TestProxyBaseTagRewrite:
    """SPAs mit <base href="/"> müssen auf den Proxy-Subpfad umgeschrieben werden."""

    def test_minio_base_tag_rewritten(
        self,
        fresh_app: Callable[[], Any],
        proxy_module: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """<base href="/"> wird auf den MinIO-Subpfad gesetzt, der Rest bleibt."""
        html = (
            b'<!doctype html><html><head><base href="/"/>'
            b'<script src="./static/js/main.js"></script></head><body></body></html>'
        )
        _install_fake(
            monkeypatch,
            proxy_module,
            _canned_response(200, html, {"content-type": "text/html; charset=utf-8"}),
        )
        with TestClient(fresh_app()) as client:
            resp = client.get("/proxy/minio/")
        assert resp.status_code == 200
        assert b'<base href="/proxy/minio/">' in resp.content
        assert b'<base href="/">' not in resp.content
        assert b'./static/js/main.js' in resp.content

    def test_html_without_base_tag_unchanged(
        self,
        fresh_app: Callable[[], Any],
        proxy_module: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """HTML ohne Wurzel-Base-Tag wird bytegenau durchgereicht."""
        html = b"<!doctype html><html><head></head><body>ok</body></html>"
        _install_fake(
            monkeypatch,
            proxy_module,
            _canned_response(200, html, {"content-type": "text/html; charset=utf-8"}),
        )
        with TestClient(fresh_app()) as client:
            resp = client.get("/proxy/mlflow/")
        assert resp.status_code == 200
        assert resp.content == html


class TestProxyUnknownTarget:
    """Testet das Verhalten bei unbekannten Ziel-Namen."""

    def test_unknown_name_returns_404(self, fresh_app: Callable[[], Any]) -> None:
        """Unbekannter Name antwortet mit HTTP 404 (kein Upstream-Kontakt)."""
        with TestClient(fresh_app()) as client:
            resp = client.get("/proxy/burp/")
        assert resp.status_code == 404


class TestProxyUpstreamError:
    """Testet das Verhalten bei unerreichbarem Upstream."""

    @pytest.mark.parametrize(
        "exc",
        [httpx.ConnectError("refused"), httpx.ConnectTimeout("slow")],
    )
    def test_upstream_error_returns_502(
        self,
        fresh_app: Callable[[], Any],
        proxy_module: Any,
        monkeypatch: pytest.MonkeyPatch,
        exc: Exception,
    ) -> None:
        """Connection-/Timeout-Fehler am Upstream liefern HTTP 502 + JSON-Body."""
        fake = _install_fake(monkeypatch, proxy_module, _canned_response(), exc=exc)
        with TestClient(fresh_app()) as client:
            resp = client.get("/proxy/grafana/")
        assert resp.status_code == 502
        assert resp.json() == {"error": "upstream unavailable", "target": "grafana"}
        assert fake.closed is True


class TestProxyRateLimitExemption:
    """/proxy darf nicht am Rate-Limit scheitern (ein Iframe-Load > 60 Requests)."""

    def test_proxy_bypasses_rate_limit_while_api_is_limited(
        self,
        fresh_app: Callable[[], Any],
        proxy_module: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Mit vollem Rate-Limit-Fenster gehen /proxy-Requests durch, /status nicht."""
        import time
        from collections import defaultdict

        import apps.api.middleware as middleware_module

        full_store: dict[str, list[float]] = defaultdict(list)
        full_store["testclient"] = [time.time() - 0.5] * 60
        monkeypatch.setattr(middleware_module, "_rate_store", full_store)

        _install_fake(monkeypatch, proxy_module, _canned_response())
        with TestClient(fresh_app()) as client:
            assert client.get("/proxy/prometheus/").status_code == 200
            assert client.get("/status").status_code == 429


class TestProxyMinioAutoLogin:
    """MinIO-Console: Auto-Login, damit der Storage-Tab ohne Formular lädt."""

    COOKIE = "token=abc123; Path=/proxy/minio; HttpOnly; SameSite=Lax; Max-Age=43200"

    @staticmethod
    def _patch_login(
        monkeypatch: pytest.MonkeyPatch, module: Any, logins: list[int]
    ) -> None:
        async def fake_login(client: Any, base_url: str) -> str | None:
            logins.append(1)
            return TestProxyMinioAutoLogin.COOKIE

        monkeypatch.setattr(module, "_minio_login", fake_login)

    def test_login_cookie_injected_into_request_and_response(
        self,
        fresh_app: Callable[[], Any],
        proxy_module: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Ohne Browser-Token: Login-Cookie geht an Upstream UND ans Browser-Endgerät."""
        self._patch_login(monkeypatch, proxy_module, [])
        fake = _install_fake(monkeypatch, proxy_module, _canned_response())
        with TestClient(fresh_app()) as client:
            resp = client.get("/proxy/minio/")
        assert resp.status_code == 200
        assert fake.request is not None
        assert fake.request.headers.get("cookie") == TestProxyMinioAutoLogin.COOKIE
        assert resp.headers["set-cookie"] == TestProxyMinioAutoLogin.COOKIE

    def test_no_login_when_browser_has_token(
        self,
        fresh_app: Callable[[], Any],
        proxy_module: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Existierendes Browser-Token: kein Login, Cookie wird weitergeleitet."""
        logins: list[int] = []
        self._patch_login(monkeypatch, proxy_module, logins)
        fake = _install_fake(monkeypatch, proxy_module, _canned_response())
        with TestClient(fresh_app()) as client:
            client.get("/proxy/minio/", headers={"cookie": "token=existing"})
        assert logins == []
        assert fake.request is not None
        assert fake.request.headers.get("cookie") == "token=existing"

    def test_stale_session_triggers_relogin_and_retry(
        self,
        fresh_app: Callable[[], Any],
        proxy_module: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """403 vom Upstream: frische Anmeldung und einmalige Wiederholung."""
        logins: list[int] = []
        self._patch_login(monkeypatch, proxy_module, logins)
        fake = _install_fake(
            monkeypatch,
            proxy_module,
            _canned_response(),
            responses=[_canned_response(403, b"forbidden"), _canned_response(200, b"ok")],
        )
        with TestClient(fresh_app()) as client:
            resp = client.get("/proxy/minio/buckets")
        assert resp.status_code == 200
        assert resp.text == "ok"
        assert len(logins) == 2
        assert len(fake.requests) == 2
        assert fake.requests[1].headers.get("cookie") == TestProxyMinioAutoLogin.COOKIE

    def test_minio_login_rewrites_cookie_path(
        self, proxy_module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """/api/v1/login mit S3-Credentials; Cookie-Path wird auf den Subpfad gesetzt."""
        secret_file = tmp_path / "minio_password.txt"
        secret_file.write_text("supersecret\n")
        monkeypatch.setattr(proxy_module, "_MINIO_SECRET_KEY_FILE", str(secret_file))
        fake = FakeUpstreamClient(
            _canned_response(
                204,
                b"",
                {"set-cookie": "token=TOK; Path=/; Expires=never; Max-Age=43200; HttpOnly"},
            )
        )
        result = asyncio.run(proxy_module._minio_login(fake, "http://minio:9001"))
        assert result == "token=TOK; Path=/proxy/minio; HttpOnly; SameSite=Lax; Max-Age=43200"
        assert fake.request is not None
        assert "api/v1/login" in str(fake.request.url)
        assert fake.request.headers["content-type"] == "application/json"

    def test_minio_login_prefers_env_key(
        self, proxy_module: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Env-Variablen (im Container vom Entrypoint exportiert) haben Vorrang."""
        monkeypatch.setenv("MINIO_SECRET_KEY", "envsecret")
        fake = FakeUpstreamClient(
            _canned_response(204, b"", {"set-cookie": "token=TOK; Path=/; HttpOnly"})
        )
        result = asyncio.run(proxy_module._minio_login(fake, "http://minio:9001"))
        assert result == "token=TOK; Path=/proxy/minio; HttpOnly; SameSite=Lax; Max-Age=43200"

    def test_minio_login_without_secret_skipped(
        self, proxy_module: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ohne Secret (Env und Datei leer): kein Login-Versuch, None."""
        monkeypatch.delenv("MINIO_SECRET_KEY", raising=False)
        monkeypatch.setattr(proxy_module, "_MINIO_SECRET_KEY_FILE", "")
        result = asyncio.run(
            proxy_module._minio_login(FakeUpstreamClient(_canned_response()), "http://minio:9001")
        )
        assert result is None


class TestProxyMount:
    """Testet, dass der Proxy-Router in der App eingebunden ist."""

    def test_router_routes_defined(self, proxy_module: Any) -> None:
        """Beide Pfadmuster existieren im Router."""
        paths = {getattr(route, "path", "") for route in proxy_module.router.routes}
        assert "/proxy/{name}" in paths
        assert "/proxy/{name}/{path:path}" in paths

    def test_router_mounted_on_app(
        self,
        fresh_app: Callable[[], Any],
        proxy_module: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Eine /proxy-Anfrage über die App erreicht den Proxy (Fake kontaktiert)."""
        fake = _install_fake(monkeypatch, proxy_module, _canned_response())
        with TestClient(fresh_app()) as client:
            resp = client.get("/proxy/prometheus/graph?h=1")
        assert resp.status_code == 200
        assert fake.request is not None
        assert str(fake.request.url) == "http://prometheus:9090/graph?h=1"
