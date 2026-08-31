"""Tests für die ClickHouse-Engine (packages/persistence/clickhouse/)."""

from __future__ import annotations

import base64
from typing import Any

import httpx
import pytest
from packages.persistence.clickhouse.engine import ClickHouseConfig, ClickHouseEngine


class _FakeResponse:
    """Simulierte httpx-Response (status 200)."""

    def __init__(self, status_code: int = 200, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


@pytest.fixture
def captured_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Monkeypatchet httpx.post und sammelt alle Aufrufe (URL + kwargs)."""
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        calls.append({"url": url, **kwargs})
        return _FakeResponse()

    monkeypatch.setattr(httpx, "post", fake_post)
    return calls


def _make_engine(**config_overrides: Any) -> ClickHouseEngine:
    defaults: dict[str, Any] = {"host": "localhost", "port": 8123, "database": "trading_events"}
    defaults.update(config_overrides)
    return ClickHouseEngine(ClickHouseConfig(**defaults))


class TestExecuteUrl:
    def test_execute_posts_to_database_path(self, captured_calls: list[dict[str, Any]]) -> None:
        """Query wird an http://host:8123/<database>/ gesendet."""
        engine = _make_engine(database="trading_events")
        engine._execute("SELECT 1")
        assert captured_calls[0]["url"] == "http://localhost:8123/trading_events/"

    def test_execute_uses_https_when_secure(self, captured_calls: list[dict[str, Any]]) -> None:
        engine = _make_engine(database="trading_events", secure=True)
        engine._execute("SELECT 1")
        assert captured_calls[0]["url"] == "https://localhost:8123/trading_events/"


class TestCreateTables:
    def test_create_database_before_table_ddl(self, captured_calls: list[dict[str, Any]]) -> None:
        """CREATE DATABASE läuft vor allen Tabellen-DDLs."""
        engine = _make_engine(database="trading_events")
        engine.create_tables()

        assert len(captured_calls) >= 5
        first_query = str(captured_calls[0]["content"])
        assert "CREATE DATABASE IF NOT EXISTS trading_events" in first_query

        for call in captured_calls[1:]:
            assert "CREATE TABLE IF NOT EXISTS" in str(call["content"])

        second_query = str(captured_calls[1]["content"])
        assert "CREATE TABLE IF NOT EXISTS candles" in second_query


class TestAuth:
    def test_auth_header_present_with_password(self, captured_calls: list[dict[str, Any]]) -> None:
        engine = _make_engine(database="trading_events", user="default", password="secret")
        engine._execute("SELECT 1")

        headers = captured_calls[0].get("headers") or {}
        expected = "Basic " + base64.b64encode(b"default:secret").decode("ascii")
        assert headers.get("Authorization") == expected

    def test_no_auth_header_without_password(self, captured_calls: list[dict[str, Any]]) -> None:
        engine = _make_engine(database="trading_events", password="")
        engine._execute("SELECT 1")

        headers = captured_calls[0].get("headers") or {}
        assert "Authorization" not in headers
