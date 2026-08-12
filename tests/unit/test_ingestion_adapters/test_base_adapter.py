"""Unit tests for apps/ingestion/base_adapter.

Tests the abstract base class ``ExchangeAdapterBase`` including:
- ConnectionConfig dataclass
- ConnectionState enum
- ConnectionError / RateLimitError
- Lifecycle properties (is_connected, connection_state)
- Reconnect with exponential backoff
- Rate-limiting (sliding window)
- Heartbeat send / loop
- Event validation before publishing
- Async context manager
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from apps.ingestion.base_adapter import (
    ConnectionConfig,
    ConnectionError,  # noqa: A004
    ConnectionState,
    ExchangeAdapterBase,
    RateLimitError,
)
from packages.streaming.schemas import SourceMetadata

# ── concrete test subclass ────────────────────────────────────────


class _TestAdapter(ExchangeAdapterBase):
    """Concrete adapter for testing base-class logic."""

    def __init__(self, config: ConnectionConfig | None = None) -> None:
        if config is None:
            config = ConnectionConfig(api_key="key", api_secret="secret")
        super().__init__(config)
        self._connected = False
        self._disconnect_called = False
        self._subscribed_streams: list[str] = []
        self._published: list[dict[str, Any]] = []

    async def connect(self) -> None:
        self._connected = True
        self._state = ConnectionState.CONNECTED

    async def disconnect(self) -> None:
        self._disconnect_called = True
        self._connected = False
        self._state = ConnectionState.DISCONNECTED

    async def subscribe(self, streams: list[str]) -> None:
        self._subscribed_streams.extend(streams)

    def _publish_event(self, raw_event: dict[str, Any]) -> None:
        self._published.append(raw_event)


# ── helpers ───────────────────────────────────────────────────────


def _valid_candle_event() -> dict[str, Any]:
    return {
        "type": "candle",
        "instrument": "BTCUSDT",
        "venue": "BINANCE",
        "open": 100.0,
        "high": 110.0,
        "low": 95.0,
        "close": 105.0,
        "volume": 50.0,
        "open_time": "2024-01-01T00:00:00",
        "close_time": "2024-01-01T00:01:00",
    }


def _invalid_candle_event() -> dict[str, Any]:
    return {
        "type": "candle",
        "instrument": "BTCUSDT",
        "venue": "BINANCE",
        "open": 100.0,
        "high": 90.0,
        "low": 95.0,
        "close": 105.0,
        "volume": 50.0,
        "open_time": "2024-01-01T00:00:00",
        "close_time": "2024-01-01T00:01:00",
    }


def _missing_field_event() -> dict[str, Any]:
    return {
        "type": "candle",
        "venue": "BINANCE",
        "open": 100.0,
        "high": 110.0,
        "low": 95.0,
        "close": 105.0,
        "volume": 50.0,
        "open_time": "2024-01-01T00:00:00",
        "close_time": "2024-01-01T00:01:00",
    }


# ══════════════════════════════════════════════════════════════════
# ConnectionConfig
# ══════════════════════════════════════════════════════════════════


class TestConnectionConfig:
    def test_defaults(self) -> None:
        cfg = ConnectionConfig(api_key="k", api_secret="s")
        assert cfg.base_url == ""
        assert cfg.ws_url == ""
        assert cfg.reconnect_delay == 1.0
        assert cfg.max_reconnect_attempts == 10
        assert cfg.heartbeat_interval == 30.0
        assert cfg.rate_limit_per_second == 10

    def test_custom_values(self) -> None:
        cfg = ConnectionConfig(
            api_key="ak", api_secret="as",
            base_url="https://test", ws_url="wss://test",
            reconnect_delay=5.0, max_reconnect_attempts=3,
            heartbeat_interval=10.0, rate_limit_per_second=5,
        )
        assert cfg.reconnect_delay == 5.0
        assert cfg.max_reconnect_attempts == 3
        assert cfg.heartbeat_interval == 10.0
        assert cfg.rate_limit_per_second == 5

    def test_frozen(self) -> None:
        cfg = ConnectionConfig(api_key="k", api_secret="s")
        from dataclasses import FrozenInstanceError

        with pytest.raises(FrozenInstanceError):
            cfg.api_key = "new"


# ══════════════════════════════════════════════════════════════════
# ConnectionState
# ══════════════════════════════════════════════════════════════════


class TestConnectionState:
    def test_enum_values(self) -> None:
        assert ConnectionState.DISCONNECTED == "disconnected"
        assert ConnectionState.CONNECTING == "connecting"
        assert ConnectionState.CONNECTED == "connected"
        assert ConnectionState.RECONNECTING == "reconnecting"


# ══════════════════════════════════════════════════════════════════
# Exceptions
# ══════════════════════════════════════════════════════════════════


class TestExceptions:
    def test_connection_error_is_exception(self) -> None:
        with pytest.raises(ConnectionError):
            raise ConnectionError("fail")

    def test_rate_limit_error_is_exception(self) -> None:
        with pytest.raises(RateLimitError):
            raise RateLimitError("too slow")


# ══════════════════════════════════════════════════════════════════
# ExchangeAdapterBase — initialisation
# ══════════════════════════════════════════════════════════════════


class TestInit:
    def test_initial_state(self) -> None:
        adapter = _TestAdapter()
        assert adapter.is_connected is False
        assert adapter.connection_state == "disconnected"

    def test_config_stored(self) -> None:
        cfg = ConnectionConfig(
            api_key="mykey", api_secret="mysecret",
            base_url="https://example.com", ws_url="wss://example.com",
        )
        adapter = _TestAdapter(cfg)
        assert adapter.config is cfg
        assert adapter.config.api_key == "mykey"


# ══════════════════════════════════════════════════════════════════
# Properties
# ══════════════════════════════════════════════════════════════════


class TestProperties:
    def test_is_connected(self) -> None:
        adapter = _TestAdapter()
        assert adapter.is_connected is False

    def test_connection_state(self) -> None:
        adapter = _TestAdapter()
        assert adapter.connection_state == "disconnected"


# ══════════════════════════════════════════════════════════════════
# Abstract methods
# ══════════════════════════════════════════════════════════════════


class TestAbstractMethods:
    def test_cannot_instantiate_base(self) -> None:
        with pytest.raises(TypeError):
            ExchangeAdapterBase(ConnectionConfig(api_key="k", api_secret="s"))


# ══════════════════════════════════════════════════════════════════
# Reconnect
# ══════════════════════════════════════════════════════════════════


class TestReconnect:
    @pytest.mark.asyncio
    async def test_reconnect_succeeds(self) -> None:
        cfg = ConnectionConfig(
            api_key="k", api_secret="s",
            reconnect_delay=0.01, max_reconnect_attempts=3,
        )
        adapter = _TestAdapter(cfg)
        adapter._running = True
        adapter._connected = False
        adapter._state = ConnectionState.DISCONNECTED

        call_count = 0

        async def _flaky_connect() -> None:
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                adapter._connected = True
                adapter._state = ConnectionState.CONNECTED
            else:
                raise ConnectionError("fail")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(adapter, "connect", _flaky_connect)
            await asyncio.wait_for(adapter._reconnect(), timeout=10)

        assert adapter.is_connected is True
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_reconnect_fails_after_max(self) -> None:
        cfg = ConnectionConfig(
            api_key="k", api_secret="s",
            reconnect_delay=0.01, max_reconnect_attempts=2,
        )
        adapter = _TestAdapter(cfg)
        adapter._running = True

        async def _always_fail() -> None:
            raise ConnectionError("fail")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(adapter, "connect", _always_fail)
            with pytest.raises(ConnectionError, match="2 Versuchen"):
                await asyncio.wait_for(adapter._reconnect(), timeout=10)

        assert adapter.is_connected is False
        assert adapter.connection_state == "disconnected"

    @pytest.mark.asyncio
    async def test_reconnect_stops_when_connect_fails(self) -> None:
        cfg = ConnectionConfig(
            api_key="k", api_secret="s",
            reconnect_delay=0.001, max_reconnect_attempts=3,
        )
        adapter = _TestAdapter(cfg)
        adapter._running = True

        async def _stop_after_first() -> None:
            adapter._running = False

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(adapter, "connect", _stop_after_first)
            await asyncio.wait_for(adapter._reconnect(), timeout=5)


# ══════════════════════════════════════════════════════════════════
# Rate limiting
# ══════════════════════════════════════════════════════════════════


class TestRateLimiting:
    @pytest.mark.asyncio
    async def test_rate_limit_wait_allows(self) -> None:
        cfg = ConnectionConfig(
            api_key="k", api_secret="s", rate_limit_per_second=100,
        )
        adapter = _TestAdapter(cfg)
        await adapter._rate_limit_wait()

    @pytest.mark.asyncio
    async def test_rate_limit_wait_throttles(self) -> None:
        cfg = ConnectionConfig(
            api_key="k", api_secret="s", rate_limit_per_second=2,
        )
        adapter = _TestAdapter(cfg)
        await asyncio.gather(
            adapter._rate_limit_wait(),
            adapter._rate_limit_wait(),
            adapter._rate_limit_wait(),
        )

    @pytest.mark.asyncio
    async def test_rate_limit_is_concurrent_safe(self) -> None:
        cfg = ConnectionConfig(
            api_key="k", api_secret="s", rate_limit_per_second=50,
        )
        adapter = _TestAdapter(cfg)
        await asyncio.gather(
            *[adapter._rate_limit_wait() for _ in range(10)]
        )


# ══════════════════════════════════════════════════════════════════
# Heartbeat
# ══════════════════════════════════════════════════════════════════


class TestHeartbeat:
    @pytest.mark.asyncio
    async def test_send_heartbeat(self) -> None:
        adapter = _TestAdapter()
        await adapter._send_heartbeat()

    @pytest.mark.asyncio
    async def test_start_heartbeat(self) -> None:
        adapter = _TestAdapter()
        adapter._running = True
        adapter._connected = True
        await adapter.start_heartbeat()

        assert adapter._heartbeat_task is not None
        assert not adapter._heartbeat_task.done()

        await adapter.stop_heartbeat()
        assert adapter._heartbeat_task is None


# ══════════════════════════════════════════════════════════════════
# Validation & Publishing
# ══════════════════════════════════════════════════════════════════


class TestValidationPublishing:
    @pytest.mark.asyncio
    async def test_validate_and_publish_valid_event(self) -> None:
        adapter = _TestAdapter()
        event = _valid_candle_event()
        result = await adapter._validate_and_publish(event)
        assert result is True
        assert len(adapter._published) == 1

    @pytest.mark.asyncio
    async def test_validate_and_publish_invalid_high_low(self) -> None:
        adapter = _TestAdapter()
        event = _invalid_candle_event()
        result = await adapter._validate_and_publish(event)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_and_publish_missing_instrument(self) -> None:
        adapter = _TestAdapter()
        event = _missing_field_event()
        result = await adapter._validate_and_publish(event)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_and_publish_unknown_type(self) -> None:
        adapter = _TestAdapter()
        event = {"type": "unknown_type", "data": 1}
        result = await adapter._validate_and_publish(event)
        assert result is False


# ══════════════════════════════════════════════════════════════════
# Helpers — _next_sequence, _build_metadata
# ══════════════════════════════════════════════════════════════════


class TestHelpers:
    def test_next_sequence_increments(self) -> None:
        adapter = _TestAdapter()
        seq1 = adapter._next_sequence()
        seq2 = adapter._next_sequence()
        assert seq2 == seq1 + 1

    def test_build_metadata(self) -> None:
        adapter = _TestAdapter()
        meta = adapter._build_metadata("test_source", "test_venue")
        assert isinstance(meta, SourceMetadata)
        assert meta.source == "test_source"
        assert meta.venue == "test_venue"


# ══════════════════════════════════════════════════════════════════
# Health check
# ══════════════════════════════════════════════════════════════════


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_health_check(self) -> None:
        cfg = ConnectionConfig(
            api_key="ak", api_secret="as",
            base_url="https://example.com",
            ws_url="wss://example.com",
            rate_limit_per_second=20,
            heartbeat_interval=15.0,
        )
        adapter = _TestAdapter(cfg)
        result = await adapter.health_check()
        assert result["connected"] is False
        assert result["state"] == "disconnected"
        assert result["api_key_set"] is True
        assert result["base_url"] == "https://example.com"
        assert result["ws_url"] == "wss://example.com"
        assert result["rate_limit"] == 20
        assert result["heartbeat_interval"] == 15.0


# ══════════════════════════════════════════════════════════════════
# Async context manager
# ══════════════════════════════════════════════════════════════════


class TestAsyncContextManager:
    @pytest.mark.asyncio
    async def test_enter_exits(self) -> None:
        adapter = _TestAdapter()
        async with adapter:
            assert adapter.is_connected is True
            assert adapter._running is True

        assert adapter._disconnect_called is True
        assert adapter.is_connected is False
