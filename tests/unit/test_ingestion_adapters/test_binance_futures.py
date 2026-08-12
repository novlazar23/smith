"""Tests for Binance Futures adapter (FuturesAdapter).

Uses only mocked HTTP/WebSocket calls — no network traffic.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from apps.ingestion.base_adapter import ConnectionState
from apps.ingestion.binance_futures import FuturesAdapter
from packages.domain.market_data.derivatives import (
    FundingRate,
    OpenInterest,
)
from packages.streaming.schemas import MarketEvent

# ── helpers ───────────────────────────────────────────────────────


def _make_config(**overrides: Any) -> Any:
    """Return a mock ConnectionConfig matching base_adapter expectations."""
    cfg = MagicMock()
    cfg.api_key = ""
    cfg.api_secret = ""
    cfg.base_url = ""
    cfg.ws_url = ""
    cfg.reconnect_delay = 1.0
    cfg.max_reconnect_attempts = 10
    cfg.heartbeat_interval = 30.0
    cfg.rate_limit_per_second = 10
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _valid_funding_rate_event() -> dict[str, Any]:
    """Return a valid fundingRate WebSocket message."""
    return {
        "e": "fundingRate",
        "E": 1609459200000,
        "s": "BTCUSDT",
        "r": "0.00010000",
        "p": "50000.00",
        "T": 1609459200000,
    }


def _valid_open_interest_event() -> dict[str, Any]:
    """Return a valid openInterest WebSocket message."""
    return {
        "oi": "1234567.89",
        "s": "BTCUSDT",
        "T": 1609459200000,
    }


def _valid_liquidation_event() -> dict[str, Any]:
    """Return a valid forceOrder (liquidation) WebSocket message."""
    return {
        "e": "forceOrder",
        "E": 1609459200000,
        "o": {
            "s": "BTCUSDT",
            "p": "48000.00",
            "q": "1.500",
            "S": "SELL",
            "T": 1609459200000,
        },
    }


def _valid_liquidation_buy_event() -> dict[str, Any]:
    """Return a valid forceOrder (long liquidation) WebSocket message."""
    return {
        "e": "forceOrder",
        "E": 1609459200000,
        "o": {
            "s": "ETHUSDT",
            "p": "3200.00",
            "q": "50.0",
            "S": "BUY",
            "T": 1609459200000,
        },
    }


def _rest_funding_rate_response() -> dict[str, Any]:
    """Return a mock REST response for /fapi/v1/fundingRate."""
    return {
        "symbol": "BTCUSDT",
        "fundingRate": "0.00010000",
        "markPrice": "50000.50",
        "nextFundingTime": 1609462800000,
        "time": 1609459200000,
    }


def _rest_open_interest_response() -> dict[str, Any]:
    """Return a mock REST response for /fapi/v1/openInterest."""
    return {
        "symbol": "BTCUSDT",
        "openInterest": "1234567.89000000",
        "time": 1609459200000,
    }


# ── __init__ ──────────────────────────────────────────────────────


class TestFuturesAdapterInit:

    def test_init_sets_base_url(self) -> None:
        adapter = FuturesAdapter(_make_config())
        assert adapter.BASE_URL == "https://fapi.binance.com"

    def test_init_sets_ws_url(self) -> None:
        adapter = FuturesAdapter(_make_config())
        assert adapter.WS_URL == "wss://fstream.binance.com:443/ws"

    def test_init_initialises_state(self) -> None:
        adapter = FuturesAdapter(_make_config())
        assert adapter._streams == {}
        assert adapter._event_handler is None

    def test_init_inherits_from_base(self) -> None:
        from apps.ingestion.base_adapter import ExchangeAdapterBase
        adapter = FuturesAdapter(_make_config())
        assert isinstance(adapter, ExchangeAdapterBase)


# ── connect / disconnect ─────────────────────────────────────────


class TestFuturesAdapterLifecycle:

    @pytest.mark.asyncio
    async def test_connect_already_connected(self) -> None:
        adapter = FuturesAdapter(_make_config())
        adapter._state = ConnectionState.CONNECTED
        with patch("aiohttp.ClientSession") as mock_session:
            await adapter.connect()
            mock_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_disconnect_already_disconnected(self) -> None:
        adapter = FuturesAdapter(_make_config())
        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect_clears_state(self) -> None:
        adapter = FuturesAdapter(_make_config())
        mock_ws = AsyncMock()
        mock_ws.closed = False
        adapter._ws_session = mock_ws
        adapter._http_session = AsyncMock()
        adapter._state = ConnectionState.CONNECTED
        adapter._streams = {"key": "val"}

        await adapter.disconnect()

        assert adapter._state == ConnectionState.DISCONNECTED
        assert adapter._ws_session is None
        assert adapter._http_session is None
        assert adapter._streams == {}

    @pytest.mark.asyncio
    async def test_connect_raises_runtime_on_disconnect_then_subscribe(self) -> None:
        adapter = FuturesAdapter(_make_config())
        with pytest.raises(RuntimeError, match="Not connected"):
            await adapter.subscribe(["btcusdt_perpetual@funding"])


# ── subscribe ─────────────────────────────────────────────────────


class TestFuturesAdapterSubscribe:

    @pytest.mark.asyncio
    async def test_subscribe_sends_ws_message(self) -> None:
        adapter = FuturesAdapter(_make_config())
        mock_ws = AsyncMock()
        mock_ws.closed = False
        mock_ws.send_str = AsyncMock()
        adapter._ws_session = mock_ws
        adapter._state = ConnectionState.CONNECTED

        await adapter.subscribe(["btcusdt_perpetual@funding"])

        mock_ws.send_str.assert_called_once()
        call_arg = mock_ws.send_str.call_args[0][0]
        parsed = json.loads(call_arg)
        assert parsed["method"] == "SUBSCRIBE"
        assert parsed["params"] == ["btcusdt_perpetual@funding"]
        assert parsed["id"] == 1

    @pytest.mark.asyncio
    async def test_subscribe_multi_streams(self) -> None:
        adapter = FuturesAdapter(_make_config())
        mock_ws = AsyncMock()
        mock_ws.closed = False
        mock_ws.send_str = AsyncMock()
        adapter._ws_session = mock_ws
        adapter._state = ConnectionState.CONNECTED

        streams = ["btcusdt_perpetual@funding", "btcusdt_perpetual@openInterest"]
        await adapter.subscribe(streams)

        call_arg = mock_ws.send_str.call_args[0][0]
        parsed = json.loads(call_arg)
        assert parsed["params"] == streams
        assert parsed["id"] == 1

    @pytest.mark.asyncio
    async def test_subscribe_empty_list(self) -> None:
        adapter = FuturesAdapter(_make_config())
        mock_ws = AsyncMock()
        mock_ws.closed = False
        mock_ws.send_str = AsyncMock()
        adapter._ws_session = mock_ws
        adapter._state = ConnectionState.CONNECTED

        await adapter.subscribe([])
        mock_ws.send_str.assert_not_called()


# ── fetch_funding_rate ────────────────────────────────────────────


class TestFuturesAdapterFetchFundingRate:

    @pytest.mark.asyncio
    async def test_fetch_funding_rate(self) -> None:
        adapter = FuturesAdapter(_make_config())
        raw = _rest_funding_rate_response()

        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = AsyncMock(return_value=raw)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.close = AsyncMock()
        adapter._http_session = mock_session

        result = await adapter.fetch_funding_rate("BTCUSDT")

        assert isinstance(result, FundingRate)
        assert result.instrument == "BTCUSDT"
        assert result.venue == "BINANCE_FUTURES"
        assert result.funding_rate == 0.0001
        assert result.mark_price == 50000.5
        assert result.next_funding_time == datetime(
            2021, 1, 1, 1, 0, tzinfo=UTC
        )

    @pytest.mark.asyncio
    async def test_fetch_funding_rate_sends_correct_url(self) -> None:
        adapter = FuturesAdapter(_make_config())

        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = AsyncMock(return_value=_rest_funding_rate_response())
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.close = AsyncMock()
        adapter._http_session = mock_session

        await adapter.fetch_funding_rate("ETHUSDT")

        call_kwargs = mock_session.get.call_args
        assert "fundingRate" in call_kwargs[0][0]
        params = call_kwargs[1]["params"]
        assert params["symbol"] == "ETHUSDT"

    @pytest.mark.asyncio
    async def test_fetch_funding_rate_uppercases_symbol(self) -> None:
        adapter = FuturesAdapter(_make_config())

        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = AsyncMock(return_value=_rest_funding_rate_response())
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.close = AsyncMock()
        adapter._http_session = mock_session

        await adapter.fetch_funding_rate("btcusdt")

        params = mock_session.get.call_args[1]["params"]
        assert params["symbol"] == "BTCUSDT"
        assert adapter.fetch_funding_rate.__code__.co_varnames  # sanity


# ── fetch_open_interest ───────────────────────────────────────────


class TestFuturesAdapterFetchOpenInterest:

    @pytest.mark.asyncio
    async def test_fetch_open_interest(self) -> None:
        adapter = FuturesAdapter(_make_config())
        raw = _rest_open_interest_response()

        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = AsyncMock(return_value=raw)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.close = AsyncMock()
        adapter._http_session = mock_session

        result = await adapter.fetch_open_interest("BTCUSDT")

        assert isinstance(result, OpenInterest)
        assert result.instrument == "BTCUSDT"
        assert result.venue == "BINANCE_FUTURES"
        assert result.open_interest == 1234567.89
        assert result.event_time == datetime(2021, 1, 1, 0, 0, tzinfo=UTC)

    @pytest.mark.asyncio
    async def test_fetch_open_interest_sends_correct_url(self) -> None:
        adapter = FuturesAdapter(_make_config())

        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = AsyncMock(return_value=_rest_open_interest_response())
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.close = AsyncMock()
        adapter._http_session = mock_session

        await adapter.fetch_open_interest("ETHUSDT")

        call_kwargs = mock_session.get.call_args
        assert "openInterest" in call_kwargs[0][0]
        params = call_kwargs[1]["params"]
        assert params["symbol"] == "ETHUSDT"


# ── fetch_recent_liquidations ─────────────────────────────────────


class TestFuturesAdapterFetchLiquidations:

    @pytest.mark.asyncio
    async def test_fetch_recent_liquidations_default(self) -> None:
        adapter = FuturesAdapter(_make_config())

        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = AsyncMock(return_value=[])
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.close = AsyncMock()
        adapter._http_session = mock_session

        result = await adapter.fetch_recent_liquidations()

        assert isinstance(result, list)
        # Liquidation data is WebSocket-only for Binance Futures REST API
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_fetch_recent_liquidations_with_symbol(self) -> None:
        adapter = FuturesAdapter(_make_config())

        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = AsyncMock(return_value=[])
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.close = AsyncMock()
        adapter._http_session = mock_session

        result = await adapter.fetch_recent_liquidations(symbol="BTCUSDT", limit=50)

        assert isinstance(result, list)
        assert len(result) == 0


# ── _parse_funding_rate ───────────────────────────────────────────


class TestFuturesAdapterParseFundingRate:

    @pytest.mark.asyncio
    async def test_parse_funding_rate_valid(self) -> None:
        adapter = FuturesAdapter(_make_config())
        event = await adapter._parse_funding_rate(_valid_funding_rate_event())

        assert isinstance(event, MarketEvent)
        assert event.event_type == "funding_rate"
        assert event.instrument == "BTCUSDT"
        assert event.payload["funding_rate"] == 0.0001
        assert event.payload["mark_price"] == 50000.0
        assert isinstance(event.payload["event_time"], datetime)

    @pytest.mark.asyncio
    async def test_parse_funding_rate_default_values(self) -> None:
        adapter = FuturesAdapter(_make_config())
        minimal: dict[str, Any] = {"e": "fundingRate", "s": "ETHUSDT"}
        event = await adapter._parse_funding_rate(minimal)

        assert event.event_type == "funding_rate"
        assert event.payload["funding_rate"] == 0.0
        assert event.payload["mark_price"] == 0.0


# ── _parse_open_interest ──────────────────────────────────────────


class TestFuturesAdapterParseOpenInterest:

    @pytest.mark.asyncio
    async def test_parse_open_interest_valid(self) -> None:
        adapter = FuturesAdapter(_make_config())
        event = await adapter._parse_open_interest(_valid_open_interest_event())

        assert isinstance(event, MarketEvent)
        assert event.event_type == "open_interest"
        assert event.instrument == "BTCUSDT"
        assert event.payload["open_interest"] == 1234567.89
        assert isinstance(event.payload["event_time"], datetime)


# ── _parse_liquidation ────────────────────────────────────────────


class TestFuturesAdapterParseLiquidation:

    @pytest.mark.asyncio
    async def test_parse_liquidation_valid(self) -> None:
        adapter = FuturesAdapter(_make_config())
        event = await adapter._parse_liquidation(_valid_liquidation_event())

        assert isinstance(event, MarketEvent)
        assert event.event_type == "liquidation"
        assert event.instrument == "BTCUSDT"
        assert event.payload["price"] == 48000.0
        assert event.payload["quantity"] == 1.5
        assert event.payload["value"] == 72000.0
        assert event.payload["side"] == "SELL"

    @pytest.mark.asyncio
    async def test_parse_liquidation_buy_side(self) -> None:
        adapter = FuturesAdapter(_make_config())
        event = await adapter._parse_liquidation(_valid_liquidation_buy_event())

        assert event.event_type == "liquidation"
        assert event.payload["side"] == "BUY"
        assert event.payload["price"] == 3200.0

    @pytest.mark.asyncio
    async def test_parse_liquidation_nested_format(self) -> None:
        adapter = FuturesAdapter(_make_config())
        # Test with flat (non-nested) forceOrder format
        flat_event: dict[str, Any] = {
            "e": "forceOrder",
            "E": 1609459200000,
            "s": "BTCUSDT",
            "p": "45000.00",
            "q": "2.0",
            "S": "BUY",
            "T": 1609459200000,
        }
        event = await adapter._parse_liquidation(flat_event)

        assert event.event_type == "liquidation"
        assert event.payload["price"] == 45000.0
        assert event.payload["quantity"] == 2.0


# ── _handle_message ───────────────────────────────────────────────


class TestFuturesAdapterHandleMessage:

    @pytest.mark.asyncio
    async def test_handle_funding_rate_valid(self) -> None:
        adapter = FuturesAdapter(_make_config())
        result = await adapter._handle_message(
            json.dumps(_valid_funding_rate_event())
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_open_interest_valid(self) -> None:
        adapter = FuturesAdapter(_make_config())
        result = await adapter._handle_message(
            json.dumps(_valid_open_interest_event())
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_liquidation_valid(self) -> None:
        adapter = FuturesAdapter(_make_config())
        result = await adapter._handle_message(
            json.dumps(_valid_liquidation_event())
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_empty_message(self) -> None:
        adapter = FuturesAdapter(_make_config())
        result = await adapter._handle_message("")
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_invalid_json(self) -> None:
        adapter = FuturesAdapter(_make_config())
        result = await adapter._handle_message("not valid json {{{")
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_unknown_event_type(self) -> None:
        adapter = FuturesAdapter(_make_config())
        result = await adapter._handle_message(
            json.dumps({"e": "unknownType"})
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_kline(self) -> None:
        adapter = FuturesAdapter(_make_config())
        kline_event = {
            "e": "kline",
            "E": 1609459200000,
            "s": "BTCUSDT",
            "k": {
                "t": 1609459200000,
                "T": 1609459260000,
                "i": "1m",
                "o": "50000.00",
                "h": "50100.00",
                "l": "49900.00",
                "c": "50050.00",
                "v": "10.5",
                "n": 150,
            },
        }
        result = await adapter._handle_message(json.dumps(kline_event))
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_trade(self) -> None:
        adapter = FuturesAdapter(_make_config())
        trade_event = {
            "e": "trade",
            "E": 1609459200000,
            "s": "BTCUSDT",
            "t": 987654321,
            "p": "50000.00",
            "q": "0.15",
            "T": 1609459200000,
            "m": False,
        }
        result = await adapter._handle_message(json.dumps(trade_event))
        assert result is None


# ── Heartbeat ─────────────────────────────────────────────────────


class TestFuturesAdapterHeartbeat:

    @pytest.mark.asyncio
    async def test_send_heartbeat_pings_ws(self) -> None:
        adapter = FuturesAdapter(_make_config())
        mock_ws = AsyncMock()
        mock_ws.closed = False
        mock_ws.ping = AsyncMock()
        adapter._ws_session = mock_ws

        await adapter._send_heartbeat()

        mock_ws.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_heartbeat_when_closed(self) -> None:
        adapter = FuturesAdapter(_make_config())
        mock_ws = AsyncMock()
        mock_ws.closed = True
        adapter._ws_session = mock_ws

        await adapter._send_heartbeat()
        # Should not raise when session is closed


# ── _publish_event ────────────────────────────────────────────────


class TestFuturesAdapterPublish:

    def test_publish_event_logs(self) -> None:
        adapter = FuturesAdapter(_make_config())
        # _publish_event should only log — no exceptions expected
        adapter._publish_event({"event_type": "funding_rate"})


# ── _ts_ms_to_dt helper ──────────────────────────────────────────


class TestTsMsToDt:

    def test_convert_ms_to_dt(self) -> None:
        from apps.ingestion.binance_futures import _ts_ms_to_dt

        result = _ts_ms_to_dt(1609459200000)
        assert result == datetime(2021, 1, 1, 0, 0, tzinfo=UTC)

    def test_convert_none_to_now(self) -> None:
        from apps.ingestion.binance_futures import _ts_ms_to_dt

        result = _ts_ms_to_dt(None)
        assert result.tzinfo == UTC
