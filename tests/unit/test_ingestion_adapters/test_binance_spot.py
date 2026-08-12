"""Tests for Binance Spot adapter (SpotAdapter).

Uses only mocked HTTP/WebSocket calls - no network traffic.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from apps.ingestion.base_adapter import ConnectionState
from apps.ingestion.binance_spot import SpotAdapter


def _make_connection_config() -> Any:
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
    return cfg


def _valid_kline_event() -> dict[str, Any]:
    """Return a valid Binance kline WebSocket message."""
    return {
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


def _valid_trade_event() -> dict[str, Any]:
    """Return a valid Binance trade WebSocket message."""
    return {
        "e": "trade",
        "E": 1609459200000,
        "s": "BTCUSDT",
        "t": 987654321,
        "p": "50000.00",
        "q": "0.15",
        "b": 11111,
        "a": 22222,
        "T": 1609459200000,
        "m": False,
        "M": True,
    }


def _valid_depth_event() -> dict[str, Any]:
    """Return a valid depth update WebSocket message."""
    return {
        "e": "depthUpdate",
        "E": 1609459200000,
        "s": "BTCUSDT",
        "lastUpdateId": 123456,
        "U": 123450,
        "u": 123456,
        "bids": [["49990.00", "1.50"], ["49980.00", "2.00"]],
        "asks": [["50010.00", "1.20"], ["50020.00", "0.80"]],
    }


def _valid_partial_depth_event() -> dict[str, Any]:
    """Return a partial depth event."""
    return {
        "e": "partialBookDepth",
        "s": "BTCUSDT",
        "lastUpdateId": 123456,
        "bids": [["49990.00", "1.50"]],
        "asks": [["50010.00", "1.20"]],
    }


def _raw_kline_row() -> list[Any]:
    """Return a raw Binance kline REST API row."""
    return [
        1609459200000,
        "50000.00",
        "50100.00",
        "49900.00",
        "50050.00",
        "10.5",
        1609459260000,
        "525000.00",
        150,
    ]


def _raw_trade_row() -> dict[str, Any]:
    """Return a raw Binance trades REST API row."""
    return {
        "id": 987654321,
        "price": "50000.00",
        "qty": "0.15",
        "buyerIsMaker": False,
        "time": 1609459200000,
        "isBestMatch": True,
    }


# -- __init__ --

class TestSpotAdapterInit:

    def test_init_sets_base_url(self) -> None:
        adapter = SpotAdapter(_make_connection_config())
        assert adapter.BASE_URL == "https://api.binance.com"

    def test_init_sets_ws_url(self) -> None:
        adapter = SpotAdapter(_make_connection_config())
        assert adapter.WS_URL == "wss://stream.binance.com:9443/ws"

    def test_init_initialises_state(self) -> None:
        adapter = SpotAdapter(_make_connection_config())
        assert adapter._streams == {}


# -- connect / disconnect --

class TestSpotAdapterLifecycle:

    @pytest.mark.asyncio
    async def test_connect_already_connected(self) -> None:
        adapter = SpotAdapter(_make_connection_config())
        adapter._state = ConnectionState.CONNECTED
        with patch("aiohttp.ClientSession") as mock_session:
            await adapter.connect()
            mock_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_disconnect_already_disconnected(self) -> None:
        adapter = SpotAdapter(_make_connection_config())
        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect_clears_state(self) -> None:
        adapter = SpotAdapter(_make_connection_config())
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
        adapter = SpotAdapter(_make_connection_config())
        with pytest.raises(RuntimeError, match="Not connected"):
            await adapter.subscribe(["btcusdt@kline_1m"])


# -- subscribe --

class TestSpotAdapterSubscribe:

    @pytest.mark.asyncio
    async def test_subscribe_sends_ws_message(self) -> None:
        adapter = SpotAdapter(_make_connection_config())
        mock_ws = AsyncMock()
        mock_ws.closed = False
        mock_ws.send_str = AsyncMock()
        adapter._ws_session = mock_ws
        adapter._state = ConnectionState.CONNECTED

        await adapter.subscribe(["btcusdt@kline_1m"])

        mock_ws.send_str.assert_called_once()
        call_arg = mock_ws.send_str.call_args[0][0]
        parsed = __import__("json").loads(call_arg)
        assert parsed["method"] == "SUBSCRIBE"
        assert parsed["params"] == ["btcusdt@kline_1m"]
        assert parsed["id"] == 1

    @pytest.mark.asyncio
    async def test_subscribe_multi_streams(self) -> None:
        adapter = SpotAdapter(_make_connection_config())
        mock_ws = AsyncMock()
        mock_ws.closed = False
        mock_ws.send_str = AsyncMock()
        adapter._ws_session = mock_ws
        adapter._state = ConnectionState.CONNECTED

        streams = ["btcusdt@kline_1m", "btcusdt@depth20"]
        await adapter.subscribe(streams)

        call_arg = mock_ws.send_str.call_args[0][0]
        parsed = __import__("json").loads(call_arg)
        assert parsed["params"] == streams
        assert parsed["id"] == 1

    @pytest.mark.asyncio
    async def test_subscribe_empty_list(self) -> None:
        adapter = SpotAdapter(_make_connection_config())
        mock_ws = AsyncMock()
        mock_ws.closed = False
        mock_ws.send_str = AsyncMock()
        adapter._ws_session = mock_ws
        adapter._state = ConnectionState.CONNECTED

        await adapter.subscribe([])
        mock_ws.send_str.assert_not_called()


# -- fetch_historical_candles --

class TestSpotAdapterFetchCandles:

    @pytest.mark.asyncio
    async def test_fetch_historical_candles(self) -> None:
        adapter = SpotAdapter(_make_connection_config())
        raw_rows = [_raw_kline_row()]

        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = AsyncMock(return_value=raw_rows)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.close = AsyncMock()
        adapter._http_session = mock_session

        candles = await adapter.fetch_historical_candles(
            symbol="btcusdt",
            interval="1m",
            limit=100,
        )

        assert len(candles) == 1
        candle = candles[0]
        from packages.streaming.schemas import Candle as CandleSC
        assert isinstance(candle, CandleSC)
        assert candle.instrument == "BTCUSDT"
        assert candle.venue == "BINANCE"
        assert candle.timeframe == "1m"
        assert candle.open == 50000.0
        assert candle.high == 50100.0
        assert candle.low == 49900.0
        assert candle.close == 50050.0
        assert candle.volume == 10.5
        assert candle.trade_count == 150
        assert candle.is_closed is True

    @pytest.mark.asyncio
    async def test_fetch_historical_candles_with_time_range(self) -> None:
        adapter = SpotAdapter(_make_connection_config())
        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = AsyncMock(return_value=[])
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.close = AsyncMock()
        adapter._http_session = mock_session

        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 2, tzinfo=UTC)

        await adapter.fetch_historical_candles(
            symbol="BTCUSDT",
            interval="1h",
            start_time=start,
            end_time=end,
        )

        call_kwargs = mock_session.get.call_args
        params = call_kwargs[1]["params"]
        assert params["startTime"] == int(start.timestamp() * 1000)
        assert params["endTime"] == int(end.timestamp() * 1000)
        assert params["symbol"] == "BTCUSDT"
        assert params["interval"] == "1h"

    @pytest.mark.asyncio
    async def test_fetch_historical_candles_invalid_skips(self) -> None:
        adapter = SpotAdapter(_make_connection_config())
        invalid_row = [
            1609459200000, "50000", "50100", "0", "50050", "10.5", 1609459260000, "0", 0
        ]
        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = AsyncMock(return_value=[invalid_row])
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.close = AsyncMock()
        adapter._http_session = mock_session

        candles = await adapter.fetch_historical_candles(
            symbol="BTCUSDT", interval="1m", limit=10
        )
        assert len(candles) == 0

    @pytest.mark.asyncio
    async def test_fetch_historical_candles_limit_capped(self) -> None:
        adapter = SpotAdapter(_make_connection_config())
        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = AsyncMock(return_value=[])
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.close = AsyncMock()
        adapter._http_session = mock_session

        await adapter.fetch_historical_candles(
            symbol="BTCUSDT", interval="1m", limit=5000
        )
        params = mock_session.get.call_args[1]["params"]
        assert params["limit"] == 1000


# -- fetch_historical_trades --

class TestSpotAdapterFetchTrades:

    @pytest.mark.asyncio
    async def test_fetch_historical_trades(self) -> None:
        adapter = SpotAdapter(_make_connection_config())
        raw = [_raw_trade_row()]

        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = AsyncMock(return_value=raw)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.close = AsyncMock()
        adapter._http_session = mock_session

        trades = await adapter.fetch_historical_trades(
            symbol="BTCUSDT", limit=100
        )

        assert len(trades) == 1
        trade = trades[0]
        from packages.streaming.schemas import Trade as TradeSC
        assert isinstance(trade, TradeSC)
        assert trade.trade_id == "987654321"
        assert trade.instrument == "BTCUSDT"
        assert trade.venue == "BINANCE"
        assert trade.price == 50000.0
        assert trade.quantity == 0.15
        assert trade.side == "buy"

    @pytest.mark.asyncio
    async def test_fetch_historical_trades_maker_sell(self) -> None:
        adapter = SpotAdapter(_make_connection_config())
        raw = [_raw_trade_row()]
        raw[0]["buyerIsMaker"] = True

        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = AsyncMock(return_value=raw)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.close = AsyncMock()
        adapter._http_session = mock_session

        trades = await adapter.fetch_historical_trades(symbol="BTCUSDT", limit=10)
        assert trades[0].side == "sell"

    @pytest.mark.asyncio
    async def test_fetch_historical_trades_with_from_id(self) -> None:
        adapter = SpotAdapter(_make_connection_config())
        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = AsyncMock(return_value=[])
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.close = AsyncMock()
        adapter._http_session = mock_session

        await adapter.fetch_historical_trades(
            symbol="BTCUSDT", from_id=999, limit=50
        )

        params = mock_session.get.call_args[1]["params"]
        assert params["fromId"] == 999

    @pytest.mark.asyncio
    async def test_fetch_historical_trades_limit_capped(self) -> None:
        adapter = SpotAdapter(_make_connection_config())
        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = AsyncMock(return_value=[])
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.close = AsyncMock()
        adapter._http_session = mock_session

        await adapter.fetch_historical_trades(symbol="BTCUSDT", limit=2000)
        params = mock_session.get.call_args[1]["params"]
        assert params["limit"] == 1000


# -- _parse_kline --

class TestSpotAdapterParseKline:

    @pytest.mark.asyncio
    async def test_parse_kline_valid(self) -> None:
        adapter = SpotAdapter(_make_connection_config())
        event = await adapter._parse_kline(_valid_kline_event())

        assert event.event_type == "candle"
        assert event.instrument == "BTCUSDT"
        assert "high" in event.payload
        assert event.payload["high"] == 50100.0

    @pytest.mark.asyncio
    async def test_parse_kline_inverted_high_low(self) -> None:
        adapter = SpotAdapter(_make_connection_config())
        event_data = _valid_kline_event()
        event_data["k"]["h"] = "49000.00"
        event_data["k"]["l"] = "50200.00"

        event = await adapter._parse_kline(event_data)
        assert event.payload["high"] == 50200.0
        assert event.payload["low"] == 49000.0


# -- _parse_trade --

class TestSpotAdapterParseTrade:

    @pytest.mark.asyncio
    async def test_parse_trade_taker_buy(self) -> None:
        adapter = SpotAdapter(_make_connection_config())
        event = await adapter._parse_trade(_valid_trade_event())

        assert event.event_type == "trade"
        assert event.payload["trade_id"] == "987654321"
        assert event.payload["price"] == 50000.0
        assert event.payload["side"] == "buy"

    @pytest.mark.asyncio
    async def test_parse_trade_maker_sell(self) -> None:
        adapter = SpotAdapter(_make_connection_config())
        event_data = _valid_trade_event()
        event_data["m"] = True

        event = await adapter._parse_trade(event_data)
        assert event.payload["side"] == "sell"


# -- _parse_depth --

class TestSpotAdapterParseDepth:

    @pytest.mark.asyncio
    async def test_parse_depth_update(self) -> None:
        adapter = SpotAdapter(_make_connection_config())
        event = await adapter._parse_depth(_valid_depth_event())

        assert event.event_type == "orderbook_snapshot"
        assert event.instrument == "BTCUSDT"
        assert len(event.payload["bids"]) == 2
        assert event.payload["bids"][0] == [49990.0, 1.50]
        assert len(event.payload["asks"]) == 2
        assert event.payload["asks"][0] == [50010.0, 1.20]

    @pytest.mark.asyncio
    async def test_parse_partial_depth(self) -> None:
        adapter = SpotAdapter(_make_connection_config())
        event = await adapter._parse_depth(_valid_partial_depth_event())

        assert event.event_type == "orderbook_snapshot"
        assert len(event.payload["bids"]) == 1
        assert len(event.payload["asks"]) == 1
        assert event.payload["bids"][0] == [49990.0, 1.50]
        assert event.payload["asks"][0] == [50010.0, 1.20]


# -- _handle_message --

class TestSpotAdapterHandleMessage:

    @pytest.mark.asyncio
    async def test_handle_kline_valid(self) -> None:
        adapter = SpotAdapter(_make_connection_config())
        result = await adapter._handle_message(
            __import__("json").dumps(_valid_kline_event())
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_trade_valid(self) -> None:
        adapter = SpotAdapter(_make_connection_config())
        result = await adapter._handle_message(
            __import__("json").dumps(_valid_trade_event())
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_depth_valid(self) -> None:
        adapter = SpotAdapter(_make_connection_config())
        result = await adapter._handle_message(
            __import__("json").dumps(_valid_depth_event())
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_empty_message(self) -> None:
        adapter = SpotAdapter(_make_connection_config())
        result = await adapter._handle_message("")
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_invalid_json(self) -> None:
        adapter = SpotAdapter(_make_connection_config())
        result = await adapter._handle_message("not valid json {{{")
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_unknown_event_type(self) -> None:
        adapter = SpotAdapter(_make_connection_config())
        result = await adapter._handle_message(
            __import__("json").dumps({"e": "unknown"})
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_ticker(self) -> None:
        adapter = SpotAdapter(_make_connection_config())
        ticker_event = {
            "e": "ticker",
            "E": 1609459200000,
            "s": "BTCUSDT",
            "c": "50050.00",
            "v": "1000.5",
        }
        result = await adapter._handle_message(
            __import__("json").dumps(ticker_event)
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_min_ticker(self) -> None:
        adapter = SpotAdapter(_make_connection_config())
        mini_ticker_event = {
            "e": "miniTicker",
            "E": 1609459200000,
            "s": "BTCUSDT",
            "c": "50050.00",
            "v": "1000.5",
        }
        result = await adapter._handle_message(
            __import__("json").dumps(mini_ticker_event)
        )
        assert result is None


# -- Candle model validation --

class TestSpotAdapterCandleValidation:

    @pytest.mark.asyncio
    async def test_candle_high_gte_low(self) -> None:
        adapter = SpotAdapter(_make_connection_config())
        candle = adapter._raw_kline_to_candle(
            [0, "100", "105", "95", "102", "10", 10, "0", 0],
            "BTCUSDT",
            "1m",
        )
        assert candle.high == 105.0
        assert candle.low == 95.0

    @pytest.mark.asyncio
    async def test_candle_invalid_low_raises(self) -> None:
        adapter = SpotAdapter(_make_connection_config())
        with pytest.raises(ValueError, match="low"):
            adapter._raw_kline_to_candle(
                [0, "100", "90", "0", "102", "10", 10, "0", 0],
                "BTCUSDT",
                "1m",
            )

    @pytest.mark.asyncio
    async def test_candle_inverted_swapped(self) -> None:
        adapter = SpotAdapter(_make_connection_config())
        candle = adapter._raw_kline_to_candle(
            [0, "100", "90", "105", "102", "10", 10, "0", 0],
            "BTCUSDT",
            "1m",
        )
        assert candle.high == 105.0
        assert candle.low == 90.0
