"""Binance Spot exchange ingestion adapter.

Connects to the Binance Spot WebSocket API for live market data
(candlesticks, trades, orderbook) and to the REST API for historical
snapshots.

References:
    - REST API: https://binance-docs.github.io/apidocs/spot/en/#public-api-definitions
    - WebSocket: https://binance-docs.github.io/apidocs/spot/en/#live-user-data-streams
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from packages.streaming.schemas import (
    Candle,
    MarketEvent,
    OrderBookSnapshot,
    Trade,
)

from .base_adapter import (
    ConnectionConfig,
    ConnectionState,
    ExchangeAdapterBase,
)

logger = logging.getLogger(__name__)


def _ts_ms_to_dt(ms: int) -> datetime:
    """Convert millisecond timestamp to datetime(UTC)."""
    return datetime.fromtimestamp(ms / 1000, tz=UTC)


class SpotAdapter(ExchangeAdapterBase):
    """Binance Spot Exchange Adapter.

    Subscribes to:
    - kline (candlestick) streams
    - ticker / miniTicker streams
    - depth (orderbook) streams

    Uses WebSocket for live data, REST for historical snapshots.
    """

    BASE_URL = "https://api.binance.com"
    WS_URL = "wss://stream.binance.com:9443/ws"

    def __init__(self, config: ConnectionConfig) -> None:
        super().__init__(config)
        self._streams: dict[str, Any] = {}
        self._ws_session: Any = None
        self._http_session: Any = None

    # -- lifecycle ---------------------------------------------------

    async def connect(self) -> None:
        """Open REST and WebSocket sessions.

        Raises:
            ConnectionError: If the connection fails.
        """
        if self._state == ConnectionState.CONNECTED:
            logger.debug("Already connected - skipping connect()")
            return

        import aiohttp

        self._http_session = aiohttp.ClientSession()
        try:
            self._ws_session = await aiohttp.ClientSession().ws_connect(
                self.WS_URL,
                heartbeat=30,
            )
        except (OSError, aiohttp.ClientError) as exc:
            await self._http_session.close()
            self._http_session = None
            raise ConnectionError(f"WebSocket connect failed: {exc}") from exc

        self._state = ConnectionState.CONNECTED
        logger.info("SpotAdapter connected to %s", self.WS_URL)

    async def disconnect(self) -> None:
        """Close WebSocket and HTTP session."""
        if self._ws_session is not None and not self._ws_session.closed:
            await self._ws_session.close()
            self._ws_session = None

        if self._http_session is not None:
            await self._http_session.close()
            self._http_session = None

        self._state = ConnectionState.DISCONNECTED
        self._streams.clear()
        logger.info("SpotAdapter disconnected")

    # -- subscription -----------------------------------------------

    async def subscribe(self, streams: list[str]) -> None:
        """Subscribe to named Binance WebSocket streams.

        Args:
            streams: Stream names such as ``["btcusdt@kline_1m",
                "btcusdt@depth20@100ms"]``.
        """
        if self._state != ConnectionState.CONNECTED:
            raise RuntimeError("Not connected - call connect() first")

        if not streams:
            return

        params: dict[str, Any] = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": 1,
        }
        payload = json.dumps(params)

        if self._ws_session is not None and not self._ws_session.closed:
            await self._ws_session.send_str(payload)
            for s in streams:
                self._streams[s] = len(self._streams)
            logger.info("Subscribed to %d streams", len(streams))

    # -- historical candles ------------------------------------------

    async def fetch_historical_candles(
        self,
        symbol: str,
        interval: str = "1m",
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 1000,
    ) -> list[Candle]:
        """Fetch historical candlestick (OHLCV) data via REST.

        Args:
            symbol: Trading pair, e.g. ``"BTCUSDT"``.
            interval: Candlestick interval.
            start_time: Start of the requested window (UTC).
            end_time: End of the requested window (UTC).
            limit: Maximum number of candles per request.

        Returns:
            List of :class:`Candle` objects sorted by open_time.
        """
        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(limit, 1000),
        }
        if start_time is not None:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time is not None:
            params["endTime"] = int(end_time.timestamp() * 1000)

        url = f"{self.BASE_URL}/api/v3/klines"
        candles: list[Candle] = []

        async with self._http_session.get(url, params=params) as resp:
            resp.raise_for_status()
            raw = await resp.json()

        for row in raw:
            try:
                candles.append(self._raw_kline_to_candle(row, symbol, interval))
            except (ValueError, KeyError) as exc:
                logger.warning("Skipping invalid kline row: %s", exc)

        return candles

    def _raw_kline_to_candle(
        self, row: list[Any], symbol: str, interval: str
    ) -> Candle:
        """Convert a Binance kline row to a :class:`Candle`.

        Row indices:
            0=open_time_ms, 1=open, 2=high, 3=low, 4=close,
            5=volume, 6=close_time_ms, 8=trade_count
        """
        open_time = _ts_ms_to_dt(int(row[0]))
        close_time = _ts_ms_to_dt(int(row[6]))
        open_price = float(row[1])
        high_price = float(row[2])
        low_price = float(row[3])
        close_price = float(row[4])
        volume = float(row[5])
        trade_count = int(row[8]) if len(row) > 8 else 0

        if low_price <= 0:
            raise ValueError(f"low must be > 0, got {low_price}")
        if high_price < low_price:
            high_price, low_price = low_price, high_price

        return Candle(
            instrument=symbol.upper(),
            venue="BINANCE",
            timeframe=interval,
            open_time=open_time,
            close_time=close_time,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
            trade_count=trade_count,
            is_closed=True,
        )

    # -- historical trades -------------------------------------------

    async def fetch_historical_trades(
        self,
        symbol: str,
        from_id: int | None = None,
        limit: int = 100,
    ) -> list[Trade]:
        """Fetch historical trades via REST.

        Args:
            symbol: Trading pair.
            from_id: Trade ID to start from.
            limit: Maximum number of trades.

        Returns:
            List of :class:`Trade` objects.
        """
        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "limit": min(limit, 1000),
        }
        if from_id is not None:
            params["fromId"] = from_id

        url = f"{self.BASE_URL}/api/v3/trades"
        trades: list[Trade] = []

        async with self._http_session.get(url, params=params) as resp:
            resp.raise_for_status()
            raw = await resp.json()

        for row in raw:
            try:
                trades.append(self._raw_trade_to_trade(row, symbol))
            except (ValueError, KeyError) as exc:
                logger.warning("Skipping invalid trade row: %s", exc)

        return trades

    def _raw_trade_to_trade(
        self, row: dict[str, Any], symbol: str
    ) -> Trade:
        """Convert a Binance trade row to a :class:`Trade`."""
        is_maker: bool = row.get("buyerIsMaker", False)
        side: str = "sell" if is_maker else "buy"

        return Trade(
            trade_id=str(row["id"]),
            instrument=symbol.upper(),
            venue="BINANCE",
            price=float(row["price"]),
            quantity=float(row["qty"]),
            side=side,
            event_time=_ts_ms_to_dt(int(row.get("time", 0))),
        )

    # -- message parsers ---------------------------------------------

    async def _parse_kline(self, data: dict[str, Any]) -> MarketEvent:
        """Parse a kline WebSocket message into a :class:`MarketEvent`."""
        kline = data.get("k", {})
        symbol = str(data.get("s", ""))
        interval = str(kline.get("i", "1m"))

        raw_row: list[Any] = [
            kline.get("t", 0),
            kline.get("o", "0"),
            kline.get("h", "0"),
            kline.get("l", "0"),
            kline.get("c", "0"),
            kline.get("v", "0"),
            kline.get("T", 0),
            0,
            kline.get("n", 0),
        ]

        candle = self._raw_kline_to_candle(raw_row, symbol, interval)
        event_id = str(self._next_sequence())
        metadata = self._build_metadata("binance", "BINANCE")

        return MarketEvent(
            event_id=event_id,
            event_type="candle",
            instrument=symbol,
            metadata=metadata,
            payload=candle.to_dict(),
        )

    async def _parse_trade(self, data: dict[str, Any]) -> MarketEvent:
        """Parse a trade WebSocket message into a :class:`MarketEvent`."""
        symbol = str(data.get("s", ""))
        is_maker: bool = data.get("m", False)
        side: str = "sell" if is_maker else "buy"

        trade = Trade(
            trade_id=str(data["t"]),
            instrument=symbol,
            venue="BINANCE",
            price=float(data["p"]),
            quantity=float(data["q"]),
            side=side,
            event_time=_ts_ms_to_dt(int(data.get("T", 0))),
        )
        event_id = str(self._next_sequence())
        metadata = self._build_metadata("binance", "BINANCE")

        return MarketEvent(
            event_id=event_id,
            event_type="trade",
            instrument=symbol,
            metadata=metadata,
            payload=trade.to_dict(),
        )

    async def _parse_depth(self, data: dict[str, Any]) -> MarketEvent:
        """Parse a depth WebSocket message into a :class:`MarketEvent`."""
        bids_raw = data.get("bids", data.get("b", []))
        asks_raw = data.get("asks", data.get("a", []))
        sequence = int(data.get("lastUpdateId", data.get("u", 0)))
        symbol = str(data.get("s", ""))

        bids: list[list[float]] = [
            [float(b[0]), float(b[1])] for b in bids_raw
        ]
        asks: list[list[float]] = [
            [float(a[0]), float(a[1])] for a in asks_raw
        ]

        snapshot = OrderBookSnapshot(
            instrument=symbol,
            venue="BINANCE",
            sequence=sequence,
            bids=bids,
            asks=asks,
            metadata={
                "lastUpdateId": data.get("lastUpdateId"),
                "updateId": data.get("u"),
            },
        )
        event_id = str(self._next_sequence())
        metadata = self._build_metadata("binance", "BINANCE")

        return MarketEvent(
            event_id=event_id,
            event_type="orderbook_snapshot",
            instrument=symbol,
            metadata=metadata,
            payload=snapshot.to_dict(),
        )

    async def _parse_ticker(self, data: dict[str, Any]) -> MarketEvent:
        """Parse a 24hr ticker event."""
        return MarketEvent(
            event_id=str(self._next_sequence()),
            event_type="ticker",
            instrument=str(data.get("s", "")),
            metadata=self._build_metadata("binance", "BINANCE"),
            payload={
                "price": float(data.get("c", 0)),
                "volume": float(data.get("v", 0)),
                "event_time": data.get("E"),
            },
        )

    async def _parse_min_ticker(self, data: dict[str, Any]) -> MarketEvent:
        """Parse a mini-ticker event."""
        return MarketEvent(
            event_id=str(self._next_sequence()),
            event_type="miniTicker",
            instrument=str(data.get("s", "")),
            metadata=self._build_metadata("binance", "BINANCE"),
            payload={
                "price": float(data.get("c", 0)),
                "volume": float(data.get("v", 0)),
                "event_time": data.get("E"),
            },
        )

    # -- message dispatch --------------------------------------------

    async def _handle_message(self, raw: str) -> None:
        """Dispatch a single raw WebSocket message to the correct parser."""
        if not raw:
            return

        try:
            data: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("Invalid JSON: %s", exc)
            return

        event_type = data.get("e", "")

        parsers: dict[str, Any] = {
            "kline": self._parse_kline,
            "trade": self._parse_trade,
            "depthUpdate": self._parse_depth,
            "partialBookDepth": self._parse_depth,
            "ticker": self._parse_ticker,
            "miniTicker": self._parse_min_ticker,
        }

        parser = parsers.get(event_type)
        if parser is None:
            logger.debug("Unknown event type: %s", event_type)
            return

        market_event = await parser(data)
        # Validate then publish via base class
        await self._validate_and_publish(market_event.payload)

    # -- rate-limit hook ---------------------------------------------

    async def _send_heartbeat(self) -> None:
        """Send WebSocket ping for heartbeat."""
        await self._rate_limit_wait()
        if self._ws_session is not None and not self._ws_session.closed:
            await self._ws_session.ping()
            logger.debug("SpotAdapter heartbeat sent.")

    # -- publish hook ------------------------------------------------

    def _publish_event(self, raw_event: dict[str, Any]) -> None:
        """Publish validated event (subclass hook).

        Connects to the message broker downstream. For now we simply
        log the published event.
        """
        logger.debug("Published event: %s", raw_event.get("event_type", "unknown"))
