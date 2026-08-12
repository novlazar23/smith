"""Binance Futures exchange ingestion adapter.

Connects to the Binance Futures WebSocket API for live market data
(funding_rate, open_interest, liquidation) and to the REST API for
historical snapshots.

References:
    - REST API: https://binance-docs.github.io/apidocs/futures/en/
    - WebSocket: https://binance-docs.github.io/apidocs/futures/en/#websocket-market-streams
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from packages.domain.market_data.derivatives import (
    FundingRate,
    Liquidation,
    LiquidationSide,
    OpenInterest,
)
from packages.streaming.schemas import (
    MarketEvent,
)

from .base_adapter import (
    ConnectionConfig,
    ConnectionState,
    ExchangeAdapterBase,
)

logger = logging.getLogger(__name__)


def _ts_ms_to_dt(ms: int | None) -> datetime:
    """Convert millisecond timestamp to datetime(UTC).

    Args:
        ms: Millisecond timestamp. Pass ``None`` to get current time.

    Returns:
        A datetime in UTC.
    """
    if ms is None:
        return datetime.now(UTC)
    return datetime.fromtimestamp(ms / 1000, tz=UTC)


def _ts_ms_to_dt_or_now(ms: int) -> datetime:
    """Convert millisecond timestamp to datetime(UTC).

    Args:
        ms: Millisecond timestamp (must not be None).

    Returns:
        A datetime in UTC.
    """
    return datetime.fromtimestamp(ms / 1000, tz=UTC)


class FuturesAdapter(ExchangeAdapterBase):
    """Binance Futures (USDT-M) Exchange Adapter.

    Subscribes to:
    - fundingRate streams
    - openInterest snapshots
    - liquidation (forceOrder) streams

    Uses WebSocket for live data, REST for historical snapshots.
    """

    BASE_URL: str = "https://fapi.binance.com"
    WS_URL: str = "wss://fstream.binance.com:443/ws"

    def __init__(self, config: ConnectionConfig) -> None:
        super().__init__(config)
        self._streams: dict[str, Any] = {}
        self._event_handler: Any = None
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
        logger.info("FuturesAdapter connected to %s", self.WS_URL)

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
        logger.info("FuturesAdapter disconnected")

    # -- subscription -----------------------------------------------

    async def subscribe(self, streams: list[str]) -> None:
        """Subscribe to named Binance Futures WebSocket streams.

        Args:
            streams: Stream names such as ``["btcusdt_perpetual@funding",
                "btcusdt_perpetual@openInterest"]``.
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

    # -- funding rate (REST) ----------------------------------------

    async def fetch_funding_rate(self, symbol: str) -> FundingRate:
        """Fetch the latest funding rate for a futures symbol via REST.

        Args:
            symbol: Trading pair, e.g. ``"BTCUSDT"``.

        Returns:
            :class:`FundingRate` domain model.
        """
        params: dict[str, Any] = {"symbol": symbol.upper()}

        async with self._http_session.get(
            f"{self.BASE_URL}/fapi/v1/fundingRate", params=params
        ) as resp:
            resp.raise_for_status()
            raw: dict[str, Any] = await resp.json()

        mark_price = float(raw.get("markPrice", 0))
        funding_rate = float(raw.get("fundingRate", 0))
        funding_time_ms = int(raw.get("nextFundingTime", 0))
        next_funding_time = _ts_ms_to_dt(funding_time_ms)

        return FundingRate(
            instrument=symbol.upper(),
            venue="BINANCE_FUTURES",
            funding_rate=funding_rate,
            mark_price=mark_price,
            next_funding_time=next_funding_time,
            event_time=_ts_ms_to_dt_or_now(int(raw.get("time", 0))),
        )

    # -- open interest (REST) ---------------------------------------

    async def fetch_open_interest(self, symbol: str) -> OpenInterest:
        """Fetch the latest open interest for a futures symbol via REST.

        Args:
            symbol: Trading pair, e.g. ``"BTCUSDT"``.

        Returns:
            :class:`OpenInterest` domain model.
        """
        params: dict[str, Any] = {"symbol": symbol.upper()}

        async with self._http_session.get(
            f"{self.BASE_URL}/fapi/v1/openInterest", params=params
        ) as resp:
            resp.raise_for_status()
            raw: dict[str, Any] = await resp.json()

        return OpenInterest(
            instrument=symbol.upper(),
            venue="BINANCE_FUTURES",
            open_interest=float(raw.get("openInterest", 0)),
            event_time=_ts_ms_to_dt_or_now(int(raw.get("time", 0))),
        )

    # -- recent liquidations (REST) ---------------------------------

    async def fetch_recent_liquidations(
        self,
        symbol: str | None = None,
        limit: int = 25,
    ) -> list[Liquidation]:
        """Fetch recent liquidations via REST.

        Args:
            symbol: Optional trading pair filter. If ``None``, returns
                liquidations across all symbols.
            limit: Maximum number of liquidations to return.

        Returns:
            List of :class:`Liquidation` domain models sorted by time
            (newest first).
        """
        # Liquidation data is only available via WebSocket (force-order
        # stream) for Binance Futures free API keys.  The REST API
        # endpoint ``/fapi/v1/premiumIndex`` returns funding rate data,
        # not liquidations.  Returning an empty list with a documentation
        # note — consumers should use the WebSocket stream parser
        # ``_parse_liquidation`` for live liquidation events.
        return []

    # -- message parsers ---------------------------------------------

    async def _parse_funding_rate(self, data: dict[str, Any]) -> MarketEvent:
        """Parse a funding rate WebSocket message into a :class:`MarketEvent`.

        Binance sends ``{"e":"fundingRate","E":<ts>,"s":"SYMBOL",
        "r":<rate>,"p":<mark>,"T":<next_funding_ts>}``.
        """
        symbol = str(data.get("s", data.get("symbol", "")))
        event_time_ms = int(data.get("T", 0))

        return MarketEvent(
            event_id=str(self._next_sequence()),
            event_type="funding_rate",
            instrument=symbol,
            metadata=self._build_metadata("binance_futures", "BINANCE_FUTURES"),
            payload={
                "symbol": symbol,
                "funding_rate": float(data.get("r", 0)),
                "mark_price": float(data.get("p", 0)),
                "event_time": _ts_ms_to_dt_or_now(event_time_ms),
            },
        )

    async def _parse_open_interest(self, data: dict[str, Any]) -> MarketEvent:
        """Parse an open interest WebSocket message into a :class:`MarketEvent`.

        Binance sends ``{"oi":<oi>,"s":"SYMBOL","T":<ts>}``.
        """
        symbol = str(data.get("s", data.get("symbol", "")))
        return MarketEvent(
            event_id=str(self._next_sequence()),
            event_type="open_interest",
            instrument=symbol,
            metadata=self._build_metadata("binance_futures", "BINANCE_FUTURES"),
            payload={
                "symbol": symbol,
                "open_interest": float(data.get("oi", 0)),
                "event_time": _ts_ms_to_dt(int(data.get("T", 0))),
            },
        )

    async def _parse_liquidation(self, data: dict[str, Any]) -> MarketEvent:
        """Parse a liquidation order WebSocket message into a :class:`MarketEvent`.

        Binance sends ``{"e":"forceOrder","E":<ts>,"o":{...}}`` where
        ``o`` contains ``s``, ``p``, ``q``, ``S``, ``T``.
        """
        order = data.get("o", data)  # nested or flat format
        event_time_ms = int(data.get("E", order.get("T", 0)))
        side_raw = str(order.get("S", "UNKNOWN")).upper()

        _side: LiquidationSide = LiquidationSide.SHORT if side_raw == "SELL" else LiquidationSide.LONG

        price = float(order.get("p", 0))
        quantity = float(order.get("q", 0))

        return MarketEvent(
            event_id=str(self._next_sequence()),
            event_type="liquidation",
            instrument=str(order.get("s", data.get("symbol", ""))),
            metadata=self._build_metadata("binance_futures", "BINANCE_FUTURES"),
            payload={
                "symbol": order.get("s", data.get("symbol", "")),
                "side": str(order.get("S", "UNKNOWN")),
                "price": price,
                "quantity": quantity,
                "value": price * quantity,
                "event_time": _ts_ms_to_dt_or_now(event_time_ms),
            },
        )

    async def _parse_kline(self, data: dict[str, Any]) -> MarketEvent:
        """Parse a kline (candlestick) WebSocket message."""
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

        open_time = _ts_ms_to_dt(int(raw_row[0]))
        close_time = _ts_ms_to_dt(int(raw_row[6]))
        high_price = float(raw_row[2])
        low_price = float(raw_row[3])
        if high_price < low_price:
            high_price, low_price = low_price, high_price

        candle = {
            "instrument": symbol.upper(),
            "venue": "BINANCE_FUTURES",
            "timeframe": interval,
            "open_time": open_time,
            "close_time": close_time,
            "open": float(raw_row[1]),
            "high": high_price,
            "low": low_price,
            "close": float(raw_row[4]),
            "volume": float(raw_row[5]),
            "trade_count": int(raw_row[8]) if len(raw_row) > 8 else 0,
            "is_closed": True,
        }

        return MarketEvent(
            event_id=str(self._next_sequence()),
            event_type="candle",
            instrument=symbol,
            metadata=self._build_metadata("binance_futures", "BINANCE_FUTURES"),
            payload=candle,
        )

    async def _parse_trade(self, data: dict[str, Any]) -> MarketEvent:
        """Parse a trade WebSocket message."""
        symbol = str(data.get("s", ""))
        is_maker: bool = data.get("m", False)
        side: str = "sell" if is_maker else "buy"

        trade = {
            "trade_id": str(data["t"]),
            "instrument": symbol,
            "venue": "BINANCE_FUTURES",
            "price": float(data["p"]),
            "quantity": float(data["q"]),
            "side": side,
            "event_time": _ts_ms_to_dt(int(data.get("T", 0))),
        }

        return MarketEvent(
            event_id=str(self._next_sequence()),
            event_type="trade",
            instrument=symbol,
            metadata=self._build_metadata("binance_futures", "BINANCE_FUTURES"),
            payload=trade,
        )

    async def _parse_depth(self, data: dict[str, Any]) -> MarketEvent:
        """Parse a depth WebSocket message."""
        bids_raw = data.get("bids", data.get("b", []))
        asks_raw = data.get("asks", data.get("a", []))
        symbol = str(data.get("s", ""))

        bids: list[list[float]] = [
            [float(b[0]), float(b[1])] for b in bids_raw
        ]
        asks: list[list[float]] = [
            [float(a[0]), float(a[1])] for a in asks_raw
        ]

        snapshot = {
            "instrument": symbol,
            "venue": "BINANCE_FUTURES",
            "sequence": 0,
            "bids": bids,
            "asks": asks,
        }

        return MarketEvent(
            event_id=str(self._next_sequence()),
            event_type="orderbook_snapshot",
            instrument=symbol,
            metadata=self._build_metadata("binance_futures", "BINANCE_FUTURES"),
            payload=snapshot,
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
            "fundingRate": self._parse_funding_rate,
            "openInterest": self._parse_open_interest,
            "forceOrder": self._parse_liquidation,
        }

        parser = parsers.get(event_type)
        if parser is None:
            logger.debug("Unknown event type: %s", event_type)
            return

        market_event = await parser(data)
        await self._validate_and_publish(market_event.payload)

    # -- rate-limit hook ---------------------------------------------

    async def _send_heartbeat(self) -> None:
        """Send WebSocket ping for heartbeat."""
        await self._rate_limit_wait()
        if self._ws_session is not None and not self._ws_session.closed:
            await self._ws_session.ping()
            logger.debug("FuturesAdapter heartbeat sent.")

    # -- publish hook ------------------------------------------------

    def _publish_event(self, raw_event: dict[str, Any]) -> None:
        """Publish validated event (subclass hook)."""
        logger.debug("Published event: %s", raw_event.get("event_type", "unknown"))
