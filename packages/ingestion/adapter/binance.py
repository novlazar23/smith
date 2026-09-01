"""Binance Futures / Spot synchronous adapter.

Connects to the Binance Futures REST API for market data fetches.
Supports candles, trades, orderbook depth, and ticker snapshots.

References:
    - REST API: https://binance-docs.github.io/apidocs/futures/en/
    - Spot REST: https://developers.binance.com/docs/spot-market
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

import aiohttp

from .base import (
    ConnectionConfig,
    ConnectionError,
    ConnectionState,
    ExchangeAdapterBase,
    VenueFees,
)

logger = logging.getLogger(__name__)

# ── Binance venue constants ──────────────────────────────────────────

BINANCE_FUTURES_VENUE = "BINANCE_FUTURES"
BINANCE_FUTURES_BASE_URL = "https://fapi.binance.com"
BINANCE_SPOT_BASE_URL = "https://api.binance.com"

BINANCE_FEES = VenueFees(
    taker_rate=0.0004,
    maker_rate=0.0001,
    spread_bps=1.0,
)


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


class BinanceAdapter(ExchangeAdapterBase):
    """Binance Futures/Spot synchronous adapter.

    Uses aiohttp for REST calls (no WebSocket for simplicity).
    Implements connect/disconnect, subscribe, and all market data
    fetchers (candles, trades, orderbook, ticker).
    """

    def __init__(
        self,
        config: ConnectionConfig | None = None,
        use_futures: bool = True,
    ) -> None:
        venue = BINANCE_FUTURES_VENUE if use_futures else "BINANCE_SPOT"
        base_url = BINANCE_FUTURES_BASE_URL if use_futures else BINANCE_SPOT_BASE_URL
        fees = BINANCE_FEES

        if config is None:
            config = ConnectionConfig()

        config.base_url = base_url
        config.venue = venue
        config.fees = fees
        super().__init__(config)

        self._http_session: aiohttp.ClientSession | None = None
        self._subscribed_streams: set[str] = set()

    # -- lifecycle ----------------------------------------------------

    async def connect(self) -> None:
        """Open an aiohttp ClientSession.

        Raises:
            ConnectionError: If session creation fails.
        """
        if self._state == ConnectionState.CONNECTED:
            logger.debug("Already connected — skipping connect()")
            return

        try:
            self._http_session = aiohttp.ClientSession()
        except Exception as exc:
            raise ConnectionError(f"Failed to create HTTP session: {exc}") from exc

        self._state = ConnectionState.CONNECTED
        logger.info(
            "BinanceAdapter connected to %s [futures=%s]",
            self.config.base_url,
            "futures" if "fapi" in self.config.base_url else "spot",
        )

    async def disconnect(self) -> None:
        """Close the HTTP session."""
        if self._http_session is not None:
            await self._http_session.close()
            self._http_session = None

        self._state = ConnectionState.DISCONNECTED
        self._subscribed_streams.clear()
        logger.info("BinanceAdapter disconnected")

    async def subscribe(self, streams: list[str]) -> None:
        """Subscribe to named streams (stored for reference).

        Args:
            streams: Stream names such as ``["btcusdt@trade", "btcusdt@kline_1m"]``.
        """
        if self._state != ConnectionState.CONNECTED:
            raise RuntimeError("Not connected — call connect() first")

        if not streams:
            return

        for s in streams:
            self._subscribed_streams.add(s)

        logger.info("BinanceAdapter subscribed to %d streams: %s", len(streams), streams)

    # -- market data: candles -----------------------------------------

    async def _fetch_candles_raw(
        self, symbol: str, interval: str = "1m", limit: int = 100
    ) -> list[dict[str, Any]]:
        """Fetch raw kline (candle) data from Binance REST API.

        Args:
            symbol: Trading pair, e.g. ``"BTCUSDT"``.
            interval: Candle interval, e.g. ``"1m"``, ``"5m"``, ``"1h"``.
            limit: Number of candles (max 1000).

        Returns:
            List of raw candle dicts from the API.
        """
        await self._rate_limit_wait()

        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(limit, 1000),
        }

        if self._http_session is None:
            raise ConnectionError("HTTP session not open — call connect() first")

        candles: list[dict[str, Any]] = []
        async with self._http_session.get(
            f"{self.config.base_url}/klines", params=params
        ) as resp:
            resp.raise_for_status()
            raw_data = await resp.json()

        for row in raw_data:
            candles.append({
                "open_time": _ts_ms_to_dt(int(row[0])),
                "close_time": _ts_ms_to_dt(int(row[6])),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
                "trade_count": int(row[8]) if len(row) > 8 else 0,
                "is_closed": bool(row[8]) if len(row) > 8 else True,
            })

        return candles

    # -- market data: trades ------------------------------------------

    async def _fetch_trades_raw(
        self, symbol: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Fetch recent trades from Binance REST API.

        Args:
            symbol: Trading pair, e.g. ``"BTCUSDT"``.
            limit: Number of recent trades (max 1000).

        Returns:
            List of raw trade dicts.
        """
        await self._rate_limit_wait()

        if self._http_session is None:
            raise ConnectionError("HTTP session not open — call connect() first")

        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "limit": min(limit, 1000),
        }

        trades: list[dict[str, Any]] = []
        async with self._http_session.get(
            f"{self.config.base_url}/trades", params=params
        ) as resp:
            resp.raise_for_status()
            raw_data = await resp.json()

        for row in raw_data:
            is_maker = bool(row.get("isBuyerMaker", False))
            trades.append({
                "trade_id": str(row.get("id", uuid.uuid4().hex[:12])),
                "price": float(row["price"]),
                "quantity": float(row["qty"]),
                "side": "sell" if is_maker else "buy",
                "event_time": _ts_ms_to_dt(int(row.get("time", 0))),
            })

        return trades

    # -- market data: orderbook ---------------------------------------

    async def _fetch_orderbook_raw(
        self, symbol: str, limit: int = 20
    ) -> dict[str, Any]:
        """Fetch orderbook snapshot from Binance REST API.

        Args:
            symbol: Trading pair, e.g. ``"BTCUSDT"``.
            limit: Depth per side (max 5000 for futures, 5000 for spot).

        Returns:
            Dict with "bids" and "asks" as list of [price, quantity].
        """
        await self._rate_limit_wait()

        if self._http_session is None:
            raise ConnectionError("HTTP session not open — call connect() first")

        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "limit": min(limit, 5000),
        }

        async with self._http_session.get(
            f"{self.config.base_url}/depth", params=params
        ) as resp:
            resp.raise_for_status()
            raw_data = await resp.json()

        return {
            "bids": [
                [float(b[0]), float(b[1])]
                for b in raw_data.get("bids", [])
            ],
            "asks": [
                [float(a[0]), float(a[1])]
                for a in raw_data.get("asks", [])
            ],
        }

    # -- market data: ticker ------------------------------------------

    async def _fetch_ticker_raw(self, symbol: str) -> dict[str, Any]:
        """Fetch 24h ticker snapshot from Binance REST API.

        Args:
            symbol: Trading pair, e.g. ``"BTCUSDT"``.

        Returns:
            Dict with price, volume, high, low, open, etc.
        """
        await self._rate_limit_wait()

        if self._http_session is None:
            raise ConnectionError("HTTP session not open — call connect() first")

        params: dict[str, Any] = {"symbol": symbol.upper()}

        async with self._http_session.get(
            f"{self.config.base_url}/ticker/24hr", params=params
        ) as resp:
            resp.raise_for_status()
            raw_data = await resp.json()

        last_price = float(raw_data.get("lastPrice", 0))
        spread = self.config.fees.spread_price(last_price)

        return {
            "symbol": raw_data.get("symbol", symbol.upper()),
            "price": last_price,
            "bid": float(raw_data.get("bidPrice", last_price - spread)),
            "ask": float(raw_data.get("askPrice", last_price + spread)),
            "high": float(raw_data.get("highPrice", 0)),
            "low": float(raw_data.get("lowPrice", 0)),
            "volume": float(raw_data.get("volume", 0)),
            "quote_volume": float(raw_data.get("quoteVolume", 0)),
            "open": float(raw_data.get("openPrice", 0)),
            "close": last_price,
            "price_change_pct": float(raw_data.get("priceChangePercent", 0)),
            "trade_count": int(raw_data.get("count", 0)),
            "event_time": _ts_ms_to_dt(int(raw_data.get("time", 0))),
        }

    # -- heartbeat hook -----------------------------------------------

    async def _send_heartbeat(self) -> None:
        """Send a ping to Binance via time endpoint."""
        await self._rate_limit_wait()

        if self._http_session is None:
            return

        try:
            async with self._http_session.get(
                f"{self.config.base_url}/time"
            ) as resp:
                if resp.status == 200:
                    logger.debug("BinanceAdapter heartbeat OK.")
                else:
                    logger.warning(
                        "BinanceAdapter heartbeat HTTP %d", resp.status,
                    )
        except Exception as exc:
            logger.warning("BinanceAdapter heartbeat failed: %s", exc)

    # -- publish hook -------------------------------------------------

    def _publish_event(self, raw_event: dict[str, Any]) -> None:
        """Publish validated event (subclass hook)."""
        logger.debug("Published event: %s", raw_event.get("type", "unknown"))

    # -- utility ------------------------------------------------------

    def get_fee_structure(self) -> dict[str, Any]:
        """Return fee structure for this Binance adapter.

        Returns:
            Dict with taker/maker rates and spread.
        """
        return {
            "venue": self.config.venue,
            "taker": self.config.fees.taker_rate,
            "maker": self.config.fees.maker_rate,
            "spread_bps": self.config.fees.spread_bps,
        }
