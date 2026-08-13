"""Dummy / simulated exchange adapter for testing and development.

Generates realistic simulated market data using a seeded random walk.
Supports candles, trades, orderbook depth, and ticker snapshots — all
with proper venue metadata attached.

Venue ID: ``DUMMY_EXCHANGE``
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from .base import (
    ConnectionConfig,
    ConnectionState,
    ExchangeAdapterBase,
    VenueFees,
)

logger = logging.getLogger(__name__)

# ── Dummy venue constants ────────────────────────────────────────────

DUMMY_VENUE = "DUMMY_EXCHANGE"

DUMMY_FEES = VenueFees(
    taker_rate=0.001,
    maker_rate=0.0005,
    spread_bps=5.0,
)

# Mapping of instrument names to base prices for reproducibility
INSTRUMENT_BASE_PRICES: dict[str, float] = {
    "BTC/USDT": 67500.0,
    "ETH/USDT": 3450.0,
    "SOL/USDT": 178.0,
    "BNB/USDT": 610.0,
    "XRP/USDT": 2.35,
}


def _make_seed(instrument: str, endpoint: str) -> int:
    """Create a deterministic seed from instrument + endpoint.

    Args:
        instrument: Trading pair.
        endpoint: Data type (e.g. "candles", "trades", "orderbook", "ticker").

    Returns:
        Integer seed for random.Random instance.
    """
    combined = f"{instrument}:{endpoint}"
    return int(hashlib.sha256(combined.encode()).hexdigest()[:8], 16)


class DummyAdapter(ExchangeAdapterBase):
    """Simulated exchange adapter.

    Generates deterministic random-walk market data based on a configurable
    seed and instrument base prices.  All events carry correct venue
    metadata (``DUMMY_EXCHANGE``).

    Example instruments (format: ``"BTC/USDT"``):
        - BTC/USDT
        - ETH/USDT
        - SOL/USDT
    """

    def __init__(
        self,
        config: ConnectionConfig | None = None,
        base_price: float = 67500.0,
        seed: int = 42,
    ) -> None:
        venue = DUMMY_VENUE
        fees = DUMMY_FEES

        if config is None:
            config = ConnectionConfig()

        config.base_url = "http://dummy.exchange/v1"
        config.venue = venue
        config.fees = fees
        super().__init__(config)

        self._base_price = base_price
        self._seed = seed
        self._rng = random.Random(seed)
        self._current_price = base_price
        self._start_time = datetime.now(UTC) - timedelta(hours=24)
        self._subscribed_streams: set[str] = set()

        # Cached orderbook levels for consistency
        self._ob_bids: list[list[float]] = []
        self._ob_asks: list[list[float]] = []
        self._generate_orderbook()

    # -- lifecycle ----------------------------------------------------

    async def connect(self) -> None:
        """Simulate connection to dummy exchange."""
        if self._state == ConnectionState.CONNECTED:
            logger.debug("Already connected — skipping connect()")
            return

        self._state = ConnectionState.CONNECTED
        logger.info("DummyAdapter connected to %s [seed=%d]", self.config.base_url, self._seed)

    async def disconnect(self) -> None:
        """Simulate disconnection."""
        self._state = ConnectionState.DISCONNECTED
        self._subscribed_streams.clear()
        logger.info("DummyAdapter disconnected")

    async def subscribe(self, streams: list[str]) -> None:
        """Store subscribed stream names.

        Args:
            streams: Stream names such as ``["BTC/USDT@trade", "BTC/USDT@kline_1m"]``.
        """
        if self._state != ConnectionState.CONNECTED:
            raise RuntimeError("Not connected — call connect() first")

        if not streams:
            return

        for s in streams:
            self._subscribed_streams.add(s)

        logger.info("DummyAdapter subscribed to %d streams: %s", len(streams), streams)

    # -- market data: candles -----------------------------------------

    async def _fetch_candles_raw(
        self, symbol: str, interval: str = "1m", limit: int = 100
    ) -> list[dict[str, Any]]:
        """Generate simulated candlestick data.

        Uses a seeded random walk based on ``_base_price``.

        Args:
            symbol: Trading pair, e.g. ``"BTC/USDT"``.
            interval: Timeframe (e.g. ``"1m"``, ``"5m"``).
            limit: Number of candles.

        Returns:
            List of simulated candle dicts.
        """
        rng = random.Random(_make_seed(symbol, "candles"))
        interval_minutes = self._interval_to_minutes(interval)
        candle_duration = timedelta(minutes=interval_minutes)
        now = datetime.now(UTC)
        open_time = now - timedelta(minutes=interval_minutes * limit)

        candles: list[dict[str, Any]] = []
        price = self._base_price

        for i in range(limit):
            close_time = open_time + candle_duration
            volatility = price * 0.002  # 0.2% per candle
            change = rng.gauss(0, volatility)
            open_price = price
            close_price = price + change
            high_price = max(open_price, close_price) + rng.uniform(0, volatility * 0.5)
            low_price = min(open_price, close_price) - rng.uniform(0, volatility * 0.5)
            volume = rng.uniform(10, 5000) * (price / 1000)

            candles.append({
                "open_time": open_time,
                "close_time": close_time,
                "open": round(open_price, 8),
                "high": round(high_price, 8),
                "low": round(low_price, 8),
                "close": round(close_price, 8),
                "volume": round(volume, 8),
                "trade_count": int(rng.randint(50, 2000)),
                "is_closed": True,
            })

            price = close_price
            open_time = close_time

        self._current_price = candles[-1]["close"] if candles else self._base_price
        return candles

    # -- market data: trades ------------------------------------------

    async def _fetch_trades_raw(
        self, symbol: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Generate simulated recent trades.

        Args:
            symbol: Trading pair, e.g. ``"BTC/USDT"``.
            limit: Number of trades.

        Returns:
            List of simulated trade dicts.
        """
        rng = random.Random(_make_seed(symbol, "trades"))
        now = datetime.now(UTC)
        trades: list[dict[str, Any]] = []
        price = self._current_price

        for i in range(limit):
            trade_time = now - timedelta(seconds=i * rng.uniform(1, 30))
            volatility = price * 0.0005
            change = rng.gauss(0, volatility)
            trade_price = max(price + change, price * 0.9)
            is_maker = rng.random() > 0.5
            trades.append({
                "trade_id": uuid.uuid4().hex[:12],
                "price": round(trade_price, 8),
                "quantity": round(rng.uniform(0.001, 10), 8),
                "side": "sell" if is_maker else "buy",
                "event_time": trade_time,
            })
            price = trade_price

        self._current_price = price
        return trades

    # -- market data: orderbook ---------------------------------------

    async def _fetch_orderbook_raw(
        self, symbol: str, limit: int = 20
    ) -> dict[str, Any]:
        """Generate simulated orderbook snapshot.

        Args:
            symbol: Trading pair, e.g. ``"BTC/USDT"``.
            limit: Depth per side.

        Returns:
            Dict with "bids" and "asks".
        """
        rng = random.Random(_make_seed(symbol, "orderbook"))
        price = self._current_price
        spread = self.config.fees.spread_price(price)
        half_spread = spread / 2

        bids: list[list[float]] = []
        asks: list[list[float]] = []
        step = price * 0.001  # 0.1% steps

        for i in range(limit):
            bid_price = price - half_spread - step * (i + 1)
            ask_price = price + half_spread + step * (i + 1)
            bids.append([
                round(max(bid_price, price * 0.5), 8),
                round(rng.uniform(0.01, 100), 8),
            ])
            asks.append([
                round(ask_price, 8),
                round(rng.uniform(0.01, 100), 8),
            ])

        self._ob_bids = bids
        self._ob_asks = asks
        return {"bids": bids, "asks": asks}

    # -- market data: ticker ------------------------------------------

    async def _fetch_ticker_raw(self, symbol: str) -> dict[str, Any]:
        """Generate simulated 24h ticker snapshot.

        Args:
            symbol: Trading pair, e.g. ``"BTC/USDT"``.

        Returns:
            Dict with price, volume, high, low, etc.
        """
        rng = random.Random(_make_seed(symbol, "ticker"))
        price = self._current_price
        spread = self.config.fees.spread_price(price)
        half_spread = spread / 2

        open_price = price * (1 + rng.uniform(-0.03, 0.03))
        high_price = max(price, open_price) * (1 + rng.uniform(0, 0.02))
        low_price = min(price, open_price) * (1 - rng.uniform(0, 0.02))
        volume = rng.uniform(1000, 500000) * (price / 1000)
        price_change = price - open_price
        price_change_pct = (price_change / open_price * 100) if open_price > 0 else 0

        return {
            "symbol": symbol,
            "price": round(price, 8),
            "bid": round(price - half_spread, 8),
            "ask": round(price + half_spread, 8),
            "high": round(high_price, 8),
            "low": round(low_price, 8),
            "volume": round(volume, 8),
            "quote_volume": round(volume * price, 2),
            "open": round(open_price, 8),
            "close": round(price, 8),
            "price_change_pct": round(price_change_pct, 4),
            "trade_count": int(rng.randint(10000, 500000)),
            "event_time": datetime.now(UTC),
        }

    # -- heartbeat hook -----------------------------------------------

    async def _send_heartbeat(self) -> None:
        """Dummy heartbeat — always succeeds."""
        self.logger.debug("DummyAdapter heartbeat OK.")

    # -- publish hook -------------------------------------------------

    def _publish_event(self, raw_event: dict[str, Any]) -> None:
        """Publish validated event (subclass hook)."""
        self.logger.debug("Published event: %s", raw_event.get("type", "unknown"))

    # -- utility ------------------------------------------------------

    def get_fee_structure(self) -> dict[str, Any]:
        """Return fee structure for this Dummy adapter.

        Returns:
            Dict with taker/maker rates and spread.
        """
        return {
            "venue": self.config.venue,
            "taker": self.config.fees.taker_rate,
            "maker": self.config.fees.maker_rate,
            "spread_bps": self.config.fees.spread_bps,
        }

    # -- internal helpers ---------------------------------------------

    @staticmethod
    def _interval_to_minutes(interval: str) -> int:
        """Convert an interval string to minutes."""
        if interval.endswith("d"):
            return int(interval[:-1]) * 1440
        if interval.endswith("h"):
            return int(interval[:-1]) * 60
        if interval.endswith("m"):
            return int(interval[:-1])
        if interval.endswith("w"):
            return int(interval[:-1]) * 10080
        return 1

    def _generate_orderbook(self) -> None:
        """Pre-generate a baseline orderbook."""
        rng = random.Random(self._seed)
        price = self._base_price
        for i in range(20):
            step = price * 0.001 * (i + 1)
            self._ob_bids.append([
                round(max(price - step, price * 0.5), 8),
                round(rng.uniform(0.01, 100), 8),
            ])
            self._ob_asks.append([
                round(price + step, 8),
                round(rng.uniform(0.01, 100), 8),
            ])