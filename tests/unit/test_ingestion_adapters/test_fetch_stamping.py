"""Regressionstests: Stamping vor Validierung in ExchangeAdapterBase.

Bug: ``fetch_candles`` / ``fetch_trades`` stempelten ``type``,
``instrument`` und ``venue`` erst *nach* dem Aufruf des
``MarketDataValidator``. Da der Validator auf ``type`` dispatcht,
wurde jede Roh-Kerze / jeder Roh-Trade als "Unknown event type"
abgelehnt und beide Methoden lieferten immer eine leere Liste.

Diese Tests sichern: gültige Roh-Events werden mit gestempelten
Metadaten zurückgegeben, ungültige weiterhin verworfen.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from packages.ingestion.adapter.base import (
    ConnectionConfig,
    ConnectionState,
    ExchangeAdapterBase,
)

SYMBOL = "BTC/USDT"
OPEN_TIME = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
CLOSE_TIME = datetime(2026, 1, 1, 12, 1, 0, tzinfo=UTC)


def _raw_candle(**overrides: Any) -> dict[str, Any]:
    """Konstruiert eine gültige Roh-Kerze ohne type/instrument/venue."""
    candle: dict[str, Any] = {
        "open_time": OPEN_TIME,
        "close_time": CLOSE_TIME,
        "open": 100.0,
        "high": 110.0,
        "low": 95.0,
        "close": 105.0,
        "volume": 50.0,
        "trade_count": 100,
        "is_closed": True,
    }
    candle.update(overrides)
    return candle


def _raw_trade(**overrides: Any) -> dict[str, Any]:
    """Konstruiert einen gültigen Roh-Trade ohne type/instrument/venue."""
    trade: dict[str, Any] = {
        "trade_id": "t-123",
        "price": 100.0,
        "quantity": 1.5,
        "side": "buy",
        "event_time": OPEN_TIME,
    }
    trade.update(overrides)
    return trade


class _FetchTestAdapter(ExchangeAdapterBase):
    """Konkreter Test-Adapter mit vorgegebenen Roh-Events."""

    def __init__(
        self,
        raw_candles: list[dict[str, Any]],
        raw_trades: list[dict[str, Any]],
        venue: str = "TEST_VENUE",
    ) -> None:
        super().__init__(ConnectionConfig(venue=venue))
        self._raw_candles = raw_candles
        self._raw_trades = raw_trades

    async def connect(self) -> None:
        """Test-Hook: markiert Adapter als verbunden."""
        self._state = ConnectionState.CONNECTED

    async def disconnect(self) -> None:
        """Test-Hook: markiert Adapter als getrennt."""
        self._state = ConnectionState.DISCONNECTED

    async def subscribe(self, streams: list[str]) -> None:
        """Test-Hook: ignoriert Streams."""
        del streams

    async def _fetch_candles_raw(
        self, symbol: str, interval: str, limit: int
    ) -> list[dict[str, Any]]:
        """Liefert die vorgegebenen Roh-Kerzen."""
        del symbol, interval, limit
        return self._raw_candles

    async def _fetch_trades_raw(self, symbol: str, limit: int) -> list[dict[str, Any]]:
        """Liefert die vorgegebenen Roh-Trades."""
        del symbol, limit
        return self._raw_trades

    async def _fetch_orderbook_raw(self, symbol: str, limit: int) -> dict[str, Any]:
        """Test-Hook: leeres Orderbook."""
        del symbol, limit
        return {"bids": [], "asks": []}

    async def _fetch_ticker_raw(self, symbol: str) -> dict[str, Any]:
        """Test-Hook: leerer Ticker."""
        del symbol
        return {}


# ══════════════════════════════════════════════════════════════════
# fetch_candles
# ══════════════════════════════════════════════════════════════════


class TestFetchCandlesStamping:
    @pytest.mark.asyncio
    async def test_valid_raw_candle_is_returned_with_stamps(self) -> None:
        """Gültige Roh-Kerze kommt zurück, gestempelt mit type/instrument/venue."""
        adapter = _FetchTestAdapter(raw_candles=[_raw_candle()], raw_trades=[])

        result = await adapter.fetch_candles(SYMBOL, "1m", 1)

        assert len(result) == 1
        candle = result[0]
        assert candle["type"] == "candle"
        assert candle["instrument"] == SYMBOL
        assert candle["venue"] == adapter.config.venue
        # Payload bleibt unverändert erhalten
        assert candle["open"] == 100.0
        assert candle["high"] == 110.0
        assert candle["low"] == 95.0
        assert candle["close"] == 105.0
        assert candle["volume"] == 50.0

    @pytest.mark.asyncio
    async def test_existing_instrument_is_preserved(self) -> None:
        """Bereits vorhandener instrument-Wert wird nicht überschrieben."""
        adapter = _FetchTestAdapter(raw_candles=[_raw_candle(instrument="KEEPME")], raw_trades=[])

        result = await adapter.fetch_candles(SYMBOL, "1m", 1)

        assert len(result) == 1
        assert result[0]["instrument"] == "KEEPME"

    @pytest.mark.asyncio
    async def test_invalid_raw_candle_is_rejected(self) -> None:
        """Ungültige Kerze (high < low) wird weiterhin verworfen."""
        adapter = _FetchTestAdapter(raw_candles=[_raw_candle(high=90.0, low=95.0)], raw_trades=[])

        assert await adapter.fetch_candles(SYMBOL, "1m", 1) == []

    @pytest.mark.asyncio
    async def test_empty_fetch_returns_empty(self) -> None:
        """Leeres Fetch-Ergebnis → leere Liste."""
        adapter = _FetchTestAdapter(raw_candles=[], raw_trades=[])

        assert await adapter.fetch_candles(SYMBOL, "1m", 1) == []


# ══════════════════════════════════════════════════════════════════
# fetch_trades
# ══════════════════════════════════════════════════════════════════


class TestFetchTradesStamping:
    @pytest.mark.asyncio
    async def test_valid_raw_trade_is_returned_with_stamps(self) -> None:
        """Gültiger Roh-Trade kommt zurück, gestempelt mit type/instrument/venue."""
        adapter = _FetchTestAdapter(raw_candles=[], raw_trades=[_raw_trade()])

        result = await adapter.fetch_trades(SYMBOL, limit=1)

        assert len(result) == 1
        trade = result[0]
        assert trade["type"] == "trade"
        assert trade["instrument"] == SYMBOL
        assert trade["venue"] == adapter.config.venue
        # Payload bleibt unverändert erhalten
        assert trade["trade_id"] == "t-123"
        assert trade["price"] == 100.0
        assert trade["quantity"] == 1.5

    @pytest.mark.asyncio
    async def test_invalid_raw_trade_is_rejected(self) -> None:
        """Ungültiger Trade (fehlende trade_id) wird weiterhin verworfen."""
        adapter = _FetchTestAdapter(raw_candles=[], raw_trades=[_raw_trade(trade_id="")])

        assert await adapter.fetch_trades(SYMBOL, limit=1) == []

    @pytest.mark.asyncio
    async def test_empty_fetch_returns_empty(self) -> None:
        """Leeres Fetch-Ergebnis → leere Liste."""
        adapter = _FetchTestAdapter(raw_candles=[], raw_trades=[])

        assert await adapter.fetch_trades(SYMBOL, limit=1) == []
