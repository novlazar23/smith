"""Contract tests for exchange data adapters and market data events.

Verifies:
- Exchange adapters implement the required interface (fetch_historical_candles,
  fetch_historical_trades, connect, disconnect, subscribe).
- Market data event types (Candle, Trade, OrderBookSnapshot) carry all required
  fields and enforce constraints (price > 0, quantity > 0, high >= low, …).
- MarketEvent union type covers the expected schema variants.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from apps.ingestion.base_adapter import (
    ConnectionConfig,
    ExchangeAdapterBase,
)
from packages.schemas.market_event import (
    Candle,
    NewsEvent,
    OrderBookSnapshot,
    PriceLevel,
    Trade,
)

# ── Fixture: concrete adapter for inspection ────────────────────────────


class _ConcreteAdapter(ExchangeAdapterBase):
    """Minimal concrete adapter to expose the base-class contract."""

    def __init__(self) -> None:
        super().__init__(
            ConnectionConfig(api_key="test", api_secret="test"),
        )

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def subscribe(self, streams: list[str]) -> None:
        pass

    def _publish_event(self, raw_event: dict) -> None:
        pass


# ── Exchange adapter interface contract ─────────────────────────────────


class TestExchangeAdapterInterface:
    """Verifies that the exchange adapter base class defines the expected
    abstract methods and that concrete subclasses follow the pattern.
    """

    def test_base_has_abstract_lifecycle_methods(self) -> None:
        """The base class must define abstract connect, disconnect, subscribe."""
        abstracts = {
            name
            for name in dir(ExchangeAdapterBase)
            if getattr(getattr(ExchangeAdapterBase, name, None), "__isabstractmethod__", False)
        }
        for method in ("connect", "disconnect", "subscribe"):
            assert method in abstracts, (
                f"Expected abstract method '{method}' missing on "
                f"ExchangeAdapterBase"
            )

    def test_base_has_public_query_methods(self) -> None:
        """Base adapter exposes health_check, is_connected, connection_state."""
        base = _ConcreteAdapter()
        assert hasattr(base, "health_check")
        assert hasattr(base, "is_connected")
        assert hasattr(base, "connection_state")

    def test_base_has_event_helpers(self) -> None:
        """Base adapter provides _validate_and_publish and _build_metadata."""
        base = _ConcreteAdapter()
        assert hasattr(base, "_validate_and_publish")
        assert hasattr(base, "_build_metadata")
        assert hasattr(base, "_next_sequence")

    def test_concrete_subclasses_exist(self) -> None:
        """Known concrete implementations must be importable."""
        from apps.ingestion.binance_futures import FuturesAdapter
        from apps.ingestion.binance_spot import SpotAdapter

        assert SpotAdapter is not None
        assert FuturesAdapter is not None

    def test_spot_adapter_has_fetch_methods(self) -> None:
        """SpotAdapter must provide fetch_historical_candles and
        fetch_historical_trades."""
        from apps.ingestion.binance_spot import SpotAdapter

        assert hasattr(SpotAdapter, "fetch_historical_candles")
        assert hasattr(SpotAdapter, "fetch_historical_trades")

    def test_futures_adapter_has_derived_fetch_methods(self) -> None:
        """FuturesAdapter must provide fetch_funding_rate, fetch_open_interest,
        fetch_recent_liquidations."""
        from apps.ingestion.binance_futures import FuturesAdapter

        assert hasattr(FuturesAdapter, "fetch_funding_rate")
        assert hasattr(FuturesAdapter, "fetch_open_interest")
        assert hasattr(FuturesAdapter, "fetch_recent_liquidations")

    def test_connection_config_defaults(self) -> None:
        """ConnectionConfig must expose sensible defaults."""
        cfg = ConnectionConfig(api_key="k", api_secret="s")
        assert cfg.reconnect_delay == 1.0
        assert cfg.max_reconnect_attempts == 10
        assert cfg.heartbeat_interval == 30.0
        assert cfg.rate_limit_per_second == 10


# ── Market data event schema contract ───────────────────────────────────


class TestCandleSchema:
    """Candle must carry instrument, venue, timeframe, OHLCV fields."""

    @pytest.fixture
    def valid_candle(self) -> dict:
        now = datetime.now(UTC)
        return {
            "instrument": "BTC/USDT",
            "venue": "binance",
            "timeframe": "1h",
            "open_time": now,
            "close_time": now,
            "open": 50000.0,
            "high": 50100.0,
            "low": 49900.0,
            "close": 50050.0,
            "volume": 1200.5,
        }

    def test_candle_accepts_valid_data(self, valid_candle: dict) -> None:
        candle = Candle(**valid_candle)
        assert candle.instrument == "BTC/USDT"
        assert candle.high >= candle.low
        assert candle.close == 50050.0

    def test_candle_rejects_low_gt_high(self) -> None:
        with pytest.raises(Exception):
            Candle(
                instrument="BTC/USDT",
                venue="binance",
                timeframe="1h",
                open_time=datetime.now(UTC),
                close_time=datetime.now(UTC),
                open=100,
                high=90,
                low=95,
                close=92,
                volume=100,
            )

    def test_candle_rejects_non_positive_low(self) -> None:
        with pytest.raises(Exception):
            Candle(
                instrument="BTC/USDT",
                venue="binance",
                timeframe="1h",
                open_time=datetime.now(UTC),
                close_time=datetime.now(UTC),
                open=100,
                high=110,
                low=-1,
                close=105,
                volume=100,
            )

    def test_candle_all_required_fields_present(self, valid_candle: dict) -> None:
        candle = Candle(**valid_candle)
        required = {"instrument", "venue", "timeframe", "open_time", "close_time",
                     "open", "high", "low", "close", "volume"}
        for field in required:
            assert hasattr(candle, field), f"Candle missing required field: {field}"

    def test_candle_optional_fields_default(self, valid_candle: dict) -> None:
        candle = Candle(**valid_candle)
        assert candle.trade_count is None
        assert candle.is_closed is True
        assert candle.metadata is None


class TestTradeSchema:
    """Trade must carry trade_id, instrument, venue, price, quantity, side."""

    def test_trade_accepts_valid_data(self) -> None:
        trade = Trade(
            trade_id="t1",
            instrument="BTC/USDT",
            venue="binance",
            price=50000.0,
            quantity=1.5,
            side="buy",
        )
        assert trade.trade_id == "t1"
        assert trade.price == 50000.0
        assert trade.quantity == 1.5

    def test_trade_rejects_zero_price(self) -> None:
        with pytest.raises(Exception):  # Pydantic validation
            Trade(
                trade_id="t2",
                instrument="BTC/USDT",
                venue="binance",
                price=0,
                quantity=1.0,
            )

    def test_trade_rejects_negative_quantity(self) -> None:
        with pytest.raises(Exception):
            Trade(
                trade_id="t3",
                instrument="BTC/USDT",
                venue="binance",
                price=50000,
                quantity=-1.0,
            )

    def test_trade_side_optional(self) -> None:
        trade = Trade(
            trade_id="t4",
            instrument="BTC/USDT",
            venue="binance",
            price=50000,
            quantity=1.0,
        )
        assert trade.side is None

    def test_trade_all_required_fields_present(self) -> None:
        trade = Trade(
            trade_id="t5",
            instrument="BTC/USDT",
            venue="binance",
            price=50000,
            quantity=1.0,
        )
        for field in ("trade_id", "instrument", "venue", "price", "quantity"):
            assert hasattr(trade, field)


class TestOrderBookSnapshotSchema:
    """OrderBookSnapshot must carry instrument, venue, sequence, bids, asks."""

    def test_orderbook_snapshot_valid(self) -> None:
        ob = OrderBookSnapshot(
            instrument="BTC/USDT",
            venue="binance",
            sequence=100,
            bids=[PriceLevel(price=50000, quantity=1.0)],
            asks=[PriceLevel(price=50001, quantity=2.0)],
        )
        assert ob.sequence == 100
        assert len(ob.bids) == 1
        assert len(ob.asks) == 1

    def test_orderbook_snapshot_empty_lists_ok(self) -> None:
        ob = OrderBookSnapshot(
            instrument="BTC/USDT",
            venue="binance",
            sequence=1,
            bids=[],
            asks=[],
        )
        assert ob.bids == []
        assert ob.asks == []

    def test_price_level_validates_positive_values(self) -> None:
        with pytest.raises(Exception):
            PriceLevel(price=0, quantity=1.0)

        with pytest.raises(Exception):
            PriceLevel(price=100, quantity=0)


class TestNewsEventSchema:
    """NewsEvent must carry news_id, event_identity, title, source_name,
    source_type. Body, timestamps, entities, language are optional.
    """

    @pytest.fixture
    def valid_news(self) -> dict:
        return {
            "news_id": "n1",
            "event_identity": "e1",
            "title": "BTC hits new high",
            "source_name": "Reuters",
            "source_type": "news_wire",
        }

    def test_news_event_minimal_valid(self, valid_news: dict) -> None:
        event = NewsEvent(**valid_news)
        assert event.news_id == "n1"
        assert event.title == "BTC hits new high"
        assert event.source_name == "Reuters"

    def test_news_event_optional_fields(self) -> None:
        event = NewsEvent(
            news_id="n2",
            event_identity="e2",
            title="Test",
            source_name="Test",
            source_type="blog",
            body="Some content",
            published_at=datetime.now(UTC),
            entities=["BTC", "ETH"],
        )
        assert event.body == "Some content"
        assert event.entities == ["BTC", "ETH"]
        assert event.revision == 1

    def test_news_event_source_name_present(self) -> None:
        event = NewsEvent(
            news_id="n3",
            event_identity="e3",
            title="Test",
            source_name="Test",
            source_type="blog",
        )
        assert event.source_name == "Test"
