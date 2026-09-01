"""Contract tests for event schemas — MarketEvent, Candle, Trade,
OrderBookSnapshot, NewsEvent, and their serialization.

Verifies:
- Event types from packages.schemas.market_event have all required fields.
- Event serialization (to_dict) and deserialization (from_dict) preserve data.
- Event type enums cover all expected variants.
- Domain market_data models are consistent with schema contracts.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from packages.domain.market_data import (
    CandleAggregation,
    FullOrderBook,
    FundingRate,
    Liquidation,
    MultiTimeframeAggregator,
    OpenInterest,
    OrderBookReconstructor,
    TradeAggregation,
    VolumeProfile,
)
from packages.domain.market_data.derivatives import LiquidationSide
from packages.schemas import market_event as market_event_pkg
from packages.schemas.market_event import (
    Candle,
    EventType,
    NewsEvent,
    OrderBookSnapshot,
    PriceLevel,
    Trade,
)
from packages.streaming.schemas import (
    Candle as StreamingCandle,
)
from packages.streaming.schemas import (
    MarketEvent as StreamingMarketEvent,
)
from packages.streaming.schemas import (
    NewsEvent as StreamingNewsEvent,
)
from packages.streaming.schemas import (
    OrderBookSnapshot as StreamingOrderBookSnapshot,
)
from packages.streaming.schemas import (
    SourceMetadata,
)
from packages.streaming.schemas import (
    Trade as StreamingTrade,
)

# ── EventType enum contract ─────────────────────────────────────────────


class TestEventType:
    """EventType must cover all expected market event categories."""

    def test_expected_event_types(self) -> None:
        expected = {
            "candle", "trade", "orderbook_snapshot", "orderbook_delta",
            "funding_rate", "open_interest", "liquidation", "news",
        }
        actual = {e.value for e in EventType}
        assert expected == actual, f"Expected {expected}, got {actual}"

    def test_event_type_is_str_enum(self) -> None:
        assert isinstance(EventType.CANDLE, str)
        assert EventType.CANDLE == "candle"


# ── Candle serialization contract ──────────────────────────────────────


class TestCandleSerialization:
    """Candle to_dict / from_dict must round-trip correctly."""

    @pytest.fixture
    def sample_candle(self) -> dict:
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
            "metadata": {"source": "ws"},
        }

    def test_candle_required_fields(self, sample_candle: dict) -> None:
        candle = Candle(**sample_candle)
        for field in ("instrument", "venue", "timeframe", "open_time",
                       "close_time", "open", "high", "low", "close", "volume"):
            assert hasattr(candle, field)

    def test_candle_roundtrip(self, sample_candle: dict) -> None:
        candle = Candle(**sample_candle)
        d = candle.model_dump()
        restored = Candle(**d)
        assert restored.instrument == candle.instrument
        assert restored.venue == candle.venue
        assert restored.high == candle.high
        assert restored.low == candle.low


# ── Trade serialization contract ───────────────────────────────────────


class TestTradeSerialization:
    """Trade to_dict / from_dict must round-trip correctly."""

    def test_trade_roundtrip(self) -> None:
        trade = Trade(
            trade_id="t1",
            instrument="BTC/USDT",
            venue="binance",
            price=50000.0,
            quantity=1.5,
            side="buy",
            metadata={"exchange_seq": 12345},
        )
        d = trade.model_dump()
        restored = Trade(**d)
        assert restored.trade_id == "t1"
        assert restored.price == 50000.0
        assert restored.quantity == 1.5
        assert restored.side == "buy"
        assert restored.metadata == {"exchange_seq": 12345}

    def test_trade_frozen(self) -> None:
        trade = Trade(
            trade_id="t2", instrument="ETH/USDT", venue="binance",
            price=3000.0, quantity=10.0,
        )
        assert trade.model_config.get("frozen") is True


# ── OrderBookSnapshot serialization contract ───────────────────────────


class TestOrderBookSnapshotSerialization:
    """OrderBookSnapshot to_dict / from_dict must round-trip correctly."""

    def test_orderbook_roundtrip(self) -> None:
        ob = OrderBookSnapshot(
            instrument="BTC/USDT",
            venue="binance",
            sequence=100,
            bids=[PriceLevel(price=50000, quantity=1.0)],
            asks=[PriceLevel(price=50001, quantity=2.0)],
            metadata={"level": "full"},
        )
        d = ob.model_dump()
        restored = OrderBookSnapshot(**d)
        assert restored.instrument == "BTC/USDT"
        assert restored.sequence == 100
        assert len(restored.bids) == 1
        assert len(restored.asks) == 1

    def test_orderbook_bids_asks_present(self) -> None:
        ob = OrderBookSnapshot(
            instrument="BTC/USDT",
            venue="binance",
            sequence=1,
            bids=[PriceLevel(price=50000, quantity=1.0)],
            asks=[PriceLevel(price=50001, quantity=2.0)],
        )
        assert hasattr(ob, "bids")
        assert hasattr(ob, "asks")
        assert hasattr(ob, "sequence")
        assert hasattr(ob, "instrument")
        assert hasattr(ob, "venue")


# ── NewsEvent serialization contract ───────────────────────────────────


class TestNewsEventSerialization:
    """NewsEvent to_dict / from_dict must round-trip correctly."""

    def test_news_event_roundtrip(self) -> None:
        now = datetime.now(UTC)
        event = NewsEvent(
            news_id="n1",
            event_identity="e1",
            title="Fed raises rates",
            source_name="Reuters",
            source_type="wire",
            body="The Federal Reserve announced...",
            published_at=now,
            received_at=now,
            entities=["USD", "FED"],
            instruments=["US10Y"],
            language="en",
        )
        d = event.model_dump()
        restored = NewsEvent(**d)
        assert restored.news_id == "n1"
        assert restored.title == "Fed raises rates"
        assert restored.source_name == "Reuters"
        assert restored.entities == ["USD", "FED"]


# ── Streaming schema serialization contract ────────────────────────────


class TestStreamingSchemaSerialization:
    """Streaming schemas must support to_dict / from_dict round-trips."""

    def test_streaming_market_event_roundtrip(self) -> None:
        meta = SourceMetadata(
            source="binance", venue="binance",
            event_time=datetime.now(UTC),
        )
        me = StreamingMarketEvent(
            event_id="me1",
            event_type="trade",
            instrument="BTC/USDT",
            metadata=meta,
            payload={"price": 50000},
        )
        d = me.to_dict()
        restored = StreamingMarketEvent.from_dict(d)
        assert restored.event_id == "me1"
        assert restored.instrument == "BTC/USDT"

    def test_streaming_candle_roundtrip(self) -> None:
        candle = StreamingCandle(
            instrument="BTC/USDT",
            venue="binance",
            timeframe="1h",
            open_time=datetime.now(UTC),
            close_time=datetime.now(UTC),
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,
            volume=1200.5,
        )
        d = candle.to_dict()
        restored = StreamingCandle.from_dict(d)
        assert restored.instrument == "BTC/USDT"
        assert restored.high == 50100.0

    def test_streaming_trade_roundtrip(self) -> None:
        trade = StreamingTrade(
            trade_id="t1",
            instrument="BTC/USDT",
            venue="binance",
            price=50000.0,
            quantity=1.5,
            side="buy",
        )
        d = trade.to_dict()
        restored = StreamingTrade.from_dict(d)
        assert restored.trade_id == "t1"
        assert restored.price == 50000.0

    def test_streaming_orderbook_roundtrip(self) -> None:
        ob = StreamingOrderBookSnapshot(
            instrument="BTC/USDT",
            venue="binance",
            sequence=100,
            bids=[(50000.0, 1.0)],
            asks=[(50001.0, 2.0)],
        )
        d = ob.to_dict()
        restored = StreamingOrderBookSnapshot.from_dict(d)
        assert restored.sequence == 100
        assert len(restored.bids) == 1

    def test_streaming_news_roundtrip(self) -> None:
        now = datetime.now(UTC)
        event = StreamingNewsEvent(
            news_id="n1",
            event_identity="e1",
            title="Test",
            body="Content",
            source_name="Test",
            source_type="blog",
            url_hash="h",
            published_at=now,
            received_at=now,
        )
        d = event.to_dict()
        restored = StreamingNewsEvent.from_dict(d)
        assert restored.news_id == "n1"
        assert restored.title == "Test"

    def test_source_metadata_roundtrip(self) -> None:
        meta = SourceMetadata(
            source="binance", venue="binance",
            event_time=datetime.now(UTC),
            sequence=42, quality=0.95,
        )
        d = meta.to_dict()
        restored = SourceMetadata.from_dict(d)
        assert restored.source == "binance"
        assert restored.sequence == 42
        assert restored.quality == 0.95


# ── MarketEvent union type contract ─────────────────────────────────────


class TestMarketEventUnion:
    """MarketEvent type alias must cover the expected event types."""

    def test_market_event_type_alias_defined(self) -> None:
        """MarketEvent should be defined in the module."""
        assert hasattr(market_event_pkg, "MarketEvent")

    def test_market_event_type_has_components(self) -> None:
        """All components of the MarketEvent union should be importable."""
        from packages.schemas.market_event import (
            Candle as ME_Candle,
        )
        from packages.schemas.market_event import (
            NewsEvent as ME_News,
        )
        from packages.schemas.market_event import (
            OrderBookSnapshot as ME_OrderBookSnapshot,
        )
        from packages.schemas.market_event import (
            Trade as ME_Trade,
        )
        # Verify classes are the correct types
        assert ME_Candle.__bases__[0].__name__ == "BaseModel"
        assert ME_Trade.__bases__[0].__name__ == "BaseModel"
        assert ME_News.__bases__[0].__name__ == "BaseModel"
        assert ME_OrderBookSnapshot.__bases__[0].__name__ == "BaseModel"


# ── Domain market_data models contract ──────────────────────────────────


class TestDomainMarketDataModels:
    """Domain-level market_data models must be importable and have required
    fields/methods.
    """

    def test_candle_aggregation_fields(self) -> None:
        """CandleAggregation(instrument, venue, timeframe, open_time,
        close_time, open, high, low, close, volume, trade_count)."""
        now = datetime.now(UTC)
        ca = CandleAggregation(
            instrument="BTC/USDT",
            venue="binance",
            timeframe="1h",
            open_time=now,
            close_time=now,
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,
            volume=1200.5,
            trade_count=100,
        )
        assert ca.instrument == "BTC/USDT"
        assert ca.timeframe == "1h"
        assert ca.high == 50100.0
        assert ca.trade_count == 100

    def test_full_orderbook_fields(self) -> None:
        """FullOrderBook must carry instrument, venue, sequence, bids, asks."""
        ob = FullOrderBook(
            instrument="BTC/USDT",
            venue="binance",
            sequence=100,
            bids=[PriceLevel(price=50000, quantity=1.0)],
            asks=[PriceLevel(price=50001, quantity=2.0)],
            event_time=datetime.now(UTC),
        )
        assert ob.instrument == "BTC/USDT"
        assert hasattr(ob, "best_bid")
        assert hasattr(ob, "best_ask")
        assert hasattr(ob, "spread")
        assert ob.best_bid == 50000.0
        assert ob.best_ask == 50001.0
        assert ob.spread == 1.0

    def test_funding_rate_fields(self) -> None:
        """FundingRate requires instrument, funding_rate, mark_price,
        next_funding_time, event_time."""
        now = datetime.now(UTC)
        fr = FundingRate(
            instrument="BTC/USDT",
            venue="binance",
            funding_rate=0.0001,
            mark_price=50000.0,
            next_funding_time=now,
            event_time=now,
        )
        assert fr.instrument == "BTC/USDT"
        assert fr.funding_rate == 0.0001
        assert fr.funding_rate_pct == 0.01

    def test_liquidation_fields(self) -> None:
        """Liquidation requires instrument, side(LiquidationSide), quantity,
        price, value, event_time."""
        now = datetime.now(UTC)
        liq = Liquidation(
            instrument="BTC/USDT",
            venue="binance",
            side=LiquidationSide.LONG,
            quantity=1.5,
            price=49000.0,
            value=73500.0,
            event_time=now,
        )
        assert liq.side == LiquidationSide.LONG
        assert liq.quantity == 1.5
        assert liq.value == 73500.0

    def test_open_interest_fields(self) -> None:
        """OpenInterest requires instrument, open_interest, event_time."""
        now = datetime.now(UTC)
        oi = OpenInterest(
            instrument="BTC/USDT",
            venue="binance",
            open_interest=50000.0,
            event_time=now,
        )
        assert oi.open_interest == 50000.0

    def test_volume_profile_fields(self) -> None:
        """VolumeProfile requires instrument, venue, start_time, end_time,
        price_levels (default [])."""
        now = datetime.now(UTC)
        vp = VolumeProfile(
            instrument="BTC/USDT",
            venue="binance",
            start_time=now,
            end_time=now,
        )
        assert vp.instrument == "BTC/USDT"
        assert vp.price_levels == []
        assert vp.total_volume == 0.0

    def test_trade_aggregation_fields(self) -> None:
        """TradeAggregation requires instrument, venue, start_time, end_time,
        trade_count, total_volume, total_value, avg_price."""
        now = datetime.now(UTC)
        ta = TradeAggregation(
            instrument="BTC/USDT",
            venue="binance",
            start_time=now,
            end_time=now,
            trade_count=50,
            total_volume=100.0,
            total_value=5000000.0,
            avg_price=50000.0,
        )
        assert ta.instrument == "BTC/USDT"
        assert ta.trade_count == 50
        assert ta.total_volume == 100.0

    def test_orderbook_reconstructor_init(self) -> None:
        """OrderBookReconstructor(instrument, venue) must be instantiable."""
        recon = OrderBookReconstructor(
            instrument="BTC/USDT", venue="binance",
        )
        assert recon._instrument == "BTC/USDT"
        assert recon._venue == "binance"

    def test_multi_timeframe_aggregator_init(self) -> None:
        """MultiTimeframeAggregator(base_timeframe) must be instantiable."""
        mta = MultiTimeframeAggregator(base_timeframe="1m")
        assert mta._base_tf == "1m"
