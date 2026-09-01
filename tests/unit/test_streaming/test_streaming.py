"""Tests für Event Streaming Layer (packages/streaming/)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from packages.streaming.base import (
    CompressionType,
    Envelope,
    EventPartitionKey,
    StreamConfig,
)
from packages.streaming.redpanda import (
    EventSerializer,
    RedpandaConsumer,
    RedpandaDeadLetterHandler,
    RedpandaProducer,
)
from packages.streaming.schemas import (
    Candle,
    MarketEvent,
    NewsEvent,
    NewsStatus,
    OrderBookSnapshot,
    SourceMetadata,
    Trade,
)
from packages.streaming.topics import TopicRegistry

# ── StreamConfig ────────────────────────────────────────────────────


class TestStreamConfig:
    def test_default_config(self) -> None:
        config = StreamConfig()
        assert config.bootstrap_servers == "localhost:9092"
        assert config.default_topic == "trading-events"
        assert config.compression == CompressionType.NONE
        assert config.acks == "all"
        assert config.consumer_group == "trading-orchestra"
        assert config.dlq_topic == "trading-events-dlq"

    def test_custom_config(self) -> None:
        config = StreamConfig(
            bootstrap_servers="redpanda:9092",
            default_topic="custom-events",
            compression=CompressionType.LZ4,
            acks="1",
            max_in_flight=50,
        )
        assert config.bootstrap_servers == "redpanda:9092"
        assert config.default_topic == "custom-events"
        assert config.compression == CompressionType.LZ4
        assert config.acks == "1"
        assert config.max_in_flight == 50


class TestEventPartitionKey:
    def test_values(self) -> None:
        assert EventPartitionKey.INSTRUMENT == "instrument"
        assert EventPartitionKey.VENUE == "venue"
        assert EventPartitionKey.RUN_ID == "run_id"
        assert EventPartitionKey.NEWS_ID == "news_id"
        assert EventPartitionKey.NONE == "none"


# ── Envelope ────────────────────────────────────────────────────────


class TestEnvelope:
    def test_default_envelope(self) -> None:
        env = Envelope(event_type="candle", source="binance", payload={"open": 50000})
        assert env.event_type == "candle"
        assert env.source == "binance"
        assert env.payload == {"open": 50000}
        assert env.schema_version == "1.0.0"
        assert isinstance(env.event_id, str)
        assert isinstance(env.timestamp, datetime)

    def test_envelope_with_headers(self) -> None:
        env = Envelope(
            event_type="trade",
            source="binance",
            payload={"price": 50000},
            headers={"correlation_id": "abc-123"},
        )
        assert env.headers["correlation_id"] == "abc-123"


# ── SourceMetadata ─────────────────────────────────────────────────


class TestSourceMetadata:
    def test_valid_metadata(self) -> None:
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        sm = SourceMetadata(source="binance", venue="binance", event_time=ts)
        assert sm.source == "binance"
        assert sm.quality == 1.0
        assert sm.revision == 1

    def test_quality_validation_high(self) -> None:
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        sm = SourceMetadata(source="test", venue="test", event_time=ts, quality=0.95)
        assert sm.quality == 0.95

    def test_quality_validation_zero(self) -> None:
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        sm = SourceMetadata(source="test", venue="test", event_time=ts, quality=0.0)
        assert sm.quality == 0.0

    def test_quality_out_of_range_high(self) -> None:
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        with pytest.raises(ValueError, match="quality must be in"):
            SourceMetadata(source="test", venue="test", event_time=ts, quality=1.1)

    def test_quality_out_of_range_low(self) -> None:
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        with pytest.raises(ValueError, match="quality must be in"):
            SourceMetadata(source="test", venue="test", event_time=ts, quality=-0.1)

    def test_negative_sequence(self) -> None:
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        with pytest.raises(ValueError, match="sequence must be"):
            SourceMetadata(source="test", venue="test", event_time=ts, sequence=-1)

    def test_revision_less_than_one(self) -> None:
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        with pytest.raises(ValueError, match="revision must be"):
            SourceMetadata(source="test", venue="test", event_time=ts, revision=0)

    def test_to_dict(self) -> None:
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        sm = SourceMetadata(source="binance", venue="binance", event_time=ts)
        d = sm.to_dict()
        assert d["source"] == "binance"
        assert d["quality"] == 1.0
        assert isinstance(d["event_time"], str)

    def test_from_dict(self) -> None:
        data = {
            "source": "binance",
            "venue": "binance",
            "event_time": "2024-01-01T00:00:00+00:00",
            "quality": 0.99,
            "sequence": 42,
            "revision": 2,
        }
        sm = SourceMetadata.from_dict(data)
        assert sm.source == "binance"
        assert sm.quality == 0.99
        assert sm.sequence == 42
        assert sm.revision == 2


# ── Candle ──────────────────────────────────────────────────────────


class TestCandle:
    def test_valid_candle(self) -> None:
        open_t = datetime(2024, 1, 1, tzinfo=UTC)
        close_t = datetime(2024, 1, 2, tzinfo=UTC)
        c = Candle(
            instrument="BTC", venue="binance", timeframe="15m",
            open_time=open_t, close_time=close_t,
            open=50000, high=51000, low=49000, close=50500, volume=100.0,
        )
        assert c.instrument == "BTC"
        assert c.high == 51000
        assert c.low == 49000

    def test_high_below_low(self) -> None:
        open_t = datetime(2024, 1, 1, tzinfo=UTC)
        close_t = datetime(2024, 1, 2, tzinfo=UTC)
        with pytest.raises(ValueError, match="high.*<.*low"):
            Candle(instrument="BTC", venue="binance", timeframe="15m",
                   open_time=open_t, close_time=close_t,
                   open=50000, high=49000, low=51000, close=50000, volume=100.0)

    def test_negative_price(self) -> None:
        open_t = datetime(2024, 1, 1, tzinfo=UTC)
        close_t = datetime(2024, 1, 2, tzinfo=UTC)
        with pytest.raises(ValueError, match="OHLC values must be"):
            Candle(instrument="BTC", venue="binance", timeframe="15m",
                   open_time=open_t, close_time=close_t,
                   open=-100, high=51000, low=49000, close=50000, volume=100.0)

    def test_negative_volume(self) -> None:
        open_t = datetime(2024, 1, 1, tzinfo=UTC)
        close_t = datetime(2024, 1, 2, tzinfo=UTC)
        with pytest.raises(ValueError, match="volume must be"):
            Candle(instrument="BTC", venue="binance", timeframe="15m",
                   open_time=open_t, close_time=close_t,
                   open=50000, high=51000, low=49000, close=50000, volume=-1.0)

    def test_to_dict(self) -> None:
        open_t = datetime(2024, 1, 1, tzinfo=UTC)
        close_t = datetime(2024, 1, 2, tzinfo=UTC)
        c = Candle(instrument="BTC", venue="binance", timeframe="15m",
                   open_time=open_t, close_time=close_t,
                   open=50000, high=51000, low=49000, close=50000, volume=100.0)
        d = c.to_dict()
        assert d["instrument"] == "BTC"
        assert isinstance(d["open_time"], str)

    def test_from_dict(self) -> None:
        data = {
            "instrument": "BTC", "venue": "binance", "timeframe": "15m",
            "open_time": "2024-01-01T00:00:00+00:00",
            "close_time": "2024-01-02T00:00:00+00:00",
            "open": 50000, "high": 51000, "low": 49000, "close": 50000,
            "volume": 100.0, "is_closed": True,
        }
        c = Candle.from_dict(data)
        assert c.instrument == "BTC"
        assert c.volume == 100.0


# ── Trade ───────────────────────────────────────────────────────────


class TestTrade:
    def test_valid_trade(self) -> None:
        t = Trade(
            trade_id="t1", instrument="BTC", venue="binance",
            price=50000, quantity=1.0, side="buy",
        )
        assert t.price == 50000
        assert t.quantity == 1.0
        assert t.side == "buy"

    def test_zero_price(self) -> None:
        with pytest.raises(ValueError, match="price must be"):
            Trade(trade_id="t1", instrument="BTC", venue="binance",
                  price=0, quantity=1.0, side="buy")

    def test_zero_quantity(self) -> None:
        with pytest.raises(ValueError, match="quantity must be"):
            Trade(trade_id="t1", instrument="BTC", venue="binance",
                  price=50000, quantity=0, side="buy")

    def test_invalid_side(self) -> None:
        with pytest.raises(ValueError, match="side must be"):
            Trade(trade_id="t1", instrument="BTC", venue="binance",
                  price=50000, quantity=1.0, side="unknown")


# ── OrderBookSnapshot ───────────────────────────────────────────────


class TestOrderBookSnapshot:
    def test_valid_snapshot(self) -> None:
        ob = OrderBookSnapshot(
            instrument="BTC", venue="binance", sequence=100,
            bids=[[50000, 1.0], [49999, 2.0]],
            asks=[[50001, 1.5], [50002, 3.0]],
        )
        assert ob.sequence == 100
        assert len(ob.bids) == 2
        assert len(ob.asks) == 2

    def test_inverted_spread(self) -> None:
        with pytest.raises(ValueError, match="Best bid.*>=.*best ask"):
            OrderBookSnapshot(
                instrument="BTC", venue="binance", sequence=1,
                bids=[[50001, 1.0]],
                asks=[[50000, 1.0]],
            )

    def test_empty_orderbook_valid(self) -> None:
        ob = OrderBookSnapshot(instrument="BTC", venue="binance", sequence=1)
        assert ob.bids == []
        assert ob.asks == []


# ── NewsEvent ───────────────────────────────────────────────────────


class TestNewsEvent:
    def test_valid_news(self) -> None:
        pub = datetime(2024, 1, 1, tzinfo=UTC)
        rec = datetime(2024, 1, 1, tzinfo=UTC)
        ne = NewsEvent(
            news_id="n1", event_identity="e1", title="Test",
            body="Body text", source_name="Bloomberg",
            source_type="news_wire", url_hash="abc",
            published_at=pub, received_at=rec,
        )
        assert ne.title == "Test"
        assert ne.status == NewsStatus.INITIAL
        assert ne.revision == 1

    def test_news_status_values(self) -> None:
        for status in NewsStatus:
            assert status in (
                NewsStatus.RUMOR, NewsStatus.INITIAL, NewsStatus.CONFIRMATION,
                NewsStatus.UPDATE, NewsStatus.CORRECTION, NewsStatus.RETRACTION,
            )

    def test_to_dict(self) -> None:
        pub = datetime(2024, 1, 1, tzinfo=UTC)
        rec = datetime(2024, 1, 1, tzinfo=UTC)
        ne = NewsEvent(
            news_id="n1", event_identity="e1", title="Test", body="",
            source_name="test", source_type="web", url_hash="x",
            published_at=pub, received_at=rec,
        )
        d = ne.to_dict()
        assert d["news_id"] == "n1"
        assert isinstance(d["published_at"], str)


# ── MarketEvent ─────────────────────────────────────────────────────


class TestMarketEvent:
    def test_candle_event(self) -> None:
        open_t = datetime(2024, 1, 1, tzinfo=UTC)
        close_t = datetime(2024, 1, 2, tzinfo=UTC)
        sm = SourceMetadata(source="binance", venue="binance", event_time=open_t)
        candle = Candle(
            instrument="BTC", venue="binance", timeframe="15m",
            open_time=open_t, close_time=close_t,
            open=50000, high=51000, low=49000, close=50500, volume=100.0,
        )
        evt = MarketEvent(
            event_id="evt-1", event_type="candle", instrument="BTC",
            metadata=sm, payload=candle.to_dict(),
        )
        assert evt.event_type == "candle"
        assert evt.instrument == "BTC"
        d = evt.to_dict()
        assert "metadata" in d

    def test_from_dict(self) -> None:
        open_t = datetime(2024, 1, 1, tzinfo=UTC)
        data = {
            "event_id": "e1", "event_type": "candle", "instrument": "BTC",
            "metadata": {
                "source": "binance", "venue": "binance",
                "event_time": "2024-01-01T00:00:00+00:00",
            },
            "payload": {"open": 50000},
        }
        evt = MarketEvent.from_dict(data)
        assert evt.instrument == "BTC"


# ── EventSerializer ─────────────────────────────────────────────────


class TestEventSerializer:
    def test_serialize_candle(self) -> None:
        open_t = datetime(2024, 1, 1, tzinfo=UTC)
        close_t = datetime(2024, 1, 2, tzinfo=UTC)
        c = Candle(instrument="BTC", venue="binance", timeframe="15m",
                   open_time=open_t, close_time=close_t,
                   open=50000, high=51000, low=49000, close=50000, volume=100.0)
        d = EventSerializer.serialize(c)
        assert "open" in d
        assert d["open"] == 50000

    def test_serialize_dict(self) -> None:
        d = EventSerializer.serialize({"key": "value"})
        assert d == {"key": "value"}

    def test_deserialize_known_type(self) -> None:
        open_t = datetime(2024, 1, 1, tzinfo=UTC)
        close_t = datetime(2024, 1, 2, tzinfo=UTC)
        data = {
            "instrument": "BTC", "venue": "binance", "timeframe": "15m",
            "open_time": open_t.isoformat(), "close_time": close_t.isoformat(),
            "open": 50000, "high": 51000, "low": 49000, "close": 50000,
            "volume": 100.0,
        }
        c = EventSerializer.deserialize(data, "Candle")
        assert isinstance(c, Candle)
        assert c.instrument == "BTC"

    def test_deserialize_unknown_type(self) -> None:
        data = {"key": "value"}
        result = EventSerializer.deserialize(data, "UnknownType")
        assert result == data

    def test_json_roundtrip(self) -> None:
        payload = {"instrument": "BTC", "price": 50000}
        json_str = EventSerializer.to_json(payload)
        decoded = EventSerializer.from_json(json_str)
        assert decoded == payload


# ── TopicRegistry ───────────────────────────────────────────────────


class TestTopicRegistry:
    def test_all_topics_defined(self) -> None:
        topics = TopicRegistry.list_topics()
        assert len(topics) == 7

    def test_event_to_topic_mapping(self) -> None:
        assert TopicRegistry.get_topic_for_event("MarketEvent") == TopicRegistry.EVENTS
        assert TopicRegistry.get_topic_for_event("Candle") == TopicRegistry.FEATURES
        assert TopicRegistry.get_topic_for_event("NewsEvent") == TopicRegistry.EVENTS
        assert TopicRegistry.get_topic_for_event("FinalDecision") == TopicRegistry.DECISIONS

    def test_unknown_event_type_defaults_to_events(self) -> None:
        topic = TopicRegistry.get_topic_for_event("UnknownType")
        assert topic == TopicRegistry.EVENTS

    def test_dlq_topic_tag(self) -> None:
        assert TopicRegistry.DLQ.dlq is True
        assert TopicRegistry.EVENTS.dlq is False

    def test_topic_name_for_event(self) -> None:
        name = TopicRegistry.get_topic_name_for_event("MarketEvent")
        assert name == "trading-events"


# ── RedpandaProducer ────────────────────────────────────────────────


class TestRedpandaProducer:
    def test_initial_state(self) -> None:
        producer = RedpandaProducer()
        assert producer.is_connected() is False

    @pytest.mark.asyncio
    async def test_send_event(self) -> None:
        producer = RedpandaProducer()
        result = await producer.send("trading-events", {"instrument": "BTC", "price": 50000})
        assert result is True
        assert producer.is_connected() is True

    @pytest.mark.asyncio
    async def test_send_batch(self) -> None:
        producer = RedpandaProducer()
        events = [
            {"event_id": "1", "price": 50000},
            {"event_id": "2", "price": 50001},
        ]
        count = await producer.send_batch("trading-events", events, keys=["BTC", "BTC"])
        assert count == 2

    @pytest.mark.asyncio
    async def test_health_check(self) -> None:
        producer = RedpandaProducer()
        result = await producer.health_check()
        assert result["backend"] == "redpanda"
        assert result["connected"] is False

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        producer = RedpandaProducer()
        await producer.send("trading-events", {"test": True})
        await producer.close()
        assert producer.is_connected() is False


# ── RedpandaConsumer ────────────────────────────────────────────────


class TestRedpandaConsumer:
    def test_initial_state(self) -> None:
        consumer = RedpandaConsumer()
        assert consumer.is_connected() is False

    def test_subscribe(self) -> None:
        consumer = RedpandaConsumer()
        result = consumer.subscribe(["trading-events", "trading-decisions"])
        assert result is True

    @pytest.mark.asyncio
    async def test_poll_empty(self) -> None:
        consumer = RedpandaConsumer()
        consumer.subscribe(["trading-events"])
        events = await consumer.poll(max_records=10)
        assert events == []

    @pytest.mark.asyncio
    async def test_poll_with_buffer(self) -> None:
        consumer = RedpandaConsumer()
        consumer.subscribe(["trading-events"])
        consumer._buffer = [
            {"event_id": "1", "event_type": "MarketEvent", "payload": {"instrument": "BTC"}},
        ]
        events = await consumer.poll(max_records=10)
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_commit_clears_buffer(self) -> None:
        consumer = RedpandaConsumer()
        consumer.subscribe(["trading-events"])
        consumer._buffer = [{"event_id": "1", "payload": {}}]
        result = await consumer.commit()
        assert result is True
        assert consumer._buffer == []

    @pytest.mark.asyncio
    async def test_seek_to_timestamp(self) -> None:
        consumer = RedpandaConsumer()
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        result = await consumer.seek_to_timestamp("trading-events", ts)
        assert result is True

    @pytest.mark.asyncio
    async def test_seek_to_offset(self) -> None:
        consumer = RedpandaConsumer()
        result = await consumer.seek_to_offset("trading-events", partition=0, offset=100)
        assert result is True

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        consumer = RedpandaConsumer()
        consumer.subscribe(["trading-events"])
        consumer._buffer = [{"event_id": "1"}]
        await consumer.close()
        assert consumer._buffer == []
        assert consumer._subscribed_topics == []


# ── RedpandaDeadLetterHandler ───────────────────────────────────────


class TestRedpandaDeadLetterHandler:
    @pytest.mark.asyncio
    async def test_handle_event(self) -> None:
        handler = RedpandaDeadLetterHandler()
        await handler.handle({"event_id": "1"}, ValueError("test error"))
        assert handler.count == 1

    @pytest.mark.asyncio
    async def test_replay_event(self) -> None:
        handler = RedpandaDeadLetterHandler()
        await handler.handle({"event_id": "1"}, ValueError("error"))
        recovered = await handler.replay("1")
        assert recovered is not None
        assert recovered["event_id"] == "1"
        assert handler.count == 0

    @pytest.mark.asyncio
    async def test_replay_unknown_event(self) -> None:
        handler = RedpandaDeadLetterHandler()
        result = await handler.replay("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_dead_events(self) -> None:
        handler = RedpandaDeadLetterHandler()
        for i in range(5):
            await handler.handle({"event_id": str(i)}, RuntimeError(f"err {i}"))
        events = await handler.list_dead_events(limit=3)
        assert len(events) == 3
        assert handler.count == 5
