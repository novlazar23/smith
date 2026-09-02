"""Tests für den ClickHouse-Feed (Range-Filter, UTC, Resample, Empty)."""

from __future__ import annotations

from datetime import UTC, date, datetime

from apps.backtest.ch_feed import ClickHouseDataFeed
from tests.unit.test_backtest.conftest import BTC, FakeEngine, make_candles, rows_from_candles


class TestRangeFiltering:
    """Die SQL-Query filtert auf instrument/venue/Zeitfenster."""

    def test_between_in_sql(self) -> None:
        engine = FakeEngine()
        feed = ClickHouseDataFeed(
            engine,
            BTC,
            start=date(2021, 5, 15),
            end=date(2021, 5, 25),
        )
        feed.get_candles()
        sql = engine.queries[0]
        assert "BETWEEN '2021-05-15 00:00:00' AND '2021-05-25 23:59:59'" in sql
        assert "instrument = 'BTC/USDT'" in sql
        assert "venue = 'BINANCE_FUTURES'" in sql
        assert "ORDER BY open_time" in sql
        assert sql.startswith("SELECT open_time, open, high, low, close, volume FROM candles_history")

    def test_venue_filter_and_escaping(self) -> None:
        engine = FakeEngine()
        feed = ClickHouseDataFeed(engine, "BTC/US'DT", venue="O\\Neil")
        feed.get_candles()
        sql = engine.queries[0]
        assert "instrument = 'BTC/US\\'DT'" in sql
        assert "venue = 'O\\\\Neil'" in sql

    def test_only_start_uses_gte(self) -> None:
        engine = FakeEngine()
        feed = ClickHouseDataFeed(engine, BTC, start=datetime(2021, 5, 15, 12, 0, tzinfo=UTC))
        feed.get_candles()
        assert "open_time >= '2021-05-15 12:00:00'" in engine.queries[0]
        assert "BETWEEN" not in engine.queries[0]

    def test_full_history_has_no_time_filter(self) -> None:
        engine = FakeEngine()
        feed = ClickHouseDataFeed(engine, BTC)
        feed.get_candles()
        sql = engine.queries[0]
        where_clause = sql.split("WHERE")[1].split("ORDER BY")[0]
        assert "BETWEEN" not in where_clause
        assert ">=" not in where_clause
        assert "<=" not in where_clause


class TestConversion:
    """Zeilen → Candle (UTC-Datetimes, Zahlen)."""

    def test_utc_datetime_conversion(self) -> None:
        engine = FakeEngine(rows=[["2021-05-15 12:30:00", "100", "101", "99", "100.5", "10"]])
        feed = ClickHouseDataFeed(engine, BTC)
        candles = feed.get_candles()
        assert len(candles) == 1
        candle = candles[0]
        assert candle.timestamp == datetime(2021, 5, 15, 12, 30, tzinfo=UTC)
        assert candle.timestamp.tzinfo is not None
        assert candle.symbol == BTC
        assert candle.open == 100.0
        assert candle.high == 101.0
        assert candle.low == 99.0
        assert candle.close == 100.5
        assert candle.volume == 10.0

    def test_rows_are_sorted_ascending(self) -> None:
        engine = FakeEngine(
            rows=[
                ["2021-05-15 00:01:00", "1", "2", "1", "2", "1"],
                ["2021-05-15 00:00:00", "1", "2", "1", "2", "1"],
            ]
        )
        feed = ClickHouseDataFeed(engine, BTC)
        candles = feed.get_candles()
        assert candles[0].timestamp < candles[1].timestamp

    def test_symbol_property(self) -> None:
        feed = ClickHouseDataFeed(FakeEngine(), BTC)
        assert feed.symbol == BTC

    def test_other_symbol_returns_empty(self) -> None:
        feed = ClickHouseDataFeed(FakeEngine(rows=[["2021-05-15 00:00:00", "1", "2", "1", "2", "1"]]), BTC)
        assert feed.get_candles("ETH/USDT") == []


class TestResample5m:
    """1m → 5m: OHLCV-Aggregation, 5-Minuten-UTC-Grenzen, Partial-Drop."""

    def test_seven_minutes_to_two_5m_candles(self) -> None:
        # 7 1m-Kerzen exakt auf einer 5m-Grenze beginnend → 2 5m-Kerzen
        candles = make_candles(7, start=datetime(2021, 5, 15, 12, 0, tzinfo=UTC), step=1.0)
        engine = FakeEngine(rows=rows_from_candles(candles))
        feed = ClickHouseDataFeed(engine, BTC, resample="5m")
        result = feed.get_candles()
        assert len(result) == 2
        first, second = result
        assert first.timestamp == datetime(2021, 5, 15, 12, 0, tzinfo=UTC)
        assert second.timestamp == datetime(2021, 5, 15, 12, 5, tzinfo=UTC)
        # erster Bucket: 12:00..12:04 (5 Kerzen)
        assert first.open == candles[0].open
        assert first.high == max(c.high for c in candles[:5])
        assert first.low == min(c.low for c in candles[:5])
        assert first.close == candles[4].close
        assert first.volume == sum(c.volume for c in candles[:5])
        # zweiter Bucket: 12:05..12:06 (2 Kerzen)
        assert second.open == candles[5].open
        assert second.close == candles[6].close
        assert second.volume == candles[5].volume + candles[6].volume

    def test_partial_leading_bucket_dropped(self) -> None:
        # 7 1m-Kerzen mittig im Bucket beginnend → leading 5m-Bucket unvollständig → drop
        candles = make_candles(7, start=datetime(2021, 5, 15, 12, 2, tzinfo=UTC), step=1.0)
        engine = FakeEngine(rows=rows_from_candles(candles))
        feed = ClickHouseDataFeed(engine, BTC, resample="5m")
        result = feed.get_candles()
        assert len(result) == 1
        assert result[0].timestamp == datetime(2021, 5, 15, 12, 5, tzinfo=UTC)
        assert result[0].volume == sum(c.volume for c in candles[3:])

    def test_invalid_resample_rejected(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="resample"):
            ClickHouseDataFeed(FakeEngine(), BTC, resample="15m")

    def test_empty_result_stays_empty_with_resample(self) -> None:
        feed = ClickHouseDataFeed(FakeEngine(), BTC, resample="5m")
        assert feed.get_candles() == []
