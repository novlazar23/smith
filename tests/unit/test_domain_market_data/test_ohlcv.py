"""Tests für OHLCV Domain Models."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest
from packages.domain.market_data.ohlcv import (
    CandleAggregation,
    MultiTimeframeAggregator,
)


def _make_candle(
    offset_min: int = 0,
    open_price: float = 100.0,
    high: float | None = None,
    low: float | None = None,
    close: float | None = None,
    volume: float = 10.0,
    trade_count: int = 5,
) -> dict:
    base_time = datetime(2024, 1, 1, 0, 0, 0) + timedelta(minutes=offset_min)
    return {
        "open_time": base_time,
        "close_time": base_time + timedelta(minutes=1),
        "open": open_price,
        "high": high or open_price * 1.01,
        "low": low or open_price * 0.99,
        "close": close or open_price * 1.005,
        "volume": volume,
        "trade_count": trade_count,
    }


class TestCandleAggregation:
    def test_from_raw_candles_single(self) -> None:
        candles = [_make_candle(offset_min=0)]
        agg = CandleAggregation.from_raw_candles(candles, "5m", "BTC/USDT", "BINANCE")
        assert agg.instrument == "BTC/USDT"
        assert agg.venue == "BINANCE"
        assert agg.timeframe == "5m"
        assert agg.sub_candle_count == 1
        assert agg.open == 100.0
        assert agg.high > 100.0
        assert agg.low < 100.0
        assert agg.volume == 10.0
        assert agg.trade_count == 5

    def test_from_raw_candles_multiple(self) -> None:
        candles = [
            _make_candle(offset_min=0, open_price=100, high=102, low=99, close=101, volume=10),
            _make_candle(offset_min=1, open_price=101, high=103, low=100, close=102, volume=20),
            _make_candle(offset_min=2, open_price=102, high=104, low=101, close=103, volume=15),
        ]
        agg = CandleAggregation.from_raw_candles(candles, "15m", "ETH/USDT", "BINANCE")
        assert agg.open == 100.0
        assert agg.close == 103.0
        assert agg.high == 104.0
        assert agg.low == 99.0
        assert agg.volume == 45.0
        assert agg.trade_count == 15
        assert agg.sub_candle_count == 3

    def test_from_raw_candles_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            CandleAggregation.from_raw_candles([], "5m", "BTC/USDT", "BINANCE")

    def test_invalid_high_low_raises(self) -> None:
        with pytest.raises(ValueError, match=r"high.*low"):
            CandleAggregation(
                instrument="BTC/USDT", venue="BINANCE", timeframe="1m",
                open_time=datetime.now(), close_time=datetime.now(),
                open=100, high=90, low=95, close=92, volume=10, trade_count=5,
            )


class TestMultiTimeframeAggregator:
    def test_aggregate_single_period(self) -> None:
        agg = MultiTimeframeAggregator(base_timeframe="1m")
        candles = [_make_candle(offset_min=i) for i in range(5)]
        result = agg.aggregate(candles, "5m", "BTC/USDT", "BINANCE")
        assert result is not None
        assert result.sub_candle_count == 5

    def test_aggregate_all_timeframes(self) -> None:
        agg = MultiTimeframeAggregator()
        candles = [_make_candle(offset_min=i) for i in range(60)]
        results = agg.aggregate_all(candles, "BTC/USDT", "BINANCE")
        assert "1h" in results
        assert results["1h"].sub_candle_count == 60

    def test_aggregate_base_raises(self) -> None:
        agg = MultiTimeframeAggregator()
        with pytest.raises(ValueError, match="base"):
            agg.aggregate([_make_candle()], "1m", "BTC/USDT", "BINANCE")

    def test_sma(self) -> None:
        agg = MultiTimeframeAggregator()
        candles = [_make_candle(offset_min=i, open_price=100 + i * 0.5) for i in range(30)]
        sma = agg.compute_simple_moving_average(candles, window=10)
        assert len(sma) == 30
        assert np.isnan(sma[0])
        assert not np.isnan(sma[9])
        # Letzter Wert sollte > 100 sein (aufwärts Trend)
        assert sma[-1] > 100

    def test_sma_too_few_raises(self) -> None:
        agg = MultiTimeframeAggregator()
        candles = [_make_candle(offset_min=i) for i in range(3)]
        with pytest.raises(ValueError, match="Need at least 10"):
            agg.compute_simple_moving_average(candles, window=10)

    def test_ema(self) -> None:
        agg = MultiTimeframeAggregator()
        candles = [_make_candle(offset_min=i, open_price=100 + i) for i in range(30)]
        ema = agg.compute_exponential_moving_average(candles, span=10)
        assert len(ema) == 30
        # EMA sollte monoton steigen bei aufwärts Trend
        assert ema[-1] > ema[0]
