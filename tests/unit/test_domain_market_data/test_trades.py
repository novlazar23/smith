"""Tests für Trade Domain Models."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from packages.domain.market_data.trades import (
    TradeAggregation,
    TradeAggregator,
    VolumeProfile,
)


def _make_trade(offset_sec: int = 0, price: float = 100.0, quantity: float = 1.0, side: str = "buy") -> dict:
    base = datetime(2024, 1, 1, 0, 0, 0)
    return {
        "event_time": base + timedelta(seconds=offset_sec),
        "price": price,
        "quantity": quantity,
        "side": side,
        "trade_id": f"trade-{offset_sec}",
    }


class TestTradeAggregation:
    def test_valid_aggregation(self) -> None:
        agg = TradeAggregation(
            instrument="BTC/USDT", venue="BINANCE",
            start_time=datetime.now(), end_time=datetime.now(),
            trade_count=10, total_volume=100.0, total_value=10000.0,
            avg_price=100.0, max_price=101.0, min_price=99.0,
        )
        assert agg.trade_count == 10
        assert agg.bid_volume == 0.0

    def test_invalid_high_low_raises(self) -> None:
        with pytest.raises(ValueError, match="max_price"):
            TradeAggregation(
                instrument="X", venue="Y",
                start_time=datetime.now(), end_time=datetime.now(),
                trade_count=1, total_volume=1.0, total_value=100.0,
                avg_price=100.0, max_price=90.0, min_price=100.0,
            )


class TestVolumeProfile:
    def test_from_trades_basic(self) -> None:
        trades = [
            _make_trade(price=100.0, quantity=10.0),
            _make_trade(price=100.0, quantity=20.0),
            _make_trade(price=101.0, quantity=5.0),
            _make_trade(price=99.0, quantity=15.0),
        ]
        vp = VolumeProfile.from_trades(trades, "BTC/USDT", "BINANCE")
        assert vp.total_volume == 50.0
        assert vp.poc_price == 100.0
        assert vp.poc_volume == 30.0
        assert len(vp.price_levels) == 3

    def test_from_trades_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            VolumeProfile.from_trades([], "X", "Y")

    def test_price_levels_sorted(self) -> None:
        trades = [
            _make_trade(price=105.0, quantity=1.0),
            _make_trade(price=100.0, quantity=2.0),
            _make_trade(price=95.0, quantity=3.0),
        ]
        vp = VolumeProfile.from_trades(trades, "BTC/USDT", "BINANCE")
        prices = [pl.price for pl in vp.price_levels]
        assert prices == sorted(prices)


class TestTradeAggregator:
    def test_aggregate_one_minute(self) -> None:
        agg = TradeAggregator(window=timedelta(minutes=1))
        trades = [_make_trade(offset_sec=i * 10) for i in range(6)]  # 6 trades in 60s
        results = agg.aggregate(trades, "BTC/USDT", "BINANCE")
        assert len(results) >= 1

    def test_aggregate_empty(self) -> None:
        agg = TradeAggregator()
        results = agg.aggregate([], "BTC/USDT", "BINANCE")
        assert results == []

    def test_aggregate_multi_minute(self) -> None:
        agg = TradeAggregator(window=timedelta(minutes=1))
        trades = [
            _make_trade(offset_sec=10, price=100.0, quantity=5.0, side="buy"),
            _make_trade(offset_sec=20, price=101.0, quantity=3.0, side="sell"),
            _make_trade(offset_sec=70, price=102.0, quantity=2.0, side="buy"),
        ]
        results = agg.aggregate(trades, "BTC/USDT", "BINANCE")
        assert len(results) == 2  # 2 verschiedene Minuten
        assert results[0].trade_count == 2
        assert results[1].trade_count == 1
        assert results[0].bid_volume == 5.0
        assert results[0].ask_volume == 3.0
