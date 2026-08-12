"""Tests für Derivatives Domain Models."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from packages.domain.market_data.derivatives import (
    DerivativeSnapshot,
    FundingRate,
    Liquidation,
    LiquidationSide,
    OpenInterest,
)


class TestFundingRate:
    def test_valid_funding_rate(self) -> None:
        fr = FundingRate(
            instrument="BTC-PERP", venue="BINANCE",
            funding_rate=0.0001, mark_price=50000.0,
            next_funding_time=datetime.now() + timedelta(hours=8),
            event_time=datetime.now(),
        )
        assert fr.funding_rate_pct == 0.01

    def test_negative_funding_rate(self) -> None:
        fr = FundingRate(
            instrument="X-PERP", venue="Y",
            funding_rate=-0.0005, mark_price=1000.0,
            next_funding_time=datetime.now() + timedelta(hours=1),
            event_time=datetime.now(),
        )
        assert fr.funding_rate_pct == -0.05

    def test_invalid_mark_price_raises(self) -> None:
        with pytest.raises(ValueError, match="mark_price"):
            FundingRate(
                instrument="X", venue="Y",
                funding_rate=0.0001, mark_price=-1.0,
                next_funding_time=datetime.now(),
                event_time=datetime.now(),
            )

    def test_invalid_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="funding_rate"):
            FundingRate(
                instrument="X", venue="Y",
                funding_rate=2.0, mark_price=100.0,
                next_funding_time=datetime.now(),
                event_time=datetime.now(),
            )


class TestOpenInterest:
    def test_valid_open_interest(self) -> None:
        oi = OpenInterest(
            instrument="BTC-PERP", venue="BINANCE",
            open_interest=100000.0,
            open_interest_value=5000000000.0,
            event_time=datetime.now(),
        )
        assert oi.open_interest == 100000.0

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="open_interest"):
            OpenInterest(
                instrument="X", venue="Y",
                open_interest=-1.0, event_time=datetime.now(),
            )


class TestLiquidation:
    def test_long_liquidation(self) -> None:
        liq = Liquidation(
            instrument="BTC-PERP", venue="BINANCE",
            side=LiquidationSide.LONG,
            quantity=1.0, price=40000.0, value=40000.0,
            event_time=datetime.now(),
        )
        assert liq.side == LiquidationSide.LONG
        assert liq.value == 40000.0

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="quantity"):
            Liquidation(
                instrument="X", venue="Y",
                side=LiquidationSide.LONG,
                quantity=0.0, price=100.0, value=100.0,
                event_time=datetime.now(),
            )


class TestDerivativeSnapshot:
    def test_total_liquidation_value(self) -> None:
        liqs = [
            Liquidation(
                instrument="X-PERP", venue="Y",
                side=LiquidationSide.LONG,
                quantity=1.0, price=100.0, value=100.0,
                event_time=datetime.now(),
            ),
            Liquidation(
                instrument="X-PERP", venue="Y",
                side=LiquidationSide.SHORT,
                quantity=0.5, price=200.0, value=100.0,
                event_time=datetime.now(),
            ),
        ]
        snapshot = DerivativeSnapshot(
            instrument="X-PERP", venue="Y",
            event_time=datetime.now(),
            recent_liquidations=liqs,
        )
        assert snapshot.total_liquidation_value == 200.0
        assert snapshot.net_liquidation_side == "balanced"

    def test_net_liquidation_side_long(self) -> None:
        liqs = [
            Liquidation(
                instrument="X-PERP", venue="Y",
                side=LiquidationSide.LONG, quantity=2.0, price=100.0, value=200.0,
                event_time=datetime.now(),
            ),
            Liquidation(
                instrument="X-PERP", venue="Y",
                side=LiquidationSide.SHORT, quantity=0.5, price=100.0, value=50.0,
                event_time=datetime.now(),
            ),
        ]
        snapshot = DerivativeSnapshot(
            instrument="X-PERP", venue="Y",
            event_time=datetime.now(),
            recent_liquidations=liqs,
        )
        assert snapshot.net_liquidation_side == "long"

    def test_no_liquidations(self) -> None:
        snapshot = DerivativeSnapshot(
            instrument="X-PERP", venue="Y",
            event_time=datetime.now(),
        )
        assert snapshot.total_liquidation_value == 0.0
        assert snapshot.net_liquidation_side is None
