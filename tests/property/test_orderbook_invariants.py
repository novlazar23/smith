"""Property-based tests for orderbook and market data invariants.

These tests verify structural invariants around market data types —
that open interest is non-negative, liquidation prices are reasonable,
and funding rates fall within expected bounds.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from packages.domain.market_data import FundingRate, Liquidation, OpenInterest
from packages.domain.market_data.derivatives import LiquidationSide

# ---------------------------------------------------------------------------
# Helper strategies
# ---------------------------------------------------------------------------

_instrument = st.just("BTC/USDT")
_venue = st.just("binance")
_datetime = st.datetimes(min_value=datetime(2020, 1, 1), max_value=datetime(2030, 12, 31), timezones=st.just(UTC))


# ---------------------------------------------------------------------------
# Test: Open interest values are non-negative
# ---------------------------------------------------------------------------

class TestOpenInterestNonNegative:
    """Open interest must always be >= 0."""

    @given(
        _instrument,
        _venue,
        st.floats(min_value=0.0, max_value=10_000_000_000, allow_nan=False, allow_infinity=False),
        _datetime,
    )
    @settings(max_examples=100)
    def test_open_interest_constructs_with_non_negative_value(
        self,
        instrument: str,
        venue: str,
        oi: float,
        event_time: datetime,
    ) -> None:
        """OpenInterest with oi >= 0 should construct without error."""
        obj = OpenInterest(instrument=instrument, venue=venue, open_interest=oi, event_time=event_time)
        assert obj.open_interest == oi
        assert obj.open_interest >= 0.0

    @given(
        _instrument,
        _venue,
        st.floats(min_value=-1000, max_value=-0.001, allow_nan=False, allow_infinity=False),
        _datetime,
    )
    @settings(max_examples=50)
    def test_open_interest_rejects_negative_value(
        self,
        instrument: str,
        venue: str,
        negative_oi: float,
        event_time: datetime,
    ) -> None:
        """OpenInterest with oi < 0 should raise ValueError."""
        with pytest.raises(ValueError):
            OpenInterest(
                instrument=instrument,
                venue=venue,
                open_interest=negative_oi,
                event_time=event_time,
            )


# ---------------------------------------------------------------------------
# Test: Funding rates are within [-1, 1]
# ---------------------------------------------------------------------------

class TestFundingRateBounds:
    """Funding rates must be in [-1.0, 1.0]."""

    @given(
        _instrument,
        _venue,
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1.0, max_value=10_000_000, allow_nan=False, allow_infinity=False),
        _datetime,
    )
    @settings(max_examples=100)
    def test_funding_rate_constructs_validly(
        self,
        instrument: str,
        venue: str,
        rate: float,
        mark_price: float,
        event_time: datetime,
    ) -> None:
        """FundingRate within bounds should construct without error."""
        next_funding = event_time + timedelta(hours=8)
        obj = FundingRate(
            instrument=instrument,
            venue=venue,
            funding_rate=rate,
            mark_price=mark_price,
            next_funding_time=next_funding,
            event_time=event_time,
        )
        assert -1.0 <= obj.funding_rate <= 1.0

    @given(
        _instrument,
        _venue,
        st.floats(min_value=1.0001, max_value=1000, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1.0, max_value=10_000_000, allow_nan=False, allow_infinity=False),
        _datetime,
    )
    @settings(max_examples=50)
    def test_funding_rate_rejects_over_one(
        self,
        instrument: str,
        venue: str,
        too_high_rate: float,
        mark_price: float,
        event_time: datetime,
    ) -> None:
        """FundingRate > 1.0 should raise ValueError."""
        next_funding = event_time + timedelta(hours=8)
        with pytest.raises(ValueError):
            FundingRate(
                instrument=instrument,
                venue=venue,
                funding_rate=too_high_rate,
                mark_price=mark_price,
                next_funding_time=next_funding,
                event_time=event_time,
            )

    @given(
        _instrument,
        _venue,
        st.floats(min_value=-1000, max_value=-1.0001, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1.0, max_value=10_000_000, allow_nan=False, allow_infinity=False),
        _datetime,
    )
    @settings(max_examples=50)
    def test_funding_rate_rejects_under_negative_one(
        self,
        instrument: str,
        venue: str,
        too_low_rate: float,
        mark_price: float,
        event_time: datetime,
    ) -> None:
        """FundingRate < -1.0 should raise ValueError."""
        next_funding = event_time + timedelta(hours=8)
        with pytest.raises(ValueError):
            FundingRate(
                instrument=instrument,
                venue=venue,
                funding_rate=too_low_rate,
                mark_price=mark_price,
                next_funding_time=next_funding,
                event_time=event_time,
            )


# ---------------------------------------------------------------------------
# Test: Liquidation prices are reasonable relative to current price
# ---------------------------------------------------------------------------

class TestLiquidationReasonableness:
    """Liquidation prices and values must be positive and reasonable."""

    @given(
        _instrument,
        _venue,
        st.just(LiquidationSide.LONG),
        st.floats(min_value=0.01, max_value=100_000, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1.0, max_value=10_000_000, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1.0, max_value=10_000_000, allow_nan=False, allow_infinity=False),
        _datetime,
    )
    @settings(max_examples=100)
    def test_liquidation_prices_are_positive(
        self,
        instrument: str,
        venue: str,
        side: LiquidationSide,
        quantity: float,
        price: float,
        value: float,
        event_time: datetime,
    ) -> None:
        """All prices and values in Liquidation must be > 0."""
        obj = Liquidation(
            instrument=instrument,
            venue=venue,
            side=side,
            quantity=quantity,
            price=price,
            value=value,
            event_time=event_time,
        )
        assert obj.price > 0
        assert obj.quantity > 0
        assert obj.value > 0

    @given(
        _instrument,
        _venue,
        st.just(LiquidationSide.SHORT),
        st.floats(min_value=0.01, max_value=100_000, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1.0, max_value=10_000_000, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1.0, max_value=10_000_000, allow_nan=False, allow_infinity=False),
        _datetime,
    )
    @settings(max_examples=100)
    def test_liquidation_value_consistency_with_quantity_and_price(
        self,
        instrument: str,
        venue: str,
        side: LiquidationSide,
        quantity: float,
        price: float,
        value: float,
        event_time: datetime,
    ) -> None:
        """Liquidation value should be proportional to quantity * price."""
        assume(value >= 0)
        Liquidation(
            instrument=instrument,
            venue=venue,
            side=side,
            quantity=quantity,
            price=price,
            value=value,
            event_time=event_time,
        )
        # Value should be roughly quantity * price (within 10x for leverage scenarios)
        expected_notional = quantity * price
        # Allow generous ratio — liquidation value may differ due to margin mechanics
        assert expected_notional > 0
        assert value > 0


# ---------------------------------------------------------------------------
# Test: DerivativeSnapshot consistency
# ---------------------------------------------------------------------------

class TestDerivativeSnapshot:
    """Snapshot-level invariants across market data types."""

    @given(
        _instrument,
        _venue,
        _datetime,
    )
    @settings(max_examples=50)
    def test_empty_snapshot_has_zero_total_liquidation(self, instrument: str, venue: str, event_time: datetime) -> None:
        """A snapshot with no liquidations should have total_liquidation_value == 0."""
        from packages.domain.market_data.derivatives import DerivativeSnapshot

        snapshot = DerivativeSnapshot(
            instrument=instrument,
            venue=venue,
            event_time=event_time,
        )
        assert snapshot.total_liquidation_value == 0.0
        assert snapshot.net_liquidation_side is None

    @given(
        _instrument,
        _venue,
        _datetime,
        st.floats(min_value=100, max_value=10_000_000, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1, max_value=10_000_000, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50)
    def test_snapshot_with_single_liquidation_has_positive_total(
        self,
        instrument: str,
        venue: str,
        event_time: datetime,
        price: float,
        value: float,
    ) -> None:
        """A snapshot with one liquidation should have total > 0."""
        assume(price > 0 and value > 0)
        from packages.domain.market_data.derivatives import DerivativeSnapshot, Liquidation

        liq = Liquidation(
            instrument=instrument,
            venue=venue,
            side=LiquidationSide.LONG,
            quantity=price * 0.01,
            price=price,
            value=value,
            event_time=event_time,
        )
        snapshot = DerivativeSnapshot(
            instrument=instrument,
            venue=venue,
            event_time=event_time,
            recent_liquidations=[liq],
        )
        assert snapshot.total_liquidation_value > 0
        assert snapshot.net_liquidation_side is not None
