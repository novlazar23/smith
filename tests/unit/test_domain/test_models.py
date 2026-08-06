"""Tests für Core Domain Models (packages/domain/)."""

import pytest
from packages.domain.models import Instrument, Order, Portfolio, Position, Trade


class TestInstrument:
    def test_freeze(self) -> None:
        inst = Instrument(symbol="BTC", venue="binance", asset_type="spot", tick_size=0.01, lot_size=0.001)
        try:
            inst.symbol = "ETH"  # type: ignore[union-attr]
            pytest.fail("Expected frozen error")
        except Exception:
            pass

    def test_min_length(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Instrument(symbol="", venue="binance", asset_type="spot", tick_size=0.01, lot_size=0.001)

    def test_tick_lot_positive(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Instrument(symbol="BTC", venue="binance", asset_type="spot", tick_size=-1, lot_size=0.001)


class TestPosition:
    def test_valid_long(self) -> None:
        pos = Position(
            instrument="BTC", venue="binance", side="long",
            quantity=1.5, entry_price=50000, unrealized_pnl=500, realized_pnl=200
        )
        assert pos.side == "long"
        assert pos.quantity == 1.5

    def test_negative_quantity_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Position(instrument="BTC", venue="binance", side="long", quantity=-1, entry_price=50000)

    def test_invalid_side_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Position(instrument="BTC", venue="binance", side="foobar", quantity=1, entry_price=50000)

    def test_notional(self) -> None:
        pos = Position(instrument="BTC", venue="binance", side="long", quantity=2, entry_price=50000)
        assert pos.notional(51000) == 102000.0

    def test_freeze(self) -> None:
        from pydantic import ValidationError

        pos = Position(instrument="BTC", venue="binance", side="long", quantity=1, entry_price=50000)
        with pytest.raises(ValidationError):
            pos.quantity = 2


class TestPortfolio:
    def test_empty_portfolio(self) -> None:
        port = Portfolio(portfolio_id="p1", total_equity=100000)
        assert len(port.positions) == 0
        assert port.total_unrealized_pnl == 0
        assert port.total_realized_pnl == 0
        assert port.total_exposure_ratio == 0

    def test_with_position(self) -> None:
        pos = Position(instrument="BTC", venue="binance", side="long", quantity=2, entry_price=50000, unrealized_pnl=500)
        port = Portfolio(portfolio_id="p1", total_equity=100000, positions={"BTC|binance": pos})
        assert port.total_unrealized_pnl == 500
        assert port.has_position("BTC|binance")
        assert port.get_position("BTC|binance") == pos

    def test_position_flat_not_found(self) -> None:
        flat = Position(instrument="BTC", venue="binance", side="flat", quantity=0, entry_price=50000)
        port = Portfolio(portfolio_id="p1", total_equity=100000, positions={"BTC|binance": flat})
        assert not port.has_position("BTC|binance")
        assert port.get_position("BTC|binance") is None

    def test_freeze(self) -> None:
        from pydantic import ValidationError

        port = Portfolio(portfolio_id="p1", total_equity=100000)
        with pytest.raises(ValidationError):
            port.total_equity = 200000


class TestTrade:
    def test_valid_trade(self) -> None:
        trade = Trade(
            trade_id="t1", portfolio_id="p1", instrument="BTC", venue="binance",
            price=50000, quantity=1.5, side="buy", commission=15
        )
        assert trade.trade_id == "t1"
        assert trade.commission == 15

    def test_freeze(self) -> None:
        from pydantic import ValidationError

        trade = Trade(trade_id="t1", portfolio_id="p1", instrument="BTC", venue="binance",
                      price=50000, quantity=1, side="buy")
        with pytest.raises(ValidationError):
            trade.price = 60000


class TestOrder:
    def test_pending_limit_order(self) -> None:
        order = Order(
            order_id="o1", portfolio_id="p1", instrument="BTC", venue="binance",
            price=50000, quantity=1, side="buy", order_type="limit"
        )
        assert order.status == "pending"
        assert order.order_type == "limit"

    def test_freeze(self) -> None:
        from pydantic import ValidationError

        order = Order(order_id="o1", portfolio_id="p1", instrument="BTC", venue="binance",
                      price=50000, quantity=1, side="buy")
        with pytest.raises(ValidationError):
            order.status = "filled"
