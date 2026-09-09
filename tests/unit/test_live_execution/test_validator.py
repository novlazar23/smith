"""Unit tests for packages.live_execution.validator.

Covers size, price, stop price, notional, side, order type, and the
risk-gate pipeline (approved / blocked / async / fail-closed).
"""

from __future__ import annotations

from typing import Any, NoReturn

import pytest
from packages.live_execution.validator import (
    OrderValidator,
    RiskGateResult,
    ValidationConfig,
    ValidationError,
)


def _codes(errors: list[ValidationError]) -> set[str]:
    return {e.code for e in errors}


def _validator(**config: Any) -> OrderValidator:
    base = {
        "min_order_size": 0.001,
        "max_order_size": 10.0,
        "min_price": 1.0,
        "max_price": 100_000.0,
        "max_notional_ratio": 0.1,
        "account_equity": 10_000.0,
    }
    base.update(config)
    return OrderValidator(config=ValidationConfig(**base))


async def _validate(validator: OrderValidator, **overrides: Any) -> list:
    # quantity 0.01 x price 50_000 = 500 notional, under the default
    # cap of 0.1 x 10_000 = 1_000, so the base case passes validation.
    kwargs = {
        "symbol": "BTC/USDT",
        "venue": "binance",
        "side": "buy",
        "quantity": 0.01,
        "price": 50_000.0,
        "order_type": "limit",
    }
    kwargs.update(overrides)
    return await validator.validate(**kwargs)


class TestSizeValidation:
    async def test_zero_quantity_rejected(self) -> None:
        errors = await _validate(_validator(), quantity=0.0)
        assert "INVALID_SIZE" in _codes(errors)

    async def test_negative_quantity_rejected(self) -> None:
        errors = await _validate(_validator(), quantity=-0.5)
        assert "INVALID_SIZE" in _codes(errors)

    async def test_below_min_size_rejected(self) -> None:
        errors = await _validate(_validator(), quantity=0.0001)
        assert "BELOW_MIN_SIZE" in _codes(errors)

    async def test_above_max_size_rejected(self) -> None:
        errors = await _validate(_validator(max_order_size=1.0), quantity=5.0)
        assert "ABOVE_MAX_SIZE" in _codes(errors)

    async def test_valid_quantity_passes(self) -> None:
        assert await _validate(_validator(), quantity=0.02) == []


class TestPriceValidation:
    async def test_limit_without_price_rejected(self) -> None:
        errors = await _validate(_validator(), order_type="limit", price=None)
        assert "MISSING_PRICE" in _codes(errors)

    async def test_market_without_price_ok(self) -> None:
        assert await _validate(_validator(), order_type="market", price=None) == []

    async def test_stop_limit_without_stop_price_rejected(self) -> None:
        errors = await _validate(_validator(), order_type="stop_limit", stop_price=None)
        assert "MISSING_STOP_PRICE" in _codes(errors)

    async def test_stop_market_without_stop_price_rejected(self) -> None:
        errors = await _validate(_validator(), order_type="stop_market", stop_price=None)
        assert "MISSING_STOP_PRICE" in _codes(errors)

    async def test_stop_order_with_stop_price_ok(self) -> None:
        assert (
            await _validate(
                _validator(), order_type="stop_limit", stop_price=49_000.0
            )
            == []
        )

    async def test_price_below_min_rejected(self) -> None:
        errors = await _validate(_validator(min_price=100.0), price=1.0)
        assert "INVALID_PRICE" in _codes(errors)

    async def test_price_above_max_rejected(self) -> None:
        errors = await _validate(_validator(max_price=60_000.0), price=99_000.0)
        assert "PRICE_TOO_HIGH" in _codes(errors)


class TestNotionalValidation:
    async def test_notional_within_cap_passes(self) -> None:
        assert await _validate(_validator(), quantity=0.01) == []

    async def test_notional_over_cap_rejected(self) -> None:
        errors = await _validate(_validator(), quantity=0.1)
        assert "NOTIONAL_EXCEEDED" in _codes(errors)

    async def test_market_order_skips_notional_check(self) -> None:
        assert (
            await _validate(_validator(), order_type="market", price=None) == []
        )


class TestSideValidation:
    @pytest.mark.parametrize("side", ["buy", "sell"])
    async def test_valid_sides_pass(self, side: str) -> None:
        assert await _validate(_validator(), side=side) == []

    @pytest.mark.parametrize("side", ["hold", "long", "BUYX", ""])
    async def test_invalid_side_rejected(self, side: str) -> None:
        errors = await _validate(_validator(), side=side)
        assert "INVALID_SIDE" in _codes(errors)

    async def test_invalid_side_error_carries_field(self) -> None:
        errors = await _validate(_validator(), side="hold")
        matching = [e for e in errors if e.code == "INVALID_SIDE"]
        assert matching[0].field == "side"
        assert matching[0].is_blocking is True


class TestOrderTypeValidation:
    @pytest.mark.parametrize(
        "order_type", ["market", "limit", "stop_limit", "stop_market"]
    )
    async def test_valid_order_types_pass(self, order_type: str) -> None:
        kwargs = {"stop_price": 49_000.0} if "stop" in order_type else {}
        assert await _validate(_validator(), order_type=order_type, **kwargs) == []

    @pytest.mark.parametrize("order_type", ["fok", "ioc", "trailing", ""])
    async def test_invalid_order_type_rejected(self, order_type: str) -> None:
        errors = await _validate(_validator(), order_type=order_type)
        assert "INVALID_ORDER_TYPE" in _codes(errors)


class TestRiskGate:
    async def test_no_risk_gate_passes(self) -> None:
        assert await _validate(_validator()) == []

    async def test_risk_gate_approved_passes(self) -> None:
        validator = _validator()
        validator._risk_gate_fn = lambda **kw: RiskGateResult(approved=True)
        assert await _validate(validator) == []

    async def test_risk_gate_blocked_rejects(self) -> None:
        validator = _validator()
        validator._risk_gate_fn = lambda **kw: RiskGateResult(
            approved=False, blocking_reasons=["drawdown too high"]
        )
        errors = await _validate(validator)
        assert "RISK_GATE_BLOCKED" in _codes(errors)

    async def test_async_risk_gate_blocked_rejects(self) -> None:
        validator = _validator()

        async def gate(**kw: Any) -> RiskGateResult:
            return RiskGateResult(approved=False, blocking_reasons=["async no"])

        validator._risk_gate_fn = gate
        errors = await _validate(validator)
        assert "RISK_GATE_BLOCKED" in _codes(errors)

    async def test_dict_risk_gate_blocked_rejects(self) -> None:
        validator = _validator()
        validator._risk_gate_fn = lambda **kw: {
            "approved": False,
            "blocking_reasons": ["dict gate no"],
        }
        errors = await _validate(validator)
        assert "RISK_GATE_BLOCKED" in _codes(errors)

    async def test_risk_gate_exception_fails_closed(self) -> None:
        validator = _validator()

        def boom(**kw: Any) -> NoReturn:
            raise RuntimeError("risk service down")

        validator._risk_gate_fn = boom
        errors = await _validate(validator)
        assert "RISK_GATE_ERROR" in _codes(errors)
        assert all(e.is_blocking for e in errors if e.code == "RISK_GATE_ERROR")
