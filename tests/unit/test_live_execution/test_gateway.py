"""Unit tests for packages.live_execution.gateway.

CCXT is faked by monkeypatching ``LiveExecutionGateway._create_exchange``
— no real exchange or network access.  The shared rollout controller is
reset per test so kill-switch / circuit-breaker state never leaks.
"""

from __future__ import annotations

from typing import Any, NoReturn

import pytest
from packages.live_execution.gateway import (
    GatewayExecutionError,
    GatewayValidationError,
    LiveExecutionGateway,
    OrderResult,
)
from packages.live_execution.order_state_machine import OrderState
from packages.rollout import (
    KillSwitchState,
    reset_rollout_controller,
)

from test_live_execution._fakes import FakeExchange, RateLimitError

VENUE = "binance"
SYMBOL = "BTC/USDT"


@pytest.fixture
def fake_exchange(monkeypatch: pytest.MonkeyPatch) -> FakeExchange:
    fake = FakeExchange()
    monkeypatch.setattr(
        LiveExecutionGateway,
        "_create_exchange",
        lambda self, venue: fake,
    )
    return fake


@pytest.fixture
def gateway(fake_exchange: FakeExchange) -> LiveExecutionGateway:
    return LiveExecutionGateway(venues=[VENUE])


@pytest.fixture(autouse=True)
def clean_rollout():
    reset_rollout_controller()
    yield
    reset_rollout_controller()


async def _submit(gw: LiveExecutionGateway, **overrides: Any) -> OrderResult:
    kwargs = {
        "venue": VENUE,
        "symbol": SYMBOL,
        "side": "buy",
        "order_type": "limit",
        "amount": 0.01,
        "price": 50_000.0,
    }
    kwargs.update(overrides)
    return await gw.submit_order(**kwargs)


class TestSubmitSuccessStates:
    async def test_open_response_is_pending(
        self, gateway: LiveExecutionGateway, fake_exchange: FakeExchange
    ) -> None:
        fake_exchange.order_response = {"id": "v-1", "status": "open", "filled": 0}
        result = await _submit(gateway)
        assert result.state is OrderState.PENDING
        assert result.status == "pending"
        assert result.order_id == "v-1"
        assert result.filled_quantity == 0.0

    async def test_closed_response_is_filled(
        self, gateway: LiveExecutionGateway, fake_exchange: FakeExchange
    ) -> None:
        fake_exchange.order_response = {
            "id": "v-2",
            "status": "closed",
            "filled": 0.01,
            "average": 50_100.0,
        }
        result = await _submit(gateway)
        assert result.state is OrderState.FILLED
        assert result.status == "filled"
        assert result.filled_quantity == 0.01
        assert result.fill_price == 50_100.0

    async def test_open_with_partial_fill(
        self, gateway: LiveExecutionGateway, fake_exchange: FakeExchange
    ) -> None:
        fake_exchange.order_response = {
            "id": "v-3",
            "status": "open",
            "filled": 0.004,
        }
        result = await _submit(gateway)
        assert result.state is OrderState.PARTIALLY_FILLED
        assert result.status == "partial_fill"
        assert result.filled_quantity == 0.004

    async def test_canceled_response(
        self, gateway: LiveExecutionGateway, fake_exchange: FakeExchange
    ) -> None:
        fake_exchange.order_response = {"id": "v-4", "status": "canceled", "filled": 0}
        result = await _submit(gateway)
        assert result.state is OrderState.CANCELLED
        assert result.status == "cancelled"

    async def test_expired_response(
        self, gateway: LiveExecutionGateway, fake_exchange: FakeExchange
    ) -> None:
        fake_exchange.order_response = {"id": "v-5", "status": "expired", "filled": 0}
        result = await _submit(gateway)
        assert result.state is OrderState.EXPIRED
        assert result.status == "expired"

    async def test_unknown_status_stays_pending(
        self, gateway: LiveExecutionGateway, fake_exchange: FakeExchange
    ) -> None:
        fake_exchange.order_response = {"id": "v-6", "status": "something_else"}
        result = await _submit(gateway)
        assert result.state is OrderState.PENDING
        assert result.status == "unknown"


class TestSubmitErrorHandling:
    async def test_ccxt_exception_raises_gateway_execution_error(
        self, gateway: LiveExecutionGateway, fake_exchange: FakeExchange
    ) -> None:
        fake_exchange.error = RuntimeError("exchange offline")
        with pytest.raises(GatewayExecutionError, match="exchange offline"):
            await _submit(gateway)
        # state machine ended in ERROR, keyed by the local order id
        states = gateway.get_all_order_states()
        assert set(states.values()) == {"ERROR"}

    async def test_rejected_message_transitions_to_rejected(
        self, gateway: LiveExecutionGateway, fake_exchange: FakeExchange
    ) -> None:
        fake_exchange.error = RuntimeError("Order rejected: insufficient balance")
        with pytest.raises(GatewayExecutionError):
            await _submit(gateway)
        states = gateway.get_all_order_states()
        assert set(states.values()) == {"REJECTED"}

    async def test_429_records_rate_limit_error(
        self, gateway: LiveExecutionGateway, fake_exchange: FakeExchange
    ) -> None:
        fake_exchange.error = RateLimitError("too many requests")
        with pytest.raises(GatewayExecutionError):
            await _submit(gateway)
        backoff = gateway._rate_limiter._states[VENUE].backoff
        assert backoff.current_delay > 1.0

    async def test_error_is_recorded_in_idempotency_store(
        self, gateway: LiveExecutionGateway, fake_exchange: FakeExchange
    ) -> None:
        fake_exchange.error = RuntimeError("boom")
        with pytest.raises(GatewayExecutionError):
            await _submit(gateway, idempotency_key="err-key")
        cached = await gateway._idempotency_store.get("err-key", VENUE)
        assert cached is not None
        assert cached["status"] == "error"
        assert "boom" in cached["error"]

    async def test_rate_limiter_acquire_failure_raises(
        self, gateway: LiveExecutionGateway, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(venue: str, tokens: int = 1, timeout: float | None = None) -> NoReturn:
            raise RuntimeError("no tokens")

        monkeypatch.setattr(gateway._rate_limiter, "acquire", boom)
        with pytest.raises(GatewayExecutionError, match="Rate limit error"):
            await _submit(gateway)


class TestSubmitIdempotency:
    async def test_duplicate_returns_cached_result(
        self, gateway: LiveExecutionGateway, fake_exchange: FakeExchange
    ) -> None:
        fake_exchange.order_response = {
            "id": "v-9",
            "status": "closed",
            "filled": 0.01,
            "average": 50_100.5,
        }
        first = await _submit(gateway, idempotency_key="dup-1")
        assert first.state is OrderState.FILLED

        second = await _submit(gateway, idempotency_key="dup-1")
        assert len(fake_exchange.create_calls) == 1
        assert second.order_id == "v-9"
        assert second.state is OrderState.FILLED
        assert second.status == "filled"
        assert second.filled_quantity == 0.01
        assert second.fill_price == 50_100.5
        assert second.error == ""
        assert second.raw_response == first.to_dict()

    async def test_duplicate_restores_cached_error_result(
        self, gateway: LiveExecutionGateway, fake_exchange: FakeExchange
    ) -> None:
        fake_exchange.error = RuntimeError("venue down")
        with pytest.raises(GatewayExecutionError):
            await _submit(gateway, idempotency_key="dup-err")

        fake_exchange.error = None  # venue is back — duplicate must not re-submit
        second = await _submit(gateway, idempotency_key="dup-err")
        assert len(fake_exchange.create_calls) == 1
        assert second.state is OrderState.ERROR
        assert second.status == "error"
        assert "venue down" in second.error


class TestSubmitValidationAndRollout:
    async def test_invalid_side_raises_validation_error(
        self, gateway: LiveExecutionGateway, fake_exchange: FakeExchange
    ) -> None:
        with pytest.raises(GatewayValidationError) as exc_info:
            await _submit(gateway, side="hold")
        assert any(e.code == "INVALID_SIDE" for e in exc_info.value.errors)
        assert fake_exchange.create_calls == []

    async def test_invalid_order_type_raises_validation_error(
        self, gateway: LiveExecutionGateway, fake_exchange: FakeExchange
    ) -> None:
        with pytest.raises(GatewayValidationError) as exc_info:
            await _submit(gateway, order_type="fok")
        assert any(e.code == "INVALID_ORDER_TYPE" for e in exc_info.value.errors)

    async def test_kill_switch_activated_blocks_submit(
        self, gateway: LiveExecutionGateway, fake_exchange: FakeExchange
    ) -> None:
        from packages.rollout import get_rollout_controller

        get_rollout_controller().force_kill(reason="test halt")
        assert (
            get_rollout_controller().kill_switch.state
            == KillSwitchState.ACTIVATED
        )
        with pytest.raises(GatewayExecutionError, match="kill switch"):
            await _submit(gateway)
        assert fake_exchange.create_calls == []

    async def test_circuit_breaker_open_blocks_submit(
        self, gateway: LiveExecutionGateway, fake_exchange: FakeExchange
    ) -> None:
        from packages.rollout import CircuitState, get_rollout_controller

        get_rollout_controller().circuit_breaker.force_open(reason="test trip")
        assert (
            get_rollout_controller().circuit_breaker.state == CircuitState.OPEN
        )
        with pytest.raises(GatewayExecutionError, match="circuit breaker"):
            await _submit(gateway)
        assert fake_exchange.create_calls == []


class TestSubmitOrderIdentity:
    async def test_state_machine_retrievable_by_venue_order_id(
        self, gateway: LiveExecutionGateway, fake_exchange: FakeExchange
    ) -> None:
        fake_exchange.order_response = {"id": "venue-77", "status": "open", "filled": 0}
        result = await _submit(gateway)
        assert result.order_id == "venue-77"
        assert gateway.get_state_machine("venue-77") is not None
        assert gateway.get_order_state("venue-77") is OrderState.PENDING

    async def test_order_id_falls_back_to_local_id(
        self, gateway: LiveExecutionGateway, fake_exchange: FakeExchange
    ) -> None:
        fake_exchange.order_response = {"id": None, "status": "open", "filled": 0}
        result = await _submit(gateway)
        assert result.order_id.startswith("live_")
        assert gateway.get_state_machine(result.order_id) is not None

    async def test_client_order_id_keyed_and_forwarded(
        self, gateway: LiveExecutionGateway, fake_exchange: FakeExchange
    ) -> None:
        result = await _submit(gateway, client_order_id="client-77")
        assert fake_exchange.create_calls[0]["params"]["clientOrderId"] == "client-77"
        assert gateway.get_state_machine("client-77") is not None
        assert gateway.get_state_machine(result.order_id) is not None


class TestCcxTParams:
    async def test_limit_order_passes_numeric_amount_and_price(
        self, gateway: LiveExecutionGateway, fake_exchange: FakeExchange
    ) -> None:
        await _submit(gateway, amount=0.1, price=49_999.5)
        kwargs = fake_exchange.create_calls[0]
        assert kwargs["amount"] == 0.1
        assert not isinstance(kwargs["amount"], str)
        assert kwargs["price"] == 49_999.5
        assert not isinstance(kwargs["price"], str)

    async def test_stop_limit_passes_numeric_stop_price(
        self, gateway: LiveExecutionGateway, fake_exchange: FakeExchange
    ) -> None:
        await _submit(gateway, order_type="stop_limit", stop_price=49_000.25)
        kwargs = fake_exchange.create_calls[0]
        assert kwargs["type"] == "stop_limit"
        assert kwargs["params"]["stopPrice"] == 49_000.25
        assert not isinstance(kwargs["params"]["stopPrice"], str)

    async def test_market_order_passes_numeric_amount(
        self, gateway: LiveExecutionGateway, fake_exchange: FakeExchange
    ) -> None:
        fake_exchange.order_response = {"id": "v-m", "status": "closed", "filled": 0.01}
        await _submit(gateway, order_type="market", price=None)
        kwargs = fake_exchange.create_calls[0]
        assert kwargs["type"] == "market"
        assert kwargs["amount"] == 0.01
        assert not isinstance(kwargs["amount"], str)
        assert "price" not in kwargs


class TestCancelOrder:
    async def test_cancel_pending_order(
        self, gateway: LiveExecutionGateway, fake_exchange: FakeExchange
    ) -> None:
        fake_exchange.order_response = {"id": "v-p", "status": "open", "filled": 0}
        await _submit(gateway)
        result = await gateway.cancel_order(VENUE, "v-p")
        assert result.state is OrderState.CANCELLED
        assert result.status == "cancelled"
        assert fake_exchange.cancel_calls[0][0] == "v-p"
        assert gateway.get_order_state("v-p") is OrderState.CANCELLED

    async def test_cancel_uses_state_machine_symbol(
        self, gateway: LiveExecutionGateway, fake_exchange: FakeExchange
    ) -> None:
        fake_exchange.order_response = {"id": "v-s", "status": "open", "filled": 0}
        await _submit(gateway)
        await gateway.cancel_order(VENUE, "v-s")
        assert fake_exchange.cancel_calls[0][1] == SYMBOL

    async def test_cancel_terminal_order_skips_exchange(
        self, gateway: LiveExecutionGateway, fake_exchange: FakeExchange
    ) -> None:
        fake_exchange.order_response = {
            "id": "v-f",
            "status": "closed",
            "filled": 0.01,
        }
        await _submit(gateway)
        result = await gateway.cancel_order(VENUE, "v-f")
        assert result.state is OrderState.FILLED
        assert result.status == "already_filled"
        assert fake_exchange.cancel_calls == []
        assert gateway.get_order_state("v-f") is OrderState.FILLED

    async def test_cancel_429_records_rate_limit_error(
        self, gateway: LiveExecutionGateway, fake_exchange: FakeExchange
    ) -> None:
        fake_exchange.order_response = {"id": "v-429", "status": "open", "filled": 0}
        await _submit(gateway)
        fake_exchange.error = RateLimitError("cancel rate limited")
        result = await gateway.cancel_order(VENUE, "v-429")
        assert result.status == "cancel_error"
        backoff = gateway._rate_limiter._states[VENUE].backoff
        assert backoff.current_delay > 1.0

    async def test_cancel_not_blocked_by_kill_switch(
        self, gateway: LiveExecutionGateway, fake_exchange: FakeExchange
    ) -> None:
        from packages.rollout import get_rollout_controller

        fake_exchange.order_response = {"id": "v-k", "status": "open", "filled": 0}
        await _submit(gateway)
        get_rollout_controller().force_kill(reason="test halt")
        result = await gateway.cancel_order(VENUE, "v-k")
        assert result.state is OrderState.CANCELLED
        assert len(fake_exchange.cancel_calls) == 1


class TestStatusQueries:
    async def test_get_order_status_uses_state_machine_symbol(
        self, gateway: LiveExecutionGateway, fake_exchange: FakeExchange
    ) -> None:
        fake_exchange.order_response = {"id": "v-q", "status": "open", "filled": 0}
        await _submit(gateway)
        data = await gateway.get_order_status(VENUE, "v-q")
        assert data["id"] == "v-q"
        assert fake_exchange.fetch_calls[0] == ("v-q", SYMBOL)

    async def test_get_order_status_unknown_order_uses_unknown_symbol(
        self, gateway: LiveExecutionGateway, fake_exchange: FakeExchange
    ) -> None:
        await gateway.get_order_status(VENUE, "never-seen")
        assert fake_exchange.fetch_calls[0] == ("never-seen", "UNKNOWN")

    async def test_get_open_orders_without_symbol_passes_none(
        self, gateway: LiveExecutionGateway, fake_exchange: FakeExchange
    ) -> None:
        orders = await gateway.get_open_orders(VENUE)
        assert orders == []
        assert fake_exchange.open_orders_calls == [None]

    async def test_get_open_orders_with_symbol(
        self, gateway: LiveExecutionGateway, fake_exchange: FakeExchange
    ) -> None:
        await gateway.get_open_orders(VENUE, symbol=SYMBOL)
        assert fake_exchange.open_orders_calls == [SYMBOL]
