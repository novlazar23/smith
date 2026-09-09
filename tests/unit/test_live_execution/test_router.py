"""Unit tests for packages.live_execution.router.

Uses a real ``LiveExecutionGateway`` whose ``_create_exchange`` is
monkeypatched to return per-venue fakes — no real exchange or network.
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
from packages.live_execution.router import OrderRouter, RouterConfig, RouteStrategy
from packages.live_execution.validator import OrderValidator, ValidationConfig
from packages.rollout import reset_rollout_controller

from test_live_execution._fakes import FakeExchange

B = "binance"
Y = "bybit"
OKX = "okx"
SYMBOL = "BTC/USDT"


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> dict[str, FakeExchange]:
    created: dict[str, FakeExchange] = {}

    def factory(self: LiveExecutionGateway, venue: str) -> FakeExchange:
        return created.setdefault(venue, FakeExchange())

    monkeypatch.setattr(LiveExecutionGateway, "_create_exchange", factory)
    return created


@pytest.fixture
def gateway() -> LiveExecutionGateway:
    # Equity large enough that 1.0 BTC @ 50k (50k notional) passes the
    # default 10% max-notional check.
    validator = OrderValidator(ValidationConfig(account_equity=1_000_000.0))
    return LiveExecutionGateway(venues=[B, Y, OKX], validator=validator)


@pytest.fixture(autouse=True)
def clean_rollout():
    reset_rollout_controller()
    yield
    reset_rollout_controller()


async def _route(router: OrderRouter, **overrides: Any) -> OrderResult:
    kwargs = {
        "symbol": SYMBOL,
        "side": "buy",
        "order_type": "limit",
        "amount": 0.01,
        "price": 50_000.0,
    }
    kwargs.update(overrides)
    return await router.route_order(**kwargs)


class TestRouteOrder:
    async def test_single_strategy_uses_default_venue(
        self, gateway: LiveExecutionGateway, fakes: dict[str, FakeExchange]
    ) -> None:
        router = OrderRouter(
            gateway,
            venues=[B, Y],
            config=RouterConfig(strategy=RouteStrategy.SINGLE, default_venue=B),
        )
        result = await _route(router)
        assert result.venue == B
        assert len(fakes[B].create_calls) == 1
        assert Y not in fakes

    async def test_explicit_venue_overrides_strategy(
        self, gateway: LiveExecutionGateway, fakes: dict[str, FakeExchange]
    ) -> None:
        router = OrderRouter(
            gateway,
            venues=[B, Y],
            config=RouterConfig(strategy=RouteStrategy.PRIORITY),
        )
        router.add_venue(B, priority=0)
        router.add_venue(Y, priority=1)
        result = await _route(router, venue=Y)
        assert result.venue == Y
        assert len(fakes[Y].create_calls) == 1
        assert B not in fakes

    async def test_priority_strategy_picks_lowest_priority(
        self, gateway: LiveExecutionGateway, fakes: dict[str, FakeExchange]
    ) -> None:
        router = OrderRouter(
            gateway,
            venues=[B, Y],
            config=RouterConfig(strategy=RouteStrategy.PRIORITY),
        )
        router.add_venue(B, priority=5)
        router.add_venue(Y, priority=1)
        result = await _route(router)
        assert result.venue == Y

    async def test_health_aware_picks_highest_health(
        self, gateway: LiveExecutionGateway, fakes: dict[str, FakeExchange]
    ) -> None:
        router = OrderRouter(
            gateway,
            venues=[B, Y],
            config=RouterConfig(strategy=RouteStrategy.HEALTH_AWARE),
        )
        router.update_health_score(B, 0.9)
        router.update_health_score(Y, 1.0)
        result = await _route(router)
        assert result.venue == Y

    async def test_no_healthy_venue_raises(
        self, gateway: LiveExecutionGateway, fakes: dict[str, FakeExchange]
    ) -> None:
        router = OrderRouter(
            gateway,
            venues=[B, Y],
            config=RouterConfig(strategy=RouteStrategy.PRIORITY),
        )
        router.update_health_score(B, 0.0)
        router.update_health_score(Y, 0.0)
        with pytest.raises(GatewayExecutionError, match="No healthy venue"):
            await _route(router)

    async def test_non_positive_amount_raises(
        self, gateway: LiveExecutionGateway, fakes: dict[str, FakeExchange]
    ) -> None:
        router = OrderRouter(gateway, venues=[B], config=RouterConfig())
        with pytest.raises(GatewayValidationError):
            await _route(router, amount=0.0)

    async def test_idempotency_key_forwarded(
        self, gateway: LiveExecutionGateway, fakes: dict[str, FakeExchange]
    ) -> None:
        router = OrderRouter(
            gateway,
            venues=[B, Y],
            config=RouterConfig(strategy=RouteStrategy.SINGLE, default_venue=B),
        )
        await _route(router, idempotency_key="route-key")
        assert await gateway._idempotency_store.exists("route-key", B) is True


class TestSplitOrder:
    async def test_split_equal_two_venues(
        self, gateway: LiveExecutionGateway, fakes: dict[str, FakeExchange]
    ) -> None:
        router = OrderRouter(gateway, venues=[B, Y], config=RouterConfig())
        result = await router.route_split_order(
            symbol=SYMBOL, side="buy", order_type="limit",
            amount=1.0, price=50_000.0,
        )
        assert [a.venue_id for a in result.allocations] == [B, Y]
        assert result.allocations[0].quantity == pytest.approx(0.5)
        assert result.allocations[1].quantity == pytest.approx(0.5)
        assert result.total_quantity == 1.0

    async def test_split_equal_three_venues_sums_to_amount(
        self, gateway: LiveExecutionGateway, fakes: dict[str, FakeExchange]
    ) -> None:
        router = OrderRouter(gateway, venues=[B, Y, OKX], config=RouterConfig())
        result = await router.route_split_order(
            symbol=SYMBOL, side="buy", order_type="limit",
            amount=1.0, price=50_000.0,
        )
        assert len(result.allocations) == 3
        total = sum(a.quantity for a in result.allocations)
        assert total == pytest.approx(1.0)

    async def test_split_proportional_uses_health_weights(
        self, gateway: LiveExecutionGateway, fakes: dict[str, FakeExchange]
    ) -> None:
        router = OrderRouter(gateway, venues=[B, Y], config=RouterConfig())
        # Both scores must stay >= 0.5 (is_healthy threshold); 1.0 : 0.5
        # gives the 2:1 allocation ratio the assertions expect.
        router.update_health_score(B, 1.0)
        router.update_health_score(Y, 0.5)
        result = await router.route_split_order(
            symbol=SYMBOL, side="buy", order_type="limit",
            amount=0.9, price=50_000.0,
            split_strategy="proportional",
        )
        by_venue = {a.venue_id: a.quantity for a in result.allocations}
        assert by_venue[B] == pytest.approx(0.6)
        assert by_venue[Y] == pytest.approx(0.3)

    async def test_split_uses_per_venue_idempotency_keys(
        self, gateway: LiveExecutionGateway, fakes: dict[str, FakeExchange]
    ) -> None:
        router = OrderRouter(gateway, venues=[B, Y], config=RouterConfig())
        await router.route_split_order(
            symbol=SYMBOL, side="buy", order_type="limit",
            amount=1.0, price=50_000.0,
            idempotency_key="base-key",
        )
        assert await gateway._idempotency_store.exists(f"base-key:{B}", B) is True
        assert await gateway._idempotency_store.exists(f"base-key:{Y}", Y) is True
        assert await gateway._idempotency_store.exists("base-key", B) is False
        assert await gateway._idempotency_store.exists("base-key", Y) is False

    async def test_split_pending_results_mark_not_successful(
        self, gateway: LiveExecutionGateway, fakes: dict[str, FakeExchange]
    ) -> None:
        router = OrderRouter(gateway, venues=[B, Y], config=RouterConfig())
        result = await router.route_split_order(
            symbol=SYMBOL, side="buy", order_type="limit",
            amount=1.0, price=50_000.0,
        )
        assert result.all_successful is False
        assert len(result.errors) == 2
        assert any(B in e for e in result.errors)
        assert any(Y in e for e in result.errors)

    async def test_split_filled_results_mark_successful(
        self, gateway: LiveExecutionGateway, fakes: dict[str, FakeExchange]
    ) -> None:
        router = OrderRouter(gateway, venues=[B, Y], config=RouterConfig())
        for venue in (B, Y):
            fakes[venue] = FakeExchange(
                order_response={"id": f"v-{venue}", "status": "closed", "filled": 0.5}
            )
        result = await router.route_split_order(
            symbol=SYMBOL, side="buy", order_type="limit",
            amount=1.0, price=50_000.0,
        )
        assert result.all_successful is True
        assert result.errors == []
        assert result.total_filled_quantity == pytest.approx(1.0)

    async def test_split_error_degrades_venue(
        self, gateway: LiveExecutionGateway, fakes: dict[str, FakeExchange]
    ) -> None:
        router = OrderRouter(gateway, venues=[B, Y], config=RouterConfig())
        # binance must fill: a PENDING leg also counts as an error, and the
        # test asserts exactly one (bybit's) error.
        fakes[B] = FakeExchange(
            order_response={"id": "v-binance", "status": "closed", "filled": 0.5}
        )
        fakes[Y] = FakeExchange(error=RuntimeError("bybit down"))
        result = await router.route_split_order(
            symbol=SYMBOL, side="buy", order_type="limit",
            amount=1.0, price=50_000.0,
        )
        assert result.all_successful is False
        assert len(result.errors) == 1
        assert Y in result.errors[0]
        assert router.get_venue(Y).health_score == pytest.approx(0.9)
        assert router.get_venue(B).health_score == pytest.approx(1.0)

    async def test_split_without_healthy_venues_raises(
        self, gateway: LiveExecutionGateway, fakes: dict[str, FakeExchange]
    ) -> None:
        router = OrderRouter(gateway, venues=[B, Y], config=RouterConfig())
        router.update_health_score(B, 0.0)
        router.update_health_score(Y, 0.0)
        with pytest.raises(GatewayExecutionError, match="No healthy venues"):
            await router.route_split_order(
                symbol=SYMBOL, side="buy", order_type="limit",
                amount=1.0, price=50_000.0,
            )


class TestVenueHealth:
    async def test_check_venue_health_healthy(
        self, gateway: LiveExecutionGateway, fakes: dict[str, FakeExchange]
    ) -> None:
        router = OrderRouter(gateway, venues=[B], config=RouterConfig())
        outcome = await router.check_venue_health(B)
        assert outcome == {"venue": B, "healthy": True, "score": 1.0}
        assert router.get_venue(B).health_score == 1.0

    async def test_check_venue_health_failure_degrades(
        self,
        gateway: LiveExecutionGateway,
        fakes: dict[str, FakeExchange],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def boom(self: LiveExecutionGateway, venue: str) -> NoReturn:
            raise RuntimeError("ccxt unavailable")

        monkeypatch.setattr(LiveExecutionGateway, "_create_exchange", boom)
        router = OrderRouter(gateway, venues=[B], config=RouterConfig())
        outcome = await router.check_venue_health(B)
        assert outcome["healthy"] is False
        assert "ccxt unavailable" in outcome["error"]
        assert router.get_venue(B).health_score == pytest.approx(0.9)


class TestVenueManagement:
    def test_get_healthy_venues_sorted_and_filtered(
        self, gateway: LiveExecutionGateway
    ) -> None:
        router = OrderRouter(gateway, venues=[], config=RouterConfig())
        router.add_venue(B, priority=0, health_score=0.9)
        router.add_venue(Y, priority=1, health_score=1.0)
        router.add_venue(OKX, priority=0, health_score=0.2)
        healthy = router.get_healthy_venues()
        assert [v.venue_id for v in healthy] == [B, Y]
        assert all(v.is_healthy for v in healthy)

    def test_remove_venue(self, gateway: LiveExecutionGateway) -> None:
        router = OrderRouter(gateway, venues=[B, Y], config=RouterConfig())
        router.remove_venue(B)
        assert router.get_venue(B) is None
        assert router.get_venue(Y) is not None

    def test_status_reports_strategy_and_venues(
        self, gateway: LiveExecutionGateway
    ) -> None:
        router = OrderRouter(
            gateway,
            venues=[B],
            config=RouterConfig(strategy=RouteStrategy.PRIORITY),
        )
        status = router.get_status()
        assert status["strategy"] == "PRIORITY"
        assert B in status["venues"]
