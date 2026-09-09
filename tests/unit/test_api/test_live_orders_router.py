"""Tests for the live order management API router.

Uses a minimal FastAPI app containing only the live orders router and a
fake gateway injected via ``live_orders._gateway`` — no real exchange,
network, or middleware involved.
"""

from __future__ import annotations

from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from apps.api.routers import live_orders
from fastapi import FastAPI
from fastapi.testclient import TestClient
from packages.live_execution import (
    GatewayExecutionError,
    GatewayValidationError,
    OrderResult,
    OrderState,
    ValidationError,
)
from packages.rollout import (
    KillSwitchState,
    get_rollout_controller,
    reset_rollout_controller,
)


class FakeGateway:
    """Stand-in for :class:`LiveExecutionGateway` that records calls."""

    def __init__(self) -> None:
        self.submit_calls: list[dict[str, Any]] = []
        self.cancel_calls: list[tuple[str, str]] = []
        self._next_id = 0
        self.submit_error: Exception | None = None
        self.submit_result: OrderResult | None = None
        self.cancel_result: OrderResult | None = None
        self.cancel_error: Exception | None = None

    def _default_result(self, kwargs: dict[str, Any]) -> OrderResult:
        self._next_id += 1
        result = OrderResult(
            idempotency_key=str(kwargs["idempotency_key"]),
            symbol=str(kwargs["symbol"]),
            venue=str(kwargs["venue"]),
            side=str(kwargs["side"]),
            order_type=str(kwargs["order_type"]),
            quantity=float(kwargs["amount"]),
            price=kwargs["price"],
        )
        result.order_id = f"live_{self._next_id:012d}"
        result.state = OrderState.PENDING
        result.status = "pending"
        return result

    async def submit_order(self, **kwargs: Any) -> OrderResult:
        # Mirror the real gateway's rollout safety gate.
        if get_rollout_controller().kill_switch.state == KillSwitchState.ACTIVATED:
            raise GatewayExecutionError(
                "submit_order blocked: rollout kill switch is activated"
            )
        self.submit_calls.append(kwargs)
        if self.submit_error is not None:
            raise self.submit_error
        return self.submit_result or self._default_result(kwargs)

    async def cancel_order(self, venue: str, order_id: str) -> OrderResult:
        self.cancel_calls.append((venue, order_id))
        if self.cancel_error is not None:
            raise self.cancel_error
        if self.cancel_result is not None:
            return self.cancel_result
        return _cancel_result(order_id, state=OrderState.CANCELLED, status="cancelled")

    def get_state_machine(self, order_id: str) -> Any:
        return None


def _cancel_result(
    order_id: str,
    *,
    state: OrderState,
    status: str,
    error: str = "",
) -> OrderResult:
    result = OrderResult(
        idempotency_key="",
        symbol="BTC/USDT",
        venue="binance",
        side="buy",
        order_type="",
        quantity=1.0,
        price=None,
    )
    result.order_id = order_id
    result.state = state
    result.status = status
    result.error = error
    return result


def _submit_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "instrument": "BTC/USDT",
        "direction": "buy",
        "quantity": 1.0,
        "order_type": "limit",
        "price": 100.0,
        "idempotency_key": "key-1",
        "venue": "binance",
    }
    payload.update(overrides)
    return payload


def _registry_order(
    order_id: str,
    side: str,
    quantity: float,
    price: float,
    *,
    when: datetime,
    state: OrderState = OrderState.FILLED,
) -> OrderResult:
    result = OrderResult(
        idempotency_key="k",
        symbol="BTC/USDT",
        venue="binance",
        side=side,
        order_type="limit",
        quantity=quantity,
        price=price,
    )
    result.order_id = order_id
    result.state = state
    result.status = state.name.lower()
    result.filled_quantity = quantity
    result.fill_price = price
    result.submitted_at = when
    return result


@pytest.fixture(autouse=True)
def _reset_live_state() -> Generator[None, None, None]:
    live_orders._order_registry.clear()
    live_orders._idempotency_index.clear()
    live_orders.set_live_price_provider(None)
    reset_rollout_controller()
    yield
    reset_rollout_controller()


class LiveTestClient(TestClient):
    """TestClient that injects an ``X-Security-Role`` header by default."""

    def __init__(self, app: FastAPI, role: str = "") -> None:
        super().__init__(app)
        self.role = role

    def request(self, method: str, url: str, **kwargs: object):
        if self.role:
            headers = dict(kwargs.get("headers") or {})
            headers.setdefault("X-Security-Role", self.role)
            kwargs["headers"] = headers
        return super().request(method, url, **kwargs)


@pytest.fixture
def client(request: pytest.Request) -> LiveTestClient:
    app = FastAPI()
    app.include_router(live_orders.router)
    role = getattr(request.cls, "ROLE", "live_operator")
    return LiveTestClient(app, role=role)


@pytest.fixture
def fake_gateway(monkeypatch: pytest.MonkeyPatch) -> FakeGateway:
    gateway = FakeGateway()
    monkeypatch.setattr(live_orders, "_gateway", gateway)
    return gateway


@pytest.fixture
def live_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "packages.governance.feature_flags.FeatureFlags.is_enabled",
        lambda self, flag, environment=None: True,
    )


class TestFeatureFlagGate:
    # Valid role — the 403 below comes from the feature flag, not the role.
    ROLE = "live_operator"

    def test_submit_blocked_when_flag_disabled(self, client: TestClient) -> None:
        assert client.post("/v1/live/orders", json=_submit_payload()).status_code == 403

    def test_list_blocked_when_flag_disabled(self, client: TestClient) -> None:
        assert client.get("/v1/live/orders").status_code == 403

    def test_cancel_blocked_when_flag_disabled(self, client: TestClient) -> None:
        response = client.post("/v1/live/cancel", json={"order_id": "x"})
        assert response.status_code == 403

    def test_kill_switch_blocked_when_flag_disabled(self, client: TestClient) -> None:
        response = client.post(
            "/v1/live/kill-switch", json={"action": "activate", "reason": "x"}
        )
        assert response.status_code == 403

    def test_pnl_blocked_when_flag_disabled(self, client: TestClient) -> None:
        assert client.get("/v1/live/pnl").status_code == 403

    def test_pnl_daily_blocked_when_flag_disabled(self, client: TestClient) -> None:
        assert client.get("/v1/live/pnl/daily").status_code == 403


class TestSubmit:
    ROLE = "live_operator"  # EXECUTE_LIVE

    def test_submit_success_returns_201_and_stores_order(
        self, client: TestClient, live_enabled: None, fake_gateway: FakeGateway
    ) -> None:
        response = client.post("/v1/live/orders", json=_submit_payload())
        assert response.status_code == 201
        body = response.json()
        assert body["order_id"] == "live_000000000001"
        assert body["state"] == "PENDING"
        assert body["idempotency_key"] == "key-1"
        assert "submitted_at" in body
        assert "live_000000000001" in live_orders._order_registry
        assert len(fake_gateway.submit_calls) == 1

    def test_submit_duplicate_idempotency_key_returns_cached_order(
        self, client: TestClient, live_enabled: None, fake_gateway: FakeGateway
    ) -> None:
        first = client.post("/v1/live/orders", json=_submit_payload())
        second = client.post("/v1/live/orders", json=_submit_payload())
        assert first.status_code == 201
        assert second.status_code == 201
        assert second.json()["order_id"] == first.json()["order_id"]
        assert len(fake_gateway.submit_calls) == 1

    def test_submit_normalizes_uppercase_direction_and_order_type(
        self, client: TestClient, live_enabled: None, fake_gateway: FakeGateway
    ) -> None:
        response = client.post(
            "/v1/live/orders", json=_submit_payload(direction="BUY", order_type="LIMIT")
        )
        assert response.status_code == 201
        call = fake_gateway.submit_calls[0]
        assert call["side"] == "buy"
        assert call["order_type"] == "limit"

    def test_submit_passes_stop_price_to_gateway(
        self, client: TestClient, live_enabled: None, fake_gateway: FakeGateway
    ) -> None:
        response = client.post(
            "/v1/live/orders",
            json=_submit_payload(order_type="stop_market", stop_price=90.0),
        )
        assert response.status_code == 201
        assert fake_gateway.submit_calls[0]["stop_price"] == 90.0

    def test_gateway_validation_error_returns_422(
        self, client: TestClient, live_enabled: None, fake_gateway: FakeGateway
    ) -> None:
        fake_gateway.submit_error = GatewayValidationError(
            [ValidationError(code="size", message="quantity below minimum")]
        )
        response = client.post("/v1/live/orders", json=_submit_payload())
        assert response.status_code == 422
        assert live_orders._order_registry == {}

    def test_gateway_execution_error_returns_500(
        self, client: TestClient, live_enabled: None, fake_gateway: FakeGateway
    ) -> None:
        fake_gateway.submit_error = GatewayExecutionError("exchange down")
        response = client.post("/v1/live/orders", json=_submit_payload())
        assert response.status_code == 500
        assert live_orders._order_registry == {}


class TestListOrders:
    ROLE = "live_operator"  # READ_METRICS

    def test_list_exact_order_id_filter(
        self, client: TestClient, live_enabled: None, fake_gateway: FakeGateway
    ) -> None:
        submitted = client.post("/v1/live/orders", json=_submit_payload())
        order_id = submitted.json()["order_id"]
        # substring must NOT match
        response = client.get("/v1/live/orders", params={"order_id": order_id[:8]})
        assert response.status_code == 200
        assert response.json() == []
        response = client.get("/v1/live/orders", params={"order_id": order_id})
        assert response.status_code == 200
        assert [row["order_id"] for row in response.json()] == [order_id]

    def test_list_status_venue_time_filters(
        self, client: TestClient, live_enabled: None, fake_gateway: FakeGateway
    ) -> None:
        client.post("/v1/live/orders", json=_submit_payload(idempotency_key="k1"))
        client.post(
            "/v1/live/orders",
            json=_submit_payload(idempotency_key="k2", venue="bybit", direction="sell"),
        )

        response = client.get("/v1/live/orders", params={"status": "PENDING"})
        assert len(response.json()) == 2
        response = client.get("/v1/live/orders", params={"status": "FILLED"})
        assert response.json() == []

        response = client.get("/v1/live/orders", params={"venue": "bybit"})
        assert [row["venue"] for row in response.json()] == ["bybit"]
        response = client.get("/v1/live/orders", params={"venue": "binance"})
        assert len(response.json()) == 1

        future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
        assert client.get("/v1/live/orders", params={"from": future}).json() == []
        assert len(client.get("/v1/live/orders", params={"to": future}).json()) == 2


class TestCancel:
    ROLE = "live_operator"  # CANCEL_ORDERS

    def test_cancel_unknown_order_returns_404(
        self, client: TestClient, live_enabled: None, fake_gateway: FakeGateway
    ) -> None:
        response = client.post("/v1/live/cancel", json={"order_id": "nope"})
        assert response.status_code == 404

    def test_cancel_terminal_order_returns_409(
        self, client: TestClient, live_enabled: None, fake_gateway: FakeGateway
    ) -> None:
        submitted = client.post("/v1/live/orders", json=_submit_payload())
        order_id = submitted.json()["order_id"]
        live_orders._order_registry[order_id].state = OrderState.FILLED
        response = client.post("/v1/live/cancel", json={"order_id": order_id})
        assert response.status_code == 409

    def test_cancel_success_updates_registry(
        self, client: TestClient, live_enabled: None, fake_gateway: FakeGateway
    ) -> None:
        submitted = client.post("/v1/live/orders", json=_submit_payload())
        order_id = submitted.json()["order_id"]
        response = client.post(
            "/v1/live/cancel", json={"order_id": order_id, "reason": "done"}
        )
        assert response.status_code == 200
        assert response.json() == {"status": "cancelled", "order_state": "CANCELLED"}
        stored = live_orders._order_registry[order_id]
        assert stored.state == OrderState.CANCELLED
        assert stored.status == "cancelled"

    def test_cancel_failure_returns_500_and_leaves_state_unchanged(
        self, client: TestClient, live_enabled: None, fake_gateway: FakeGateway
    ) -> None:
        submitted = client.post("/v1/live/orders", json=_submit_payload())
        order_id = submitted.json()["order_id"]
        stored = live_orders._order_registry[order_id]
        fake_gateway.cancel_result = _cancel_result(
            order_id, state=OrderState.ERROR, status="cancel_error", error="venue timeout"
        )
        response = client.post("/v1/live/cancel", json={"order_id": order_id})
        assert response.status_code == 500
        assert stored.state == OrderState.PENDING
        assert stored.status == "pending"
        assert stored.error == ""

    def test_cancel_exception_returns_500_and_leaves_state_unchanged(
        self, client: TestClient, live_enabled: None, fake_gateway: FakeGateway
    ) -> None:
        submitted = client.post("/v1/live/orders", json=_submit_payload())
        order_id = submitted.json()["order_id"]
        stored = live_orders._order_registry[order_id]
        fake_gateway.cancel_error = RuntimeError("connection dropped")
        response = client.post("/v1/live/cancel", json={"order_id": order_id})
        assert response.status_code == 500
        assert stored.state == OrderState.PENDING
        assert stored.status == "pending"


class TestKillSwitch:
    ROLE = "risk_manager"  # MANAGE_KILL_SWITCH

    def test_activate_blocks_new_submits_via_rollout_state(
        self, client: TestClient, live_enabled: None, fake_gateway: FakeGateway
    ) -> None:
        response = client.post(
            "/v1/live/kill-switch", json={"action": "activate", "reason": "halt"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["state"] == KillSwitchState.ACTIVATED
        assert body["confirmed"] is True
        assert body["affected_orders"] == []

        # Submit as live_operator (has EXECUTE_LIVE); blocked by the
        # activated kill switch in the rollout controller, not by RBAC.
        response = client.post(
            "/v1/live/orders",
            json=_submit_payload(idempotency_key="k2"),
            headers={"X-Security-Role": "live_operator"},
        )
        assert response.status_code == 500
        assert fake_gateway.submit_calls == []

    def test_deactivate_works(
        self, client: TestClient, live_enabled: None, fake_gateway: FakeGateway
    ) -> None:
        client.post("/v1/live/kill-switch", json={"action": "activate", "reason": "halt"})
        response = client.post(
            "/v1/live/kill-switch", json={"action": "deactivate", "reason": "resumed"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["state"] == KillSwitchState.DISABLED
        assert body["affected_orders"] == []

    def test_activate_auto_cancel_uses_cancel_result(
        self, client: TestClient, live_enabled: None, fake_gateway: FakeGateway
    ) -> None:
        submitted = client.post(
            "/v1/live/orders", json=_submit_payload(),
            headers={"X-Security-Role": "live_operator"},
        )
        order_id = submitted.json()["order_id"]
        response = client.post(
            "/v1/live/kill-switch", json={"action": "activate", "reason": "halt"}
        )
        assert response.status_code == 200
        assert response.json()["affected_orders"] == [order_id]
        stored = live_orders._order_registry[order_id]
        assert stored.state == OrderState.CANCELLED
        assert stored.status == "cancelled"
        assert fake_gateway.cancel_calls == [("binance", order_id)]

    def test_activate_cancel_failure_leaves_state_unchanged(
        self, client: TestClient, live_enabled: None, fake_gateway: FakeGateway
    ) -> None:
        submitted = client.post(
            "/v1/live/orders", json=_submit_payload(),
            headers={"X-Security-Role": "live_operator"},
        )
        order_id = submitted.json()["order_id"]
        fake_gateway.cancel_result = _cancel_result(
            order_id, state=OrderState.ERROR, status="cancel_error", error="timeout"
        )
        response = client.post(
            "/v1/live/kill-switch", json={"action": "activate", "reason": "halt"}
        )
        assert response.status_code == 200
        assert response.json()["affected_orders"] == [order_id]
        stored = live_orders._order_registry[order_id]
        assert stored.state == OrderState.PENDING
        assert stored.status == "pending"


class TestPnlEndpoints:
    ROLE = "live_operator"  # VIEW_LIVE_PNL

    def test_pnl_uses_tracker_without_fake_estimate(
        self, client: TestClient, live_enabled: None
    ) -> None:
        day1 = datetime(2025, 6, 1, tzinfo=UTC)
        day2 = datetime(2025, 6, 2, tzinfo=UTC)
        registry = live_orders._order_registry
        registry["o1"] = _registry_order("o1", "buy", 10.0, 100.0, when=day1)
        registry["o2"] = _registry_order("o2", "sell", 10.0, 110.0, when=day2)

        response = client.get("/v1/live/pnl")
        assert response.status_code == 200
        body = response.json()
        assert body["realized"] == pytest.approx(100.0)
        # no price provider → 0.0, never a notional-based guess
        assert body["unrealized"] == 0.0
        assert body["win_rate"] == pytest.approx(1.0)
        assert body["profit_factor"] is None
        assert body["max_drawdown"] == 0.0

    def test_pnl_daily_uses_tracker(self, client: TestClient, live_enabled: None) -> None:
        day1 = datetime(2025, 6, 1, tzinfo=UTC)
        day2 = datetime(2025, 6, 2, tzinfo=UTC)
        registry = live_orders._order_registry
        registry["o1"] = _registry_order("o1", "buy", 10.0, 100.0, when=day1)
        registry["o2"] = _registry_order("o2", "sell", 10.0, 110.0, when=day2)

        response = client.get("/v1/live/pnl/daily")
        assert response.status_code == 200
        rows = response.json()
        assert len(rows) == 1
        assert rows[0]["date"] == "2025-06-02"
        assert rows[0]["realized"] == pytest.approx(100.0)
        assert rows[0]["unrealized"] == 0.0
        assert rows[0]["pnl"] == pytest.approx(100.0)

    def test_price_provider_seam_affects_unrealized_pnl(
        self, client: TestClient, live_enabled: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        day1 = datetime(2025, 6, 1, tzinfo=UTC)
        live_orders._order_registry["o1"] = _registry_order(
            "o1", "buy", 10.0, 100.0, when=day1
        )
        monkeypatch.setattr(
            live_orders, "_price_provider", lambda venue, symbol: 120.0
        )
        response = client.get("/v1/live/pnl")
        assert response.status_code == 200
        assert response.json()["unrealized"] == pytest.approx(200.0)


class TestRBAC:
    """RBAC fail-closed behaviour when the feature flag is enabled."""

    ROLE = ""  # no default role; each test sets it explicitly via headers

    def _role_headers(self, role: str) -> dict:
        return {} if not role else {"X-Security-Role": role}

    def test_orders_requires_execute_live(
        self, client: TestClient, live_enabled: None, fake_gateway: FakeGateway
    ) -> None:
        # No role -> 403
        assert (
            client.post("/v1/live/orders", json=_submit_payload()).status_code == 403
        )
        # viewer lacks EXECUTE_LIVE -> 403
        r = client.post(
            "/v1/live/orders", json=_submit_payload(),
            headers={"X-Security-Role": "viewer"},
        )
        assert r.status_code == 403
        # live_operator has EXECUTE_LIVE -> 201
        r = client.post(
            "/v1/live/orders", json=_submit_payload(idempotency_key="k-ro"),
            headers={"X-Security-Role": "live_operator"},
        )
        assert r.status_code == 201

    def test_cancel_requires_cancel_orders(
        self, client: TestClient, live_enabled: None, fake_gateway: FakeGateway
    ) -> None:
        # live_operator holds CANCEL_ORDERS -> allowed
        assert (
            client.post(
                "/v1/live/cancel",
                json={"order_id": "live_000000000001"},
                headers={"X-Security-Role": "live_operator"},
            ).status_code
            in (404, 409, 500, 200)  # not 403: RBAC passed
        )
        # viewer lacks CANCEL_ORDERS -> 403
        r = client.post(
            "/v1/live/cancel",
            json={"order_id": "live_000000000001"},
            headers={"X-Security-Role": "viewer"},
        )
        assert r.status_code == 403

    def test_kill_switch_requires_manage_kill_switch(
        self, client: TestClient, live_enabled: None, fake_gateway: FakeGateway
    ) -> None:
        # live_operator lacks MANAGE_KILL_SWITCH -> 403
        r = client.post(
            "/v1/live/kill-switch",
            json={"action": "activate", "reason": "x"},
            headers={"X-Security-Role": "live_operator"},
        )
        assert r.status_code == 403
        # risk_manager has it -> 200
        r = client.post(
            "/v1/live/kill-switch",
            json={"action": "activate", "reason": "x"},
            headers={"X-Security-Role": "risk_manager"},
        )
        assert r.status_code == 200

    def test_pnl_requires_view_live_pnl(
        self, client: TestClient, live_enabled: None, fake_gateway: FakeGateway
    ) -> None:
        # viewer lacks VIEW_LIVE_PNL -> 403
        r = client.get("/v1/live/pnl", headers={"X-Security-Role": "viewer"})
        assert r.status_code == 403
        # live_operator has it -> 200
        r = client.get("/v1/live/pnl", headers={"X-Security-Role": "live_operator"})
        assert r.status_code == 200

    def test_unknown_role_rejected(
        self, client: TestClient, live_enabled: None, fake_gateway: FakeGateway
    ) -> None:
        r = client.get("/v1/live/pnl", headers={"X-Security-Role": "wizard"})
        assert r.status_code == 403
