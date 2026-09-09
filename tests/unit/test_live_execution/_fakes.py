"""Shared fake CCXT exchange for live_execution unit tests."""

from __future__ import annotations

from typing import Any


class FakeExchange:
    """Records calls and replays a canned CCXT response or raises."""

    def __init__(
        self,
        order_response: dict | None = None,
        error: Exception | None = None,
    ) -> None:
        self.order_response = dict(
            order_response or {"id": "venue-1", "status": "open", "filled": 0}
        )
        self.error = error
        self.create_calls: list[dict] = []
        self.cancel_calls: list[tuple[str, str]] = []
        self.fetch_calls: list[tuple[str, str]] = []
        self.open_orders_calls: list[str | None] = []

    async def create_order(self, **kwargs: Any) -> dict:
        self.create_calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return dict(self.order_response)

    async def cancel_order(self, order_id: str, symbol: str | None = None, **kw: Any) -> dict:
        self.cancel_calls.append((order_id, symbol or ""))
        if self.error is not None:
            raise self.error
        return {"id": order_id, "status": "canceled"}

    async def fetch_order(self, order_id: str, symbol: str | None = None, **kw: Any) -> dict:
        self.fetch_calls.append((order_id, symbol or ""))
        return {"id": order_id, "status": "open"}

    async def fetch_open_orders(self, symbol: str | None = None, **kw: Any) -> list:
        self.open_orders_calls.append(symbol)
        return []


class RateLimitError(RuntimeError):
    status_code = 429
