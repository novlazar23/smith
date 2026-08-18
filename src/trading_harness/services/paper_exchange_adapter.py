"""PaperExchangeAdapter — Bridge zwischen Phase-4-PaperExchange und Phase-5-ExchangeAdapter.

Mappt die ExchangeAdapter-Schnittstelle (submit_order mit Preis) auf die
PaperExchange.execute_order-Signatur (TradeProposal + current_price).

Order Lifecycle:
- submit_order → PaperExchange.execute_order → FILLED/REJECTED
- get_order_status → PaperExchange.get_trade → status query
- cancel_order → PaperExchange.cancel_trade → CANCELLED

Dient als erste echte Exchange-Integration in die LiveExecutionService-Pipeline.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from trading_harness.models import TradeProposal
from trading_harness.services.exchange_adapter import ExchangeAdapter
from trading_harness.services.paper_exchange import PaperExchange

if TYPE_CHECKING:
    from trading_harness.models import PaperTrade


class PaperExchangeAdapter(ExchangeAdapter):
    """Adapter, der PaperExchange als ExchangeAdapter darstellt.

    submit_order() erstellt ein TradeProposal und delegiert an
    PaperExchange.execute_order(). get_order_status und cancel_order sind
    über PaperExchange.get_trade() bzw. PaperExchange.cancel_trade()
    implementiert.
    """

    def __init__(
        self,
        paper_exchange: PaperExchange | None = None,
        run_id: str = "run-1",
    ) -> None:
        self._paper_exchange = paper_exchange or PaperExchange()
        self._run_id = run_id

    @property
    def name(self) -> str:
        return "PAPER"

    def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_type: str = "MARKET",
    ) -> dict[str, Any]:
        """Erstellt ein TradeProposal aus den Adapter-Parametern und führt es
        via PaperExchange.execute_order() aus.

        order_type wird ignoriert (PaperExchange simuliert immer MARKET).
        """
        if price <= 0:
            return {
                "order_id": None,
                "status": "REJECTED",
                "error": "INVALID_PRICE",
            }

        if not symbol:
            return {
                "order_id": None,
                "status": "REJECTED",
                "error": "MISSING_SYMBOL",
            }

        side_upper = side.upper()
        if side_upper not in ("LONG", "SHORT", "BUY", "SELL"):
            return {
                "order_id": None,
                "status": "REJECTED",
                "error": "INVALID_SIDE",
            }

        # Normalize side: exchange uses LONG/SHORT, paper uses LONG/SHORT
        normalized_side = side_upper
        if side_upper == "BUY":
            normalized_side = "LONG"
        elif side_upper == "SELL":
            normalized_side = "SHORT"

        proposal = TradeProposal(
            decision_id=f"paper-{uuid.uuid4()}",
            symbol=symbol,
            side=normalized_side,
            equity=100000.0,  # Standard-Startkapital für Paper Trading
            entry_price=price,
            stop_price=price * 0.95,  # Default 5 % Stop-Loss
            target_price=price * 1.05,  # Default 5 % Target
            requested_quantity=quantity,
        )

        paper_trade = self._paper_exchange.execute_order(
            proposal=proposal,
            current_price=price,
        )

        return {
            "order_id": paper_trade.id,
            "status": paper_trade.status.value,
            "trade_id": paper_trade.trade_id,
            "actual_quantity": paper_trade.actual_quantity,
            "actual_price": paper_trade.actual_price,
            "fees": paper_trade.fees,
            "error": paper_trade.reject_reason,
        }

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        """Holt Order-Status via PaperExchange.get_trade().

        Returns dict with order_id, status, and trade details.
        """
        trade: PaperTrade | None = self._paper_exchange.get_trade(order_id)

        if trade is None:
            return {
                "order_id": order_id,
                "status": "NOT_FOUND",
                "error": "ORDER_NOT_FOUND",
            }

        return {
            "order_id": trade.id,
            "status": trade.status.value,
            "trade_id": trade.trade_id,
            "symbol": trade.symbol,
            "side": trade.side,
            "actual_quantity": trade.actual_quantity,
            "actual_price": trade.actual_price,
            "fees": trade.fees,
            "rejected_reason": trade.reject_reason,
        }

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Storniert Order via PaperExchange.cancel_trade().

        Nur Trades im CREATED/PENDING Status können storniert werden.
        """
        result = self._paper_exchange.cancel_trade(order_id)

        if not result.get("success"):
            return {
                "success": False,
                "order_id": order_id,
                "error": result.get("error", "UNKNOWN_ERROR"),
            }

        trade: PaperTrade = result["trade"]
        return {
            "success": True,
            "order_id": trade.id,
            "status": trade.status.value,
            "trade": {
                "trade_id": trade.trade_id,
                "symbol": trade.symbol,
                "side": trade.side,
                "actual_quantity": trade.actual_quantity,
                "actual_price": trade.actual_price,
                "fees": trade.fees,
            },
        }

    def get_balance(self, symbol: str) -> float:
        """Stub — kein echtes Guthaben im Paper-MVP."""
        return 100000.0

    def get_ticker(self, symbol: str) -> dict[str, float]:
        """Stub — keine echten Marktdaten im Paper-Adapter."""
        return {"bid": 0.0, "ask": 0.0, "last": 0.0}