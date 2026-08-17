"""ShadowModeLogger — logs orders without executing them (shadow trading).

Records every trade attempt with simulated outcomes for backtesting
and validation. Does not touch the exchange.
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from trading_harness.services.exchange_adapter import (
    ExchangeAdapter,
    ExchangeAdapterError,
)

logger = logging.getLogger(__name__)


class ShadowTradeRecord(BaseModel):
    """Ein einzelner Shadow-Trade-Eintrag."""

    id: str = Field(default_factory=lambda: f"shadow-{uuid.uuid4().hex[:8]}")
    decision_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    order_type: str = "MARKET"
    simulated_status: str = "FILLED"
    simulated_fill_price: float = 0.0
    simulated_slippage: float = 0.0
    simulated_commission: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    run_id: str = ""

    @property
    def pnl_estimate(self) -> float:
        """Geschätzter PnL basierend auf Slippage + Commission."""
        direction = 1 if self.side.upper() in ("BUY", "LONG") else -1
        return direction * (self.simulated_fill_price - self.price) * self.quantity


class ShadowModeLogger:
    """Shadow-Mode-Logger für Backtesting und Validierung.

    Loggt alle Orders ohne sie auszuführen. Simulierte Fills basierend
    auf Slippage-Modell (0.05% Standard).
    """

    def __init__(
        self,
        default_slippage: float = 0.0005,
        default_commission: float = 0.001,
        auto_fill_price: bool = True,
    ) -> None:
        self._slippage = default_slippage
        self._commission = default_commission
        self._auto_fill_price = auto_fill_price
        self._records: list[ShadowTradeRecord] = []
        self._lock = threading.Lock()

    def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_type: str = "MARKET",
    ) -> dict[str, Any]:
        """Shadow-Mode submit_order — loggt Order, gibt simulierte Antwort."""
        record = self.log_order(
            decision_id=f"shadow-{uuid.uuid4().hex[:8]}",
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            order_type=order_type,
        )
        return {
            "order_id": record.id,
            "status": record.simulated_status,
            "filled_price": record.simulated_fill_price,
            "slippage": record.simulated_slippage,
            "commission": record.simulated_commission,
        }

    def log_order(
        self,
        decision_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_type: str = "MARKET",
        run_id: str = "",
    ) -> ShadowTradeRecord:
        """Loggt eine Order im Shadow-Mode."""
        fill_price = price
        if self._auto_fill_price:
            # Simulierte Fill-Preis-Berechnung mit Slippage
            slippage = price * self._slippage * (1 if side.upper() in ("BUY", "LONG") else -1)
            fill_price = price + slippage

        commission = fill_price * quantity * self._commission

        record = ShadowTradeRecord(
            decision_id=decision_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            order_type=order_type,
            simulated_status="FILLED",
            simulated_fill_price=fill_price,
            simulated_slippage=abs(fill_price - price),
            simulated_commission=commission,
            run_id=run_id,
        )

        with self._lock:
            self._records.append(record)
        logger.info(
            "ShadowOrder: %s %s %.4f @ %.2f (filled @ %.2f, slippage: %.4f%%, commission: %.4f)",
            side, symbol, quantity, price, fill_price,
            record.simulated_slippage / price * 100 if price else 0,
            commission,
        )
        return record

    def get_records(
        self,
        decision_id: str | None = None,
        symbol: str | None = None,
        run_id: str | None = None,
    ) -> list[ShadowTradeRecord]:
        """Shadow-Records abrufen (optional gefiltert)."""
        with self._lock:
            records = self._records[:]
        if decision_id:
            records = [r for r in records if r.decision_id == decision_id]
        if symbol:
            records = [r for r in records if r.symbol == symbol]
        if run_id:
            records = [r for r in records if r.run_id == run_id]
        return records

    @property
    def record_count(self) -> int:
        with self._lock:
            return len(self._records)

    @property
    def total_commission(self) -> float:
        with self._lock:
            return sum(r.simulated_commission for r in self._records)

    @property
    def total_slippage(self) -> float:
        with self._lock:
            return sum(r.simulated_slippage for r in self._records)

    @property
    def estimated_pnl(self) -> float:
        with self._lock:
            return sum(r.pnl_estimate for r in self._records)

    def summary(self) -> dict[str, Any]:
        """Zusammenfassung aller Shadow-Orders."""
        with self._lock:
            total = len(self._records)
            fills = sum(1 for r in self._records if r.simulated_status == "FILLED")
            rejected = sum(1 for r in self._records if r.simulated_status == "REJECTED")
        return {
            "total_orders": total,
            "filled": fills,
            "rejected": rejected,
            "total_slippage": self.total_slippage,
            "total_commission": self.total_commission,
            "estimated_pnl": self.estimated_pnl,
        }


class ShadowModeAdapter(ExchangeAdapter):
    """ExchangeAdapter, der ShadowModeLogger als Fallback verwendet.

    Wenn live_execution=True, wird die Order an den echten Adapter gesendet.
    Wenn live_execution=False oder Fehler, wird die Order geloggt.
    """

    def __init__(
        self,
        delegate: ExchangeAdapter | None = None,
        shadow: ShadowModeLogger | None = None,
    ) -> None:
        self._delegate = delegate
        self._shadow = shadow or ShadowModeLogger()
        self._name = delegate.name if delegate else "SHADOW"

    @property
    def name(self) -> str:
        return self._name

    def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_type: str = "MARKET",
    ) -> dict[str, Any]:
        if self._delegate:
            try:
                return self._delegate.submit_order(
                    symbol, side, quantity, price, order_type
                )
            except ExchangeAdapterError as exc:
                logger.warning("Delegate failed, falling back to shadow: %s", exc)
        return self._shadow.submit_order(
            symbol, side, quantity, price, order_type
        )

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        if self._delegate:
            return self._delegate.get_order_status(order_id)
        return {"status": "UNKNOWN", "error": "NO_DELEGATE"}

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        if self._delegate:
            return self._delegate.cancel_order(order_id)
        return {"success": False, "error": "NO_DELEGATE"}

    def get_balance(self, symbol: str) -> float:
        if self._delegate:
            return self._delegate.get_balance(symbol)
        return 100000.0

    def get_ticker(self, symbol: str) -> dict[str, float]:
        if self._delegate:
            return self._delegate.get_ticker(symbol)
        return {"bid": 50000.0, "ask": 50001.0, "last": 50000.5}