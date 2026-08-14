from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from trading_harness.models import PaperTrade, PaperTradeStatus, TradeProposal

if TYPE_CHECKING:
    from trading_harness.services.paper_trade_store import (
        PaperTradeStore,
    )
    from trading_harness.services.risk_engine import RiskEngine


class PaperExchange:
    """Simulates order execution with configurable fill rate and fee rate.

    Deterministic only - no random without seed. Thread-safe via RLock.
    """

    def __init__(
        self,
        fill_rate: float = 0.8,
        fee_rate: float = 0.001,
        risk_engine: RiskEngine | None = None,
        stores: PaperTradeStore | None = None,
    ) -> None:
        if not 0.0 <= fill_rate <= 1.0:
            raise ValueError(
                f"fill_rate must be between 0 and 1, got {fill_rate}"
            )
        if fee_rate < 0:
            raise ValueError(f"fee_rate must be >= 0, got {fee_rate}")

        self.fill_rate = fill_rate
        self.fee_rate = fee_rate
        self.risk_engine = risk_engine
        self.stores = stores  # type: ignore[assignment]
        self._lock = threading.RLock()

    def execute_order(
        self,
        proposal: TradeProposal,
        current_price: float,
        fill_rate_override: float | None = None,
    ) -> PaperTrade:
        """Simulates order execution for a TradeProposal.

        Args:
            proposal: The validated trade proposal.
            current_price: The live market price at execution time.
            fill_rate_override: Optional override for the configured fill rate.

        Returns:
            A PaperTrade with execution details or rejection status.
        """
        trade = self._build_trade(proposal, current_price, fill_rate_override)
        with self._lock:
            self.stores.add(trade)
        return trade

    def _build_trade(
        self,
        proposal: TradeProposal,
        current_price: float,
        fill_rate_override: float | None = None,
    ) -> PaperTrade:
        """Build a PaperTrade from a proposal, applying all business rules."""
        # Validate price
        if current_price <= 0:
            return self._rejected(proposal, "INVALID_PRICE")

        # Validate symbol
        if not proposal.symbol:
            return self._rejected(proposal, "MISSING_SYMBOL")

        # Check policy
        if self.risk_engine:
            allowed_symbols = set(
                self.risk_engine.policy.get("allowed_symbols", [])
            )
            if proposal.symbol not in allowed_symbols:
                return self._rejected(proposal, "SYMBOL_NOT_ALLOWED")

        # Deterministic slippage
        slippage = abs(current_price * proposal.expected_slippage_bps / 10000)
        if proposal.side.upper() in ("LONG", "BUY"):
            actual_price = current_price + slippage
        else:
            actual_price = current_price - slippage

        # Fill rate
        rate = (
            fill_rate_override
            if fill_rate_override is not None
            else self.fill_rate
        )
        actual_quantity = proposal.requested_quantity * rate

        # Fee
        fees = abs(actual_quantity * actual_price) * self.fee_rate

        return PaperTrade(
            trade_id=proposal.decision_id,
            run_id="run-1",
            symbol=proposal.symbol,
            side=proposal.side,
            equity=proposal.equity,
            entry_price=proposal.entry_price,
            requested_leverage=proposal.requested_leverage,
            requested_quantity=proposal.requested_quantity,
            actual_quantity=actual_quantity,
            actual_price=actual_price,
            stop_price=proposal.stop_price,
            target_price=proposal.target_price,
            fill_rate=rate,
            slippage_bps=proposal.expected_slippage_bps,
            fees=fees,
            status=PaperTradeStatus.FILLED,
        )

    def _rejected(
        self, proposal: TradeProposal, reason: str
    ) -> PaperTrade:
        """Build a rejected PaperTrade."""
        return PaperTrade(
            trade_id=proposal.decision_id,
            run_id="run-1",
            symbol=proposal.symbol,
            side=proposal.side,
            equity=proposal.equity,
            entry_price=proposal.entry_price,
            stop_price=proposal.stop_price,
            target_price=proposal.target_price,
            status=PaperTradeStatus.REJECTED,
            reject_reason=reason,
        )