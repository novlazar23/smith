"""Regressionstests: Paper-Execution-Wiring (WI-P4-4/WI-P4-5).

Die ausgelieferte Paper-Wiring in `api/routes.py` baute `PaperExchange()`
ohne Stores: jede Order endete in `RuntimeError("PaperExchange stores
not configured")` (3x in Folge -> Kill-Switch-Auto-Trigger), und
PositionManager/PortfolioTracker waren nie verdrahtet.

Diese Tests stellen sicher, dass der Paper-Execution-Stack wie ausgeliefert
funktionsfähig ist:

- routes.py baut den Stack über die Factory `build_paper_execution_stack`
- PaperExchange hat konfigurierte Stores
- PaperExchangeAdapter() ohne Parameter ist sicher (In-Memory-Store)
- Ein Fill fließt komplett: TradeProposal -> PaperTrade (Store) ->
  PaperPosition (PositionManager) -> PortfolioState/PnL (PortfolioTracker)
"""

from __future__ import annotations

import pytest

from trading_harness.api import routes
from trading_harness.models import PaperPositionStatus, PaperTradeStatus
from trading_harness.services.paper_exchange_adapter import PaperExchangeAdapter
from trading_harness.services.paper_execution_stack import (
    build_paper_execution_stack,
)


class TestShippedPaperWiring:
    """Regression: die in routes.py ausgelieferte Paper-Wiring."""

    def test_routes_paper_exchange_has_stores(self) -> None:
        """PaperExchange in routes.py muss Stores konfiguriert haben."""
        assert routes._paper_exchange.stores is not None

    def test_routes_submit_returns_filled(self) -> None:
        """Ein gültiger Paper-Order über die routes-Wiring füllt."""
        result = routes._paper_adapter.submit_order(
            symbol="BTCUSDT", side="LONG", quantity=0.01, price=100.0
        )
        assert result["status"] == "FILLED"
        assert result["error"] is None
        assert result["order_id"] is not None

    def test_default_adapter_submit_is_safe(self) -> None:
        """PaperExchangeAdapter() ohne Parameter darf keinen RuntimeError werfen."""
        adapter = PaperExchangeAdapter()
        result = adapter.submit_order(
            symbol="BTCUSDT", side="LONG", quantity=0.01, price=100.0
        )
        assert result["status"] == "FILLED"
        assert result["order_id"] is not None

    def test_routes_fill_persists_trade(self) -> None:
        """Ein Fill über die routes-Wiring persistiert den PaperTrade."""
        result = routes._paper_adapter.submit_order(
            symbol="BTCUSDT", side="LONG", quantity=0.01, price=100.0
        )
        assert result["status"] == "FILLED"
        trade = routes.paper_execution_stack.trade_store.get(result["order_id"])
        assert trade is not None
        assert trade.status is PaperTradeStatus.FILLED
        assert trade.symbol == "BTCUSDT"

    def test_routes_fill_opens_position(self) -> None:
        """Ein Fill über die routes-Wiring öffnet eine PaperPosition."""
        result = routes._paper_adapter.submit_order(
            symbol="BTCUSDT", side="LONG", quantity=0.01, price=100.0
        )
        assert result["status"] == "FILLED"
        stack = routes.paper_execution_stack
        positions = [
            p for p in stack.position_manager.get_open_positions()
            if p.trade_id == result["trade_id"]
        ]
        assert len(positions) == 1
        assert positions[0].status is PaperPositionStatus.OPEN
        assert positions[0].symbol == "BTCUSDT"
        assert positions[0].quantity == pytest.approx(0.01 * 0.8)

    def test_routes_fill_updates_portfolio(self) -> None:
        """Ein Fill über die routes-Wiring aktualisiert den Portfolio-Status."""
        result = routes._paper_adapter.submit_order(
            symbol="BTCUSDT", side="LONG", quantity=0.01, price=100.0
        )
        assert result["status"] == "FILLED"
        states = routes.paper_execution_stack.portfolio_store.all()
        assert len(states) >= 1
        assert "BTCUSDT" in states[-1].positions

    def test_live_execution_remains_disabled(self) -> None:
        """Sicherheit: Live Execution bleibt deaktiviert, Pipeline intakt."""
        assert routes.execution_config.live_execution_enabled is False
        assert routes.live_execution_service._exchange_adapter is routes._paper_adapter


class TestPaperExecutionStackFactory:
    """Factory `build_paper_execution_stack` (aus routes.py extrahiert)."""

    def test_factory_builds_wired_components(self) -> None:
        stack = build_paper_execution_stack()
        assert stack.paper_exchange.stores is stack.trade_store
        assert stack.paper_adapter._paper_exchange is stack.paper_exchange
        assert stack.position_manager is not None
        assert stack.portfolio_tracker is not None
        assert stack.portfolio_tracker.start_equity == 100000.0

    def test_fill_flows_trade_position_pnl(self) -> None:
        stack = build_paper_execution_stack()
        result = stack.paper_adapter.submit_order(
            symbol="BTCUSDT", side="LONG", quantity=0.01, price=100.0
        )
        assert result["status"] == "FILLED"

        trade = stack.trade_store.get(result["order_id"])
        assert trade is not None
        assert trade.status is PaperTradeStatus.FILLED
        assert trade.actual_quantity == pytest.approx(0.008)

        positions = [
            p for p in stack.position_manager.get_open_positions()
            if p.trade_id == result["trade_id"]
        ]
        assert len(positions) == 1
        position = positions[0]
        assert position.status is PaperPositionStatus.OPEN
        assert position.entry_price == pytest.approx(100.0)
        assert position.quantity == pytest.approx(0.008)

        states = stack.portfolio_store.all()
        assert len(states) == 1
        assert states[0].positions["BTCUSDT"] == pytest.approx(0.008)
        assert states[0].current_equity == pytest.approx(100000.0)

    def test_price_update_flows_to_pnl(self) -> None:
        stack = build_paper_execution_stack()
        result = stack.paper_adapter.submit_order(
            symbol="BTCUSDT", side="LONG", quantity=0.01, price=100.0
        )
        assert result["status"] == "FILLED"

        position = next(
            p for p in stack.position_manager.get_open_positions()
            if p.trade_id == result["trade_id"]
        )
        stack.position_manager.update_price(position.id, 105.0)
        state = stack.portfolio_tracker.update(
            stack.position_manager.get_open_positions()
        )
        assert state.total_unrealized_pnl == pytest.approx(5.0 * 0.008)
        assert state.current_equity == pytest.approx(100000.04)

    def test_rejected_order_does_not_open_position(self) -> None:
        stack = build_paper_execution_stack()
        result = stack.paper_adapter.submit_order(
            symbol="BTCUSDT", side="LONG", quantity=0.01, price=0.0
        )
        assert result["status"] == "REJECTED"
        assert result["error"] == "INVALID_PRICE"
        assert stack.trade_store.all() == []
        assert stack.position_manager.get_open_positions() == []
        assert stack.portfolio_store.all() == []
