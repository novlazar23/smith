"""PaperExecutionStack — vollständige Paper-Execution-Wiring (WI-P4-4/WI-P4-5).

Bündelt die in `api/routes.py` ausgelieferte Paper-Execution-Wiring:

- ``PaperExchange`` mit persistenten ``Persisted*Store``s (PostgreSQL,
  In-Memory-Fallback ohne Datenbank)
- ``PositionManager`` — öffnet eine Position für jeden Fill
- ``PortfolioTracker`` — aktualisiert Portfolio-Status und PnL
- ``PaperExchangeAdapter`` — ExchangeAdapter-Schnittstelle, ruft bei
  jedem Fill ``handle_fill`` auf

Damit fließt ein Paper-Fill vollständig:
TradeProposal -> PaperTrade (Store) -> PaperPosition (PositionManager)
-> PortfolioState/PnL (PortfolioTracker).

Live Execution bleibt davon unberührt und standardmäßig deaktiviert.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from trading_harness.models import PaperTradeStatus
from trading_harness.services.paper_exchange import PaperExchange
from trading_harness.services.paper_exchange_adapter import PaperExchangeAdapter
from trading_harness.services.paper_trade_store import PersistedPaperTradeStore
from trading_harness.services.portfolio_tracker import (
    PersistedPortfolioStore,
    PortfolioTracker,
)
from trading_harness.services.position_manager import PositionManager
from trading_harness.services.position_stores import PersistedPaperPositionStore

if TYPE_CHECKING:
    from trading_harness.models import PaperTrade
    from trading_harness.services.db import Database

logger = logging.getLogger(__name__)


class PaperExecutionStack:
    """Verdrahteter Paper-Execution-Stack (Exchange + Stores + Positionen + PnL)."""

    def __init__(
        self,
        db: Database | None = None,
        start_equity: float = 100000.0,
    ) -> None:
        self.trade_store = PersistedPaperTradeStore(db)
        self.position_store = PersistedPaperPositionStore(db)
        self.portfolio_store = PersistedPortfolioStore(db)
        self.paper_exchange = PaperExchange(stores=self.trade_store)
        self.position_manager = PositionManager(store=self.position_store)
        self.portfolio_tracker = PortfolioTracker(
            start_equity=start_equity, store=self.portfolio_store
        )
        self.paper_adapter = PaperExchangeAdapter(
            paper_exchange=self.paper_exchange,
            on_fill=self.handle_fill,
        )

    def handle_fill(self, trade: PaperTrade) -> None:
        """Reagiert auf einen FILLED-PaperTrade: Position öffnen + PnL aktualisieren."""
        if trade.status is not PaperTradeStatus.FILLED or trade.actual_quantity <= 0:
            return
        position = self.position_manager.open_position(trade)
        state = self.portfolio_tracker.update(
            self.position_manager.get_open_positions()
        )
        logger.info(
            "Paper fill: trade_id=%s symbol=%s qty=%s price=%s position=%s equity=%s",
            trade.trade_id,
            trade.symbol,
            trade.actual_quantity,
            trade.actual_price,
            position.id,
            state.current_equity,
        )


def build_paper_execution_stack(db: Database | None = None) -> PaperExecutionStack:
    """Factory für den in `api/routes.py` ausgelieferten Paper-Execution-Stack."""
    return PaperExecutionStack(db=db)
