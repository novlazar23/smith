"""ShadowExecutionBackend — PaperExecutionStack als Shadow-Trading-Execution (WI-ST-03).

Dünne Delegationsschicht zwischen Shadow-Trading (Spec §5) und dem
verdrahteten Paper-Execution-Stack:

- Nimmt ein TradeProposal entgegen und leitet es genau einmal an
  ``stack.paper_adapter.submit_order()`` weiter
- Mappt das Paper-Adapter-Ergebnis deterministisch auf das stabile
  ``ShadowExecutionResult``-Schema (Spec §5.2)
- Simuliert selbst weder Orders noch Fills — Fills, Positionen und PnL
  fließen ausschließlich über den ``PaperExecutionStack``
- Importiert und ruft keine Live-Exchange-Layer auf; die Isolation wird
  durch Spying-Tests abgesichert (``tests/test_shadow_execution_backend.py``)

Status-Mapping (deterministisch, Spec §5.2):

- Adapter-Status ``FILLED``   -> ``status="FILLED"`` mit
  ``trade_id=response["trade_id"]``,
  ``filled_price=float(response["actual_price"])``,
  ``quantity=float(response["actual_quantity"])``, ``reason=None``
- Adapter-Status ``REJECTED`` -> ``status="REJECTED"`` mit
  ``reason=str(response.get("error") or "REJECTED")``;
  ``trade_id``/``filled_price``/``quantity`` sind ``None``
- Jeder andere Adapter-Status -> fail-closed: ``status="ERROR"`` mit
  ``reason=f"UNEXPECTED_ADAPTER_STATUS: {status}"``; übrige Felder ``None``
- Ausnahme in ``submit_order`` -> fail-closed: ``status="ERROR"`` mit
  ``reason=f"EXECUTION_ERROR: {type(exc).__name__}"`` (Details via
  ``logger.exception``); übrige Felder ``None``
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from trading_harness.models import TradeProposal
from trading_harness.services.paper_execution_stack import PaperExecutionStack

logger = logging.getLogger(__name__)


class ShadowExecutionResult(BaseModel):
    """Ergebnis einer Shadow Execution gemäß Spec §5.2 (keine Defaults)."""

    trade_id: str | None
    status: str  # "FILLED" | "REJECTED" | "ERROR"
    filled_price: float | None
    quantity: float | None
    reason: str | None


class ShadowExecutionBackend:
    """Dünnes Wrapper-Interface für Shadow-Trading auf dem Paper-Stack.

    Einziger Delegationziel ist ``stack.paper_adapter.submit_order()``.
    Es gibt keine eigene Order-/Fill-Simulation und keinen Zugriff auf
    Live-Exchange-Endpunkte.
    """

    def __init__(self, stack: PaperExecutionStack) -> None:
        self._stack = stack

    def execute(self, proposal: TradeProposal) -> ShadowExecutionResult:
        """Führt das Proposal auf dem Paper-Stack aus und mappt das Ergebnis."""
        try:
            response = self._stack.paper_adapter.submit_order(
                symbol=proposal.symbol,
                side=proposal.side,
                quantity=proposal.requested_quantity,
                price=proposal.entry_price,
            )
        except Exception as exc:  # Shadow-Loop darf nicht crashen; Details im Log
            logger.exception(
                "Shadow execution fehlgeschlagen: decision_id=%s symbol=%s",
                proposal.decision_id,
                proposal.symbol,
            )
            return ShadowExecutionResult(
                trade_id=None,
                status="ERROR",
                filled_price=None,
                quantity=None,
                reason=f"EXECUTION_ERROR: {type(exc).__name__}",
            )
        return self._map_response(response)

    @staticmethod
    def _map_response(response: dict[str, Any]) -> ShadowExecutionResult:
        """Mappt die Paper-Adapter-Antwort deterministisch auf ShadowExecutionResult (Spec §5.2)."""
        status = str(response.get("status", ""))

        if status == "FILLED":
            return ShadowExecutionResult(
                trade_id=response["trade_id"],
                status="FILLED",
                filled_price=float(response["actual_price"]),
                quantity=float(response["actual_quantity"]),
                reason=None,
            )

        if status == "REJECTED":
            return ShadowExecutionResult(
                trade_id=None,
                status="REJECTED",
                filled_price=None,
                quantity=None,
                reason=str(response.get("error") or "REJECTED"),
            )

        # Fail-closed: jeder unerwartete Adapter-Status wird als ERROR gemeldet.
        return ShadowExecutionResult(
            trade_id=None,
            status="ERROR",
            filled_price=None,
            quantity=None,
            reason=f"UNEXPECTED_ADAPTER_STATUS: {status}",
        )
