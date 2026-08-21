"""Tests: ShadowExecutionBackend — PaperExecutionStack als Shadow-Trading-Execution (WI-ST-03).

Das Backend ist eine dünne Delegationsschicht auf den verdrahteten
PaperExecutionStack. Diese Tests sichern ab:

- ShadowExecutionResult-Schema (Spec §5.2): exakt die Felder
  trade_id/status/filled_price/quantity/reason in der Spec-Reihenfolge,
  ohne Defaults; status ist ein gewöhnliches str-Feld ("FILLED" |
  "REJECTED" | "ERROR")
- Vollständiger Fill-Fluss (Spec ST.5): BUY-Proposal -> "FILLED", genau
  eine offene PaperPosition, Portfolio-Equity-Fluss
- Delegation: execute() ruft nur stack.paper_adapter.submit_order() auf
- Ergebnis-Mapping (Spec §5.2): Adapter-FILLED -> "FILLED" (Fill-Daten),
  Adapter-REJECTED -> "REJECTED" (Reason aus Adapter-Error), unerwarteter
  Adapter-Status -> "ERROR" (fail-closed), Adapter-Ausnahme -> "ERROR"
- Isolation (harte Grenze): execute() erreicht weder die Crypto-
  Execution-Layer (CryptoExecutionRouter, alle Crypto-Adapter) noch die
  übrigen Exchange-Adapter-Write-Pfade (submit/cancel/status/balance)
- Import-Guard: das Backend-Modul importiert keine Exchange- oder
  Live-Execution-Pfade
"""

from __future__ import annotations

import ast
import importlib
import typing
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from pydantic_core import PydanticUndefined

from trading_harness.models import PaperPositionStatus, TradeProposal
from trading_harness.services.crypto_exchange_adapter import (
    BaseCryptoExchangeAdapter,
    BinanceExchangeAdapter,
    BitgetExchangeAdapter,
    BybitExchangeAdapter,
    CoinbaseExchangeAdapter,
    CryptoExecutionRouter,
)
from trading_harness.services.exchange_adapter import ExchangeAdapter, StubExchangeAdapter
from trading_harness.services.paper_execution_stack import build_paper_execution_stack
from trading_harness.services.shadow_execution_backend import (
    ShadowExecutionBackend,
    ShadowExecutionResult,
)
from trading_harness.services.shadow_mode_logger import ShadowModeAdapter

#: Exchange-Layer-Klassen, die von Shadow Execution niemals berührt werden dürfen.
SPY_TARGET_CLASSES: list[type] = [
    CryptoExecutionRouter,
    BaseCryptoExchangeAdapter,
    BybitExchangeAdapter,
    BitgetExchangeAdapter,
    BinanceExchangeAdapter,
    CoinbaseExchangeAdapter,
    ExchangeAdapter,
    StubExchangeAdapter,
    ShadowModeAdapter,
]

#: Endpunkte, die von Shadow Execution niemals aufgerufen werden dürfen.
SPY_METHODS = ("submit_order", "cancel_order", "get_order_status", "get_balance")

#: Modulpfad-Teile, die das Backend-Modul niemals importieren darf.
FORBIDDEN_IMPORT_PARTS = (
    "crypto",
    "router",
    "live_execution",
    "exchange_adapter",
    "shadow_mode",
    "bybit",
    "binance",
    "bitget",
    "coinbase",
    "stub",
)


def _buy_proposal(**overrides: Any) -> TradeProposal:
    """Gültiges BUY-Proposal für den Test-Use-Case (BTCUSDT @ 100.0)."""
    params: dict[str, Any] = {
        "decision_id": "decision-st-03",
        "symbol": "BTCUSDT",
        "side": "BUY",
        "equity": 100000.0,
        "entry_price": 100.0,
        "stop_price": 95.0,
        "target_price": 105.0,
        "requested_quantity": 0.01,
    }
    params.update(overrides)
    return TradeProposal(**params)


class _SpyPaperAdapter:
    """Adapter-Doppel: liefert ein vorgegebenes Submit-Ergebnis oder wirft."""

    def __init__(
        self,
        response: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._response = response
        self._error = error

    def submit_order(self, **kwargs: Any) -> dict[str, Any]:
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return dict(self._response)

    def get_order_status(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {}

    def cancel_order(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {}

    def get_balance(self, *args: Any, **kwargs: Any) -> float:
        return 0.0


class _SpyStack:
    """Stack-Doppel: stellt nur die Eigenschaft bereit, die das Backend nutzt."""

    def __init__(self, adapter: _SpyPaperAdapter) -> None:
        self.paper_adapter = adapter


def _filled_response(trade_id: str = "trade-spy-1", price: float = 100.0) -> dict[str, Any]:
    return {
        "order_id": "paper-order-1",
        "status": "FILLED",
        "trade_id": trade_id,
        "actual_quantity": 0.008,
        "actual_price": price,
        "fees": 0.0008,
        "error": None,
    }


def _make_spy(label: str, recorded: list[str]) -> Any:
    """Fabrik für Counting-Spies (klassenweit gepatcht: self als erstes Arg).

    Jeder Aufruf wird mit dem Label des Endpunkts protokolliert. Der Rückgabe-
    wert ist irrelevant: im korrekten Verhalten werden die Spies nie aufgerufen.
    """

    def _spy(*args: Any, **kwargs: Any) -> dict[str, Any]:
        recorded.append(label)
        return {"status": "REJECTED", "error": "SPY_ENDPOINT_MUST_NOT_BE_CALLED"}

    return _spy


class TestShadowExecutionResultSchema:
    """Schema nach Spec §5.2: stabile Felder, Spec-Reihenfolge, keine Defaults."""

    def test_result_fields_and_spec_order(self) -> None:
        """Exakt die fünf Spec-Felder in der Reihenfolge von Spec §5.2."""
        assert set(ShadowExecutionResult.model_fields) == {
            "trade_id",
            "status",
            "filled_price",
            "quantity",
            "reason",
        }
        assert list(ShadowExecutionResult.model_fields) == [
            "trade_id",
            "status",
            "filled_price",
            "quantity",
            "reason",
        ]

    def test_result_fields_have_no_defaults(self) -> None:
        """Kein Feld darf einen Default haben (Spec §5.2: keine Defaults)."""
        for name, field in ShadowExecutionResult.model_fields.items():
            assert field.is_required(), f"{name} hat einen unerlaubten Default"
            assert field.default is PydanticUndefined, f"{name} hat einen unerlaubten Default"
            assert field.default_factory is None, f"{name} hat eine unerlaubte Default-Fabrik"

    def test_status_is_plain_str_and_explicit_construction(self) -> None:
        """status ist ein gewöhnliches str-Feld (kein Enum); alle Felder explizit."""
        assert typing.get_type_hints(ShadowExecutionResult)["status"] is str

        result = ShadowExecutionResult(
            trade_id="trade-1",
            status="FILLED",
            filled_price=100.0,
            quantity=0.008,
            reason=None,
        )
        assert result.trade_id == "trade-1"
        assert type(result.status) is str
        assert result.status == "FILLED"
        assert result.filled_price == 100.0
        assert result.quantity == 0.008
        assert result.reason is None


class TestShadowBackendPaperFill:
    """Spec ST.5: ein BUY-Proposal fließt vollständig durch den Paper-Stack."""

    def test_shadow_backend_executes_paper_fill(self) -> None:
        """BUY-Proposal -> "FILLED" mit genau einer offenen Position und Equity-Fluss."""
        stack = build_paper_execution_stack()
        backend = ShadowExecutionBackend(stack)

        result = backend.execute(_buy_proposal())

        assert result.status == "FILLED"
        assert result.filled_price == pytest.approx(100.0)
        assert result.quantity == pytest.approx(0.008)
        assert result.trade_id is not None
        assert result.reason is None

        positions = [
            p for p in stack.position_manager.get_open_positions()
            if p.trade_id == result.trade_id
        ]
        assert len(positions) == 1
        position = positions[0]
        assert position.status is PaperPositionStatus.OPEN
        assert position.symbol == "BTCUSDT"
        assert position.quantity == pytest.approx(0.01 * 0.8)

        states = stack.portfolio_store.all()
        assert len(states) == 1
        assert states[0].positions["BTCUSDT"] == pytest.approx(0.008)
        assert states[0].current_equity == pytest.approx(100000.0)

        stack.position_manager.update_price(position.id, 105.0)
        state = stack.portfolio_tracker.update(stack.position_manager.get_open_positions())
        assert state.current_equity == pytest.approx(100000.04)


class TestShadowBackendDelegation:
    """Das Backend delegiert ausschließlich an stack.paper_adapter.submit_order()."""

    def test_execute_delegates_only_to_paper_adapter_submit(self) -> None:
        """submit_order wird genau einmal mit den Proposal-Parametern aufgerufen."""
        stack = build_paper_execution_stack()
        calls: dict[str, list[tuple[tuple[Any, ...], dict[str, Any]]]] = {
            name: [] for name in SPY_METHODS
        }

        def _instance_spy(name: str) -> Any:
            def _spy(*args: Any, **kwargs: Any) -> dict[str, Any]:
                calls[name].append((args, kwargs))
                return _filled_response()

            return _spy

        backend = ShadowExecutionBackend(stack)
        with mock.patch.object(stack.paper_adapter, "submit_order", _instance_spy("submit_order")), \
                mock.patch.object(stack.paper_adapter, "cancel_order", _instance_spy("cancel_order")), \
                mock.patch.object(
                    stack.paper_adapter, "get_order_status", _instance_spy("get_order_status")
                ), \
                mock.patch.object(stack.paper_adapter, "get_balance", _instance_spy("get_balance")):
            result = backend.execute(_buy_proposal())

        assert result.status == "FILLED"
        assert len(calls["submit_order"]) == 1
        args, kwargs = calls["submit_order"][0]
        assert kwargs == {"symbol": "BTCUSDT", "side": "BUY", "quantity": 0.01, "price": 100.0}
        assert args == ()
        for name in ("cancel_order", "get_order_status", "get_balance"):
            assert calls[name] == []


class TestShadowBackendResultMapping:
    """Deterministisches Mapping der Paper-Adapter-Ergebnisse (Spec §5.2)."""

    def test_filled_maps_to_filled(self) -> None:
        """FILLED-Antwort -> "FILLED" mit Fill-Preis, Fill-Menge und trade_id."""
        backend = ShadowExecutionBackend(_SpyStack(_SpyPaperAdapter(response=_filled_response())))

        result = backend.execute(_buy_proposal())

        assert result.status == "FILLED"
        assert result.filled_price == pytest.approx(100.0)
        assert result.quantity == pytest.approx(0.008)
        assert result.trade_id == "trade-spy-1"
        assert result.reason is None

    def test_rejected_invalid_side_maps_to_rejected(self) -> None:
        """REJECTED/INVALID_SIDE -> "REJECTED" mit Reason, kein Fill."""
        stack = build_paper_execution_stack()
        backend = ShadowExecutionBackend(stack)

        result = backend.execute(_buy_proposal(side="HOLD"))

        assert result.status == "REJECTED"
        assert result.reason == "INVALID_SIDE"
        assert result.filled_price is None
        assert result.quantity is None
        assert result.trade_id is None
        assert stack.position_manager.get_open_positions() == []

    def test_rejected_missing_symbol_maps_to_rejected(self) -> None:
        """REJECTED/MISSING_SYMBOL -> "REJECTED" mit Reason, kein Fill."""
        stack = build_paper_execution_stack()
        backend = ShadowExecutionBackend(stack)

        result = backend.execute(_buy_proposal(symbol=""))

        assert result.status == "REJECTED"
        assert result.reason == "MISSING_SYMBOL"
        assert result.filled_price is None
        assert result.quantity is None
        assert result.trade_id is None
        assert stack.position_manager.get_open_positions() == []

    def test_unexpected_adapter_status_maps_to_error(self) -> None:
        """Unerwarteter Adapter-Status (z. B. PENDING) -> fail-closed "ERROR"."""
        pending = {
            "order_id": "paper-order-pending",
            "status": "PENDING",
            "trade_id": "trade-pending",
            "actual_quantity": 0.0,
            "actual_price": 0.0,
            "fees": 0.0,
            "error": None,
        }
        backend = ShadowExecutionBackend(_SpyStack(_SpyPaperAdapter(response=pending)))

        result = backend.execute(_buy_proposal())

        assert result.status == "ERROR"
        assert result.reason == "UNEXPECTED_ADAPTER_STATUS: PENDING"
        assert result.filled_price is None
        assert result.quantity is None
        assert result.trade_id is None

    def test_adapter_exception_maps_to_error(self) -> None:
        """Adapter-Ausnahme -> fail-closed "ERROR" mit Typ-Reason, kein Crash."""
        backend = ShadowExecutionBackend(
            _SpyStack(_SpyPaperAdapter(error=RuntimeError("boom")))
        )

        result = backend.execute(_buy_proposal())

        assert result.status == "ERROR"
        assert result.reason == "EXECUTION_ERROR: RuntimeError"
        assert result.filled_price is None
        assert result.quantity is None
        assert result.trade_id is None


class TestShadowBackendIsolation:
    """Harte Grenze: Shadow Execution erreicht keine Exchange-/Live-Write-Pfade."""

    def test_execute_never_reaches_exchange_write_endpoints(self) -> None:
        """Harte Grenze: 0 Aufrufe aller Exchange-Layer-Endpunkte; Fill fließt."""
        stack = build_paper_execution_stack()
        backend = ShadowExecutionBackend(stack)
        recorded: list[str] = []

        patches = [
            mock.patch.object(
                cls, name, _make_spy(f"{cls.__name__}.{name}", recorded), create=True
            )
            for cls in SPY_TARGET_CLASSES
            for name in SPY_METHODS
        ]
        # Abdeckungssicherung: 9 Exchange-Layer-Klassen x 4 Endpunkte = 36 Spies.
        assert len(patches) == len(SPY_TARGET_CLASSES) * len(SPY_METHODS) == 36
        for patch in patches:
            patch.start()
        try:
            result = backend.execute(_buy_proposal())
        finally:
            for patch in reversed(patches):
                patch.stop()

        assert recorded == [], f"Isolation verletzt, aufgerufene Endpunkte: {recorded}"
        assert result.status == "FILLED"
        assert result.filled_price == pytest.approx(100.0)

    def test_backend_module_imports_no_exchange_or_live_execution_paths(self) -> None:
        """Das Backend-Modul importiert keine Exchange- oder Live-Execution-Pfade."""
        module = importlib.import_module("trading_harness.services.shadow_execution_backend")
        source = Path(module.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)

        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module)

        offenders = sorted(
            module_name
            for module_name in imported
            if any(part in module_name for part in FORBIDDEN_IMPORT_PARTS)
        )
        assert not offenders, f"Backend importiert verbotene Pfade: {offenders}"
