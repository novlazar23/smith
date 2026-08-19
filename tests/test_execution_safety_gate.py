"""Tests für das Live Execution Safety Gate (docs/handoff.md Item 3).

Abgedeckte Anforderungen (docs/spec-phase5-live-execution.md):
- R5.3  Audit-Log: jeder Live-Trade-Versuch (FILLED/REJECTED/ERROR) wird
         persistiert; Credentials erscheinen nie im Audit-Log (R5.20).
- R5.7  Kill-Switch-Monitoring: kill_switch_status() meldet Zustand,
         letztes Toggle und Toggle-Anzahl; aktiver Kill Switch blockiert.
- R5.23 Maximal-Kapitaleinsatz standardmäßig auf den minimalen Test-Betrag
         (0.01 Einheiten) begrenzt, konfigurierbar.
- R5.24 Position Sizing wird auf das Kapital begrenzt:
         quantity > max_capital -> MAX_CAPITAL_EXCEEDED.
- Gate  verify_safety_gate() prüft Kill-Switch-Präsenz und plausible
         Kapitalgrenzen; activate_live() ist fail-closed (aktiviert nur bei
         bestandenem Gate).
"""

from __future__ import annotations

import json

import pytest

from trading_harness.services.credential_manager import CredentialManager
from trading_harness.services.exchange_adapter import (
    ExchangeAdapter,
    StubExchangeAdapter,
)
from trading_harness.services.execution_store import ExecutionLogStore
from trading_harness.services.kill_switch import KillSwitch
from trading_harness.services.live_execution_service import (
    ExecutionConfig,
    LiveExecutionService,
    SafetyGateResult,
)
from trading_harness.services.paper_exchange import PaperExchange
from trading_harness.services.paper_exchange_adapter import PaperExchangeAdapter
from trading_harness.services.paper_trade_store import InMemoryPaperTradeStore


def _paper_adapter() -> PaperExchangeAdapter:
    """PaperAdapter mit 100 % Fill-Rate und 0 Gebühren (deterministisch)."""
    pe = PaperExchange(fill_rate=1.0, fee_rate=0.0, stores=InMemoryPaperTradeStore())
    return PaperExchangeAdapter(paper_exchange=pe)


def _service(
    *,
    config: ExecutionConfig | None = None,
    exchange_adapter: ExchangeAdapter | None = None,
    kill_switch: KillSwitch | None = None,
    log_store: ExecutionLogStore | None = None,
    credential_manager: CredentialManager | None = None,
) -> LiveExecutionService:
    return LiveExecutionService(
        kill_switch=kill_switch if kill_switch is not None else KillSwitch(enabled=False),
        exchange_adapter=exchange_adapter if exchange_adapter is not None else _paper_adapter(),
        credential_manager=credential_manager,
        config=config or ExecutionConfig(live_execution_enabled=True),
        log_store=log_store,
    )


def _submit(
    svc: LiveExecutionService,
    decision_id: str,
    *,
    symbol: str = "BTCUSD",
    side: str = "LONG",
    quantity: float = 1.0,
    price: float = 50000.0,
) -> dict:
    return svc.submit_order(
        decision_id=decision_id,
        run_id="run-1",
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
    )


# ---------------------------------------------------------------------------
# R5.3 + R5.20 — Audit-Log: persistiert, ohne Credentials
# ---------------------------------------------------------------------------


class TestAuditLogPersistence:
    def test_filled_order_persisted_to_store(self) -> None:
        store = ExecutionLogStore()
        svc = _service(
            config=ExecutionConfig(
                live_execution_enabled=True, min_capital=0.01, max_capital=10.0
            ),
            log_store=store,
        )

        result = _submit(svc, "d-persist-filled")

        assert result["status"] == "FILLED"
        entries = store.get_all()
        assert len(entries) == 1
        assert entries[0]["decision_id"] == "d-persist-filled"
        assert entries[0]["symbol"] == "BTCUSD"
        assert entries[0]["status"] == "FILLED"
        assert entries[0]["order_id"] is not None

    def test_rejected_order_persisted_to_store(self) -> None:
        store = ExecutionLogStore()
        svc = _service(
            config=ExecutionConfig(
                live_execution_enabled=True,
                min_capital=0.01,
                max_capital=10.0,
                symbol_whitelist=["BTCUSD"],
            ),
            log_store=store,
        )

        # DOGEUSD steht nicht in der SymbolWhitelist -> REJECTED
        result = _submit(svc, "d-persist-rejected", symbol="DOGEUSD")

        assert result["status"] == "REJECTED"
        assert result["error"] == "SYMBOL_NOT_WHITELISTED"
        entries = store.get_all()
        assert len(entries) == 1
        assert entries[0]["decision_id"] == "d-persist-rejected"
        assert entries[0]["status"] == "REJECTED"

    def test_error_result_persisted_to_store(self) -> None:
        store = ExecutionLogStore()
        svc = _service(
            config=ExecutionConfig(
                live_execution_enabled=True, min_capital=0.01, max_capital=10.0
            ),
            exchange_adapter=StubExchangeAdapter(),  # NOT_IMPLEMENTED -> ERROR
            log_store=store,
        )

        result = _submit(svc, "d-persist-error")

        assert result["status"] == "ERROR"
        entries = store.get_all()
        assert len(entries) == 1
        assert entries[0]["decision_id"] == "d-persist-error"
        assert entries[0]["status"] == "ERROR"

    def test_credentials_never_persisted_to_store(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TRADE_API_KEY", "super-secret-key-123")
        monkeypatch.setenv("TRADE_API_SECRET", "super-secret-value-456")
        credentials = CredentialManager()
        assert credentials.is_configured("TRADE_API_KEY") is True
        assert credentials.is_configured("TRADE_API_SECRET") is True

        store = ExecutionLogStore()
        svc = _service(
            config=ExecutionConfig(
                live_execution_enabled=True, min_capital=0.01, max_capital=10.0
            ),
            log_store=store,
            credential_manager=credentials,
        )

        result = _submit(svc, "d-persist-nocreds")
        assert result["status"] == "FILLED"

        for entry in store.get_all():
            blob = json.dumps(entry, default=str)
            assert "super-secret-key-123" not in blob
            assert "super-secret-value-456" not in blob


# ---------------------------------------------------------------------------
# R5.7 — Kill-Switch-Monitoring
# ---------------------------------------------------------------------------


class TestKillSwitchMonitoring:
    def test_kill_switch_status_reports_initial_state(self) -> None:
        svc = _service(
            config=ExecutionConfig(live_execution_enabled=True),
            kill_switch=KillSwitch(enabled=False),
        )

        status = svc.kill_switch_status()

        assert status["enabled"] is False
        assert status["toggle_count"] == 0
        assert "last_toggled_at" in status

    def test_kill_switch_status_reflects_toggle(self) -> None:
        ks = KillSwitch(enabled=False)
        svc = _service(
            config=ExecutionConfig(live_execution_enabled=True),
            kill_switch=ks,
        )

        ks.activate()

        status = svc.kill_switch_status()
        assert status["enabled"] is True
        assert status["toggle_count"] == 1

    def test_active_kill_switch_still_blocks_orders(self) -> None:
        svc = _service(
            config=ExecutionConfig(
                live_execution_enabled=True, min_capital=0.01, max_capital=10.0
            ),
            kill_switch=KillSwitch(enabled=True),
        )

        result = _submit(svc, "d-killblock")

        assert result["status"] == "REJECTED"
        assert result["error"] == "KILL_SWITCH_ACTIVE"


# ---------------------------------------------------------------------------
# R5.23 + R5.24 — Maximal-Kapitaleinsatz (Einheiten)
# ---------------------------------------------------------------------------


class TestMaxCapitalCap:
    def test_quantity_above_max_capital_rejected(self) -> None:
        svc = _service(
            config=ExecutionConfig(
                live_execution_enabled=True, min_capital=0.01, max_capital=0.01
            ),
        )

        # quantity=1.0 > max_capital=0.01 -> blockiert
        result = _submit(svc, "d-maxcap-exceed")

        assert result["status"] == "REJECTED"
        assert result["error"] == "MAX_CAPITAL_EXCEEDED"

    def test_quantity_within_max_capital_passes(self) -> None:
        svc = _service(
            config=ExecutionConfig(
                live_execution_enabled=True, min_capital=0.01, max_capital=10.0
            ),
        )

        result = _submit(svc, "d-maxcap-ok")

        assert result["status"] == "FILLED"

    def test_default_cap_equals_min_capital(self) -> None:
        # max_capital=None -> Standard-Cap = min_capital = 0.01 Einheiten
        svc = _service(
            config=ExecutionConfig(
                live_execution_enabled=True, min_capital=0.01, max_capital=None
            ),
        )

        # quantity=1.0 > 0.01 -> blockiert
        result = _submit(svc, "d-maxcap-default-exceed")

        assert result["status"] == "REJECTED"
        assert result["error"] == "MAX_CAPITAL_EXCEEDED"

    def test_minimal_test_amount_allowed_by_default(self) -> None:
        # R5.23: standardmäßig darf exakt der minimale Test-Betrag (0.01
        # Einheiten) durchkommen — der gewünschte Sicherheitsdefault.
        svc = _service(
            config=ExecutionConfig(
                live_execution_enabled=True, min_capital=0.01, max_capital=None
            ),
        )

        result = _submit(svc, "d-min-test-amount", quantity=0.01)

        assert result["status"] == "FILLED"
        assert result["order_id"] is not None


# ---------------------------------------------------------------------------
# Safety Gate + fail-closed activate_live()
# ---------------------------------------------------------------------------


class TestSafetyGate:
    def test_verify_safety_gate_passes_with_safe_defaults(self) -> None:
        svc = _service(config=ExecutionConfig(live_execution_enabled=False))

        result: SafetyGateResult = svc.verify_safety_gate()

        assert isinstance(result, SafetyGateResult)
        assert result.ready is True
        assert result.failed == []

    def test_verify_safety_gate_fails_on_nonpositive_min_capital(self) -> None:
        svc = _service(
            config=ExecutionConfig(live_execution_enabled=False, min_capital=0.0)
        )

        result = svc.verify_safety_gate()

        assert result.ready is False
        assert "min_capital_positive" in result.failed

    def test_verify_safety_gate_fails_on_nonpositive_max_capital(self) -> None:
        svc = _service(
            config=ExecutionConfig(
                live_execution_enabled=False, min_capital=0.01, max_capital=0.0
            )
        )

        result = svc.verify_safety_gate()

        assert result.ready is False
        assert "max_capital_valid" in result.failed

    def test_activate_live_is_fail_closed_when_gate_fails(self) -> None:
        svc = _service(
            config=ExecutionConfig(live_execution_enabled=False, min_capital=0.0)
        )
        assert svc.is_live_enabled is False

        result = svc.activate_live()

        assert isinstance(result, SafetyGateResult)
        assert result.ready is False
        assert svc.is_live_enabled is False  # fail-closed: nicht aktiviert

    def test_activate_live_enables_when_gate_passes(self) -> None:
        svc = _service(config=ExecutionConfig(live_execution_enabled=False))
        assert svc.is_live_enabled is False

        result = svc.activate_live()

        assert result.ready is True
        assert svc.is_live_enabled is True

    def test_deactivate_live_still_works(self) -> None:
        svc = _service(config=ExecutionConfig(live_execution_enabled=False))
        svc.activate_live()
        assert svc.is_live_enabled is True

        svc.deactivate_live()
        assert svc.is_live_enabled is False
