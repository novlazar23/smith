"""Tests für Read/Trade API Trennung (R5.21–R5.22)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from trading_harness.main import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Helper: build a mock settings object
# ---------------------------------------------------------------------------


def _mock_settings(
    read_api_key: str = "",
    trade_api_key: str = "",
) -> dict:
    return {
        "read_api_key": read_api_key,
        "trade_api_key": trade_api_key,
    }


# ---------------------------------------------------------------------------
# Unit: security module
# ---------------------------------------------------------------------------


class TestRequireTradeKey:
    """Unit tests für require_trade_key dependency."""

    def test_no_key_configured_allows_access(self):
        """Wenn trade_api_key nicht gesetzt, wird kein Key verlangt."""
        # Verhalten belegt durch Integrationstests in TestExecutionOrderWithAuth

    def test_key_from_header_valid(self):
        """Gültiger Trade-Key im Header wird akzeptiert."""
        from trading_harness.api.security import _verify_key

        # Keine Exception = OK
        _verify_key("correct-key", "correct-key")

    def test_key_from_header_invalid(self):
        """Ungültiger Trade-Key wirft 403."""
        from fastapi import HTTPException

        from trading_harness.api.security import _verify_key

        with pytest.raises(HTTPException) as exc_info:
            _verify_key("wrong-key", "correct-key")
        assert exc_info.value.status_code == 403

    def test_key_missing(self):
        """Fehlender Key wirft 401."""
        from fastapi import HTTPException

        from trading_harness.api.security import _verify_key

        with pytest.raises(HTTPException) as exc_info:
            _verify_key(None, "correct-key")
        assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# Integration: routes mit mock Config
# ---------------------------------------------------------------------------


class TestExecutionOrderWithAuth:
    """POST /execution/orders — Auth-Verhalten."""

    def test_order_no_key_no_config(self):
        """Kein Key konfiguriert — Order ohne Key erlaubt (backward compat)."""
        with patch(
            "trading_harness.api.security.get_settings",
            return_value=type(
                "MockSettings",
                (),
                {"read_api_key": "", "trade_api_key": ""},
            )(),
        ):
            response = client.post(
                "/execution/orders",
                json={
                    "decision_id": "dec-test-001",
                    "run_id": "run-1",
                    "symbol": "BTCUSDT",
                    "side": "LONG",
                    "quantity": 1.0,
                    "price": 50000.0,
                },
            )
            # Execution ist disabled, also REJECTED — NICHT 401
            assert response.status_code == 200
            assert response.json()["status"] == "REJECTED"

    def test_order_wrong_key_rejected(self):
        """Falscher Trade-Key → 403."""
        mock_settings = type(
            "MockSettings",
            (),
            {"read_api_key": "", "trade_api_key": "secret-trade-key"},
        )()
        with patch(
            "trading_harness.api.security.get_settings",
            return_value=mock_settings,
        ):
            response = client.post(
                "/execution/orders",
                json={
                    "decision_id": "dec-test-002",
                    "symbol": "BTCUSDT",
                    "side": "LONG",
                    "quantity": 1.0,
                    "price": 50000.0,
                },
                headers={"X-Trade-API-Key": "wrong-key"},
            )
            assert response.status_code == 403

    def test_order_missing_key_rejected(self):
        """Kein Key wenn Key verlangt wird → 401."""
        mock_settings = type(
            "MockSettings",
            (),
            {"read_api_key": "", "trade_api_key": "secret-trade-key"},
        )()
        with patch(
            "trading_harness.api.security.get_settings",
            return_value=mock_settings,
        ):
            response = client.post(
                "/execution/orders",
                json={
                    "decision_id": "dec-test-003",
                    "symbol": "BTCUSDT",
                    "side": "LONG",
                    "quantity": 1.0,
                    "price": 50000.0,
                },
            )
            assert response.status_code == 401

    def test_order_correct_key_accepted(self):
        """Korrekter Trade-Key → Request wird durchgereicht."""
        mock_settings = type(
            "MockSettings",
            (),
            {"read_api_key": "", "trade_api_key": "secret-trade-key"},
        )()
        with patch(
            "trading_harness.api.security.get_settings",
            return_value=mock_settings,
        ):
            response = client.post(
                "/execution/orders",
                json={
                    "decision_id": "dec-test-004",
                    "run_id": "run-1",
                    "symbol": "BTCUSDT",
                    "side": "LONG",
                    "quantity": 1.0,
                    "price": 50000.0,
                },
                headers={"X-Trade-API-Key": "secret-trade-key"},
            )
            # Execution disabled → REJECTED, NICHT 401
            assert response.status_code == 200
            assert response.json()["status"] == "REJECTED"


class TestKillSwitchToggleWithAuth:
    """POST /execution/kill-switch/{enabled} — Auth-Verhalten."""

    def test_kill_switch_toggle_wrong_key(self):
        """Falscher Key → 403."""
        mock_settings = type(
            "MockSettings",
            (),
            {"read_api_key": "", "trade_api_key": "secret-trade-key"},
        )()
        with patch(
            "trading_harness.api.security.get_settings",
            return_value=mock_settings,
        ):
            response = client.post(
                "/execution/kill-switch/True",
                headers={"X-Trade-API-Key": "wrong"},
            )
            assert response.status_code == 403

    def test_kill_switch_toggle_no_key(self):
        """Kein Key → 401."""
        mock_settings = type(
            "MockSettings",
            (),
            {"read_api_key": "", "trade_api_key": "secret-trade-key"},
        )()
        with patch(
            "trading_harness.api.security.get_settings",
            return_value=mock_settings,
        ):
            response = client.post("/execution/kill-switch/True")
            assert response.status_code == 401


class TestReadEndpointsWithAuth:
    """GET /execution/status, /execution/logs — Read-Key."""

    def test_status_with_read_key(self):
        """Read-Endpoint mit gültigem Read-Key → 200."""
        mock_settings = type(
            "MockSettings",
            (),
            {"read_api_key": "secret-read-key", "trade_api_key": ""},
        )()
        with patch(
            "trading_harness.api.security.get_settings",
            return_value=mock_settings,
        ):
            response = client.get(
                "/execution/status",
                headers={"X-Read-API-Key": "secret-read-key"},
            )
            assert response.status_code == 200

    def test_status_with_wrong_read_key(self):
        """Read-Endpoint mit falschem Read-Key → 403."""
        mock_settings = type(
            "MockSettings",
            (),
            {"read_api_key": "secret-read-key", "trade_api_key": ""},
        )()
        with patch(
            "trading_harness.api.security.get_settings",
            return_value=mock_settings,
        ):
            response = client.get(
                "/execution/status",
                headers={"X-Read-API-Key": "wrong"},
            )
            assert response.status_code == 403

    def test_logs_with_read_key(self):
        """Logs-Endpoint mit gültigem Read-Key → 200."""
        mock_settings = type(
            "MockSettings",
            (),
            {"read_api_key": "secret-read-key", "trade_api_key": ""},
        )()
        with patch(
            "trading_harness.api.security.get_settings",
            return_value=mock_settings,
        ):
            response = client.get(
                "/execution/logs",
                headers={"X-Read-API-Key": "secret-read-key"},
            )
            assert response.status_code == 200


class TestBackwardCompatibility:
    """Backward-Compatibility: Kein Key konfiguriert = alles geht."""

    def test_order_without_key_config(self):
        """Ohne Config-Key: Order ohne Key durchlassen."""
        mock_settings = type(
            "MockSettings",
            (),
            {"read_api_key": "", "trade_api_key": ""},
        )()
        with patch(
            "trading_harness.api.security.get_settings",
            return_value=mock_settings,
        ):
            response = client.post(
                "/execution/orders",
                json={
                    "decision_id": "dec-bc-001",
                    "symbol": "BTCUSDT",
                    "side": "LONG",
                    "quantity": 1.0,
                    "price": 50000.0,
                },
            )
            assert response.status_code == 200

    def test_status_without_key_config(self):
        """Ohne Config-Key: Status ohne Key durchlassen."""
        mock_settings = type(
            "MockSettings",
            (),
            {"read_api_key": "", "trade_api_key": ""},
        )()
        with patch(
            "trading_harness.api.security.get_settings",
            return_value=mock_settings,
        ):
            response = client.get("/execution/status")
            assert response.status_code == 200

    def test_kill_switch_toggle_without_key_config(self):
        """Ohne Config-Key: Kill Switch Toggle ohne Key durchlassen."""
        mock_settings = type(
            "MockSettings",
            (),
            {"read_api_key": "", "trade_api_key": ""},
        )()
        with patch(
            "trading_harness.api.security.get_settings",
            return_value=mock_settings,
        ):
            response = client.post("/execution/kill-switch/True")
            assert response.status_code == 200


class TestKillSwitchStateIsolation:
    """API-Tests dürfen den konfigurierten Kill-Switch-State-Pfad nicht verändern (WI-P5-11).

    Regression aus dem WI-P5-10-Review (MINOR-1): Der keyless
    Backward-Compat-Toggle hat ``enabled: true`` in das echte
    ``data/kill_switch.json`` geschrieben → ``make run`` nach
    ``make check`` startete mit einem durch die Test-Suite aktivierten
    Kill Switch.
    """

    def test_kill_switch_toggle_via_api_leaves_real_state_file_untouched(self):
        """Ein API-Toggle (auch keyless) darf das echte State-File nicht anfassen."""
        from trading_harness.api import routes

        mock_settings = type(
            "MockSettings",
            (),
            {"read_api_key": "", "trade_api_key": ""},
        )()
        real_path = Path(routes.settings.kill_switch_state_path)
        before = real_path.read_bytes() if real_path.exists() else None
        with patch(
            "trading_harness.api.security.get_settings",
            return_value=mock_settings,
        ):
            try:
                response = client.post("/execution/kill-switch/True")
                assert response.status_code == 200
            finally:
                # Best-effort-Cleanup ohne hartes Assert — ein
                # Cleanup-Fehler darf die Original-Exception des Tests
                # nicht maskieren (NIT aus den WI-P5-10-/WI-P5-11-Reviews).
                # Singleton nicht im aktiven Zustand für Folgetests
                # hinterlassen.
                response = client.post("/execution/kill-switch/False")
        after = real_path.read_bytes() if real_path.exists() else None
        assert after == before, (
            f"API-Test hat den echten Kill-Switch-State-Pfad verändert: {real_path}"
        )