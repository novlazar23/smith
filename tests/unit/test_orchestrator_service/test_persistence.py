"""Tests für persist_shadow_decision() gegen eine Fake-Connection."""

from __future__ import annotations

import json

import pytest
from apps.orchestrator_service.service import ShadowDecision, persist_shadow_decision

from .conftest import FakeConnection


def _decision(
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
) -> ShadowDecision:
    return ShadowDecision(
        run_id="orch-20260101T120000Z-BTC/USDT",
        instrument="BTC/USDT",
        decision="NO_TRADE",
        confidence=0.42,
        reason="No active agents (all shadow)",
        first_round_count=3,
        second_round_count=3,
        latency_ms=12.5,
        errors=errors if errors is not None else [],
        warnings=warnings if warnings is not None else [],
    )


class TestPersistShadowDecision:
    """persist_shadow_decision() schreibt die erwarteten Parameter."""

    def test_executes_insert_and_commits(self) -> None:
        """Genau ein INSERT und ein Commit pro Entscheidung."""
        conn = FakeConnection()

        persist_shadow_decision(conn, _decision())

        assert len(conn.executed) == 1
        assert conn.commits == 1
        params = conn.executed[0][1]
        assert params["run_id"] == "orch-20260101T120000Z-BTC/USDT"
        assert params["instrument"] == "BTC/USDT"
        assert params["decision"] == "NO_TRADE"
        assert params["confidence"] == 0.42
        assert params["reason"] == "No active agents (all shadow)"
        assert params["first_round_count"] == 3
        assert params["second_round_count"] == 3
        assert params["latency_ms"] == 12.5
        assert params["probabilities"] is None
        assert params["base_close"] is None

    def test_joins_errors_and_warnings(self) -> None:
        """Fehler- und Warnlisten werden mit Newline verbunden persistiert."""
        conn = FakeConnection()

        persist_shadow_decision(
            conn, _decision(errors=["e1", "e2"], warnings=["w1", "w2", "w3"])
        )

        params = conn.executed[0][1]
        assert params["errors"] == "e1\ne2"
        assert params["warnings"] == "w1\nw2\nw3"

    def test_empty_lists_become_none(self) -> None:
        """Leere Fehler/Warnlisten werden als NULL persistiert."""
        conn = FakeConnection()

        persist_shadow_decision(conn, _decision())

        params = conn.executed[0][1]
        assert params["errors"] is None
        assert params["warnings"] is None

    def test_serializes_probabilities_and_base_close(self) -> None:
        """predicted_probabilities als JSON-String, base_close durchgereicht."""
        conn = FakeConnection()
        decision = ShadowDecision(
            run_id="orch-20260101T120000Z-BTC/USDT",
            instrument="BTC/USDT",
            decision="LONG",
            confidence=0.6,
            reason="test",
            first_round_count=4,
            second_round_count=4,
            latency_ms=10.0,
            predicted_probabilities={"up": 0.7, "down": 0.2, "range": 0.1},
            base_close=105.5,
        )

        persist_shadow_decision(conn, decision)

        params = conn.executed[0][1]
        assert json.loads(params["probabilities"]) == {"up": 0.7, "down": 0.2, "range": 0.1}
        assert params["base_close"] == 105.5

    def test_commit_failure_propagates(self) -> None:
        """Ein Commit-Fehler wird weitergeworfen (wird vom Zyklus abgefangen)."""
        conn = FakeConnection()
        conn.fail_commit = True

        with pytest.raises(RuntimeError, match="commit failed"):
            persist_shadow_decision(conn, _decision())

        assert len(conn.executed) == 1
        assert conn.commits == 0
