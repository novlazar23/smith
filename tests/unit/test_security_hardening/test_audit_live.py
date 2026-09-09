"""Deterministic tests for the append-only hash-chained live audit trail."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

from packages.security.hardening.audit_live import (
    GENESIS_HASH,
    LiveAuditTrail,
    get_live_audit,
    reset_live_audit,
)


def _fixed_clock() -> datetime:
    return datetime(2025, 4, 1, 12, 0, 0, tzinfo=UTC)


class TestBasicRecording:
    def test_empty_trail_is_valid(self) -> None:
        assert LiveAuditTrail().verify_integrity() is True

    def test_record_appends_and_sequences(self) -> None:
        trail = LiveAuditTrail(clock=_fixed_clock)
        e1 = trail.record("a", actor="x", resource="r1")
        e2 = trail.record("b", actor="y", resource="r2")
        assert e1.sequence == 0
        assert e2.sequence == 1
        assert len(trail) == 2
        assert e1.previous_hash == GENESIS_HASH
        assert e2.previous_hash == e1.entry_hash
        assert e1.entry_hash != e2.entry_hash
        assert e1.timestamp == _fixed_clock()


class TestTamperDetection:
    def test_tampered_entry_detected(self) -> None:
        trail = LiveAuditTrail(clock=_fixed_clock)
        trail.record("a")
        e1 = trail.entries[0]
        # Replace the first entry with a forged one (same shape, different actor).
        trail._entries[0] = dataclasses.replace(e1, actor="hacker")
        assert trail.verify_integrity() is False

    def test_mutated_details_detected(self) -> None:
        trail = LiveAuditTrail(clock=_fixed_clock)
        trail.record("a", details={"note": "ok"})
        trail.entries[0].details["note"] = "tampered"
        assert trail.verify_integrity() is False

    def test_clean_trail_verifies(self) -> None:
        trail = LiveAuditTrail(clock=_fixed_clock)
        for i in range(5):
            trail.record(f"op{i}", details={"i": i})
        assert trail.verify_integrity() is True


class TestSensitiveRedaction:
    def test_secret_values_are_redacted(self) -> None:
        trail = LiveAuditTrail(clock=_fixed_clock)
        trail.record(
            "login",
            details={
                "api_key": "sk-abc123",
                "password": "hunter2",
                "token": "tok-xyz",
                "safe": "visible",
            },
        )
        entry = trail.entries[0]
        assert entry.details["api_key"] == "[REDACTED]"
        assert entry.details["password"] == "[REDACTED]"
        assert entry.details["token"] == "[REDACTED]"
        assert entry.details["safe"] == "visible"

    def test_non_string_sensitive_values_pass_through(self) -> None:
        trail = LiveAuditTrail(clock=_fixed_clock)
        trail.record("n", details={"api_key": 12345})
        assert trail.entries[0].details["api_key"] == 12345


class TestModuleAccessors:
    def test_reset_replaces_shared_trail(self) -> None:
        trail1 = get_live_audit()
        trail1.record("x")
        trail2 = reset_live_audit()
        assert trail1 is not trail2
        assert get_live_audit() is trail2
        assert len(trail2) == 0
        # Old trail keeps its entries (no shared mutation).
        assert len(trail1) == 1
