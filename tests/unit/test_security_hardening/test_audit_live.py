"""Deterministic tests for the append-only hash-chained live audit trail."""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from packages.security.hardening.audit_live import (
    AUDIT_LOG_PATH_ENV,
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

    def test_reset_live_audit_uses_env_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        audit_path = tmp_path / "nested" / "audit.jsonl"
        monkeypatch.setenv(AUDIT_LOG_PATH_ENV, str(audit_path))
        try:
            trail = reset_live_audit(clock=_fixed_clock)
            trail.record("persisted", actor="tester")
            payload = json.loads(audit_path.read_text(encoding="utf-8").strip())
            assert payload["action"] == "persisted"
            assert payload["actor"] == "tester"
            assert payload["entry_hash"] == trail.entries[0].entry_hash
        finally:
            monkeypatch.delenv(AUDIT_LOG_PATH_ENV, raising=False)
            reset_live_audit()


class TestPersistentAudit:
    def test_record_appends_jsonl(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.jsonl"
        trail = LiveAuditTrail(clock=_fixed_clock, path=audit_path)
        trail.record("op1", actor="a")
        trail.record("op2", actor="b")
        lines = audit_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        second = json.loads(lines[1])
        assert first["action"] == "op1"
        assert second["action"] == "op2"
        assert second["previous_hash"] == first["entry_hash"]

    def test_record_creates_missing_parent_dirs(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "a" / "b" / "audit.jsonl"
        trail = LiveAuditTrail(path=audit_path)
        trail.record("op")
        assert audit_path.exists()
