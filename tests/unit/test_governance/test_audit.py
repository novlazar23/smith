"""Tests für Audit Trail — EPIC-11."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from packages.governance.audit import AuditEntry, AuditTrail
from packages.governance.state_machine import AgentState


class TestAuditEntry:
    """Testet AuditEntry-Datenklasse."""

    def test_defaults(self) -> None:
        entry = AuditEntry(
            entry_id="AUDIT-000001",
            event_type="state_transition",
            agent_id="a1",
            actor="system",
        )
        assert entry.entry_id == "AUDIT-000001"
        assert entry.event_type == "state_transition"
        assert entry.agent_id == "a1"
        assert entry.actor == "system"
        assert entry.timestamp.tzinfo == UTC
        assert entry.details == {}
        assert entry.previous_state is None
        assert entry.new_state is None
        assert entry.approved is None
        assert entry.rollback_hash is None

    def test_custom_values(self) -> None:
        now = datetime.now(UTC)
        entry = AuditEntry(
            entry_id="AUDIT-000042",
            event_type="review",
            agent_id="a1",
            actor="reviewer-1",
            timestamp=now,
            details={"score": 0.85, "comment": "good"},
            previous_state=AgentState.SHADOW,
            new_state=AgentState.ACTIVE,
            approved=True,
            rollback_hash="abc123",
        )
        assert entry.actor == "reviewer-1"
        assert entry.details == {"score": 0.85, "comment": "good"}
        assert entry.previous_state == AgentState.SHADOW
        assert entry.new_state == AgentState.ACTIVE
        assert entry.approved is True
        assert entry.rollback_hash == "abc123"

    def test_to_dict(self) -> None:
        entry = AuditEntry(
            entry_id="AUDIT-000001",
            event_type="state_transition",
            agent_id="a1",
            actor="system",
            details={"reason": "test"},
        )
        d = entry.to_dict()
        assert d["entry_id"] == "AUDIT-000001"
        assert d["event_type"] == "state_transition"
        assert d["agent_id"] == "a1"
        assert d["actor"] == "system"
        assert d["details"] == {"reason": "test"}
        assert d["previous_state"] is None
        assert d["new_state"] is None
        assert d["approved"] is None
        assert "timestamp" in d


class TestAuditTrail:
    """Testet AuditTrail-Klasse."""

    @pytest.fixture
    def trail(self) -> AuditTrail:
        return AuditTrail()

    def test_empty_initially(self, trail: AuditTrail) -> None:
        assert trail.entries == []
        assert trail.total_entries == 0

    # --- log_state_transition ---

    def test_log_state_transition_creates_entry(self, trail: AuditTrail) -> None:
        entry = trail.log_state_transition("a1", None, AgentState.SHADOW)
        assert entry.entry_id == "AUDIT-000001"
        assert entry.event_type == "state_transition"
        assert entry.agent_id == "a1"
        assert entry.previous_state is None
        assert entry.new_state == AgentState.SHADOW

    def test_log_state_transition_incremental_ids(self, trail: AuditTrail) -> None:
        trail.log_state_transition("a1", None, AgentState.SHADOW)
        trail.log_state_transition("a1", AgentState.SHADOW, AgentState.ACTIVE)
        trail.log_state_transition("a2", None, AgentState.SHADOW)
        assert trail.entries[0].entry_id == "AUDIT-000001"
        assert trail.entries[1].entry_id == "AUDIT-000002"
        assert trail.entries[2].entry_id == "AUDIT-000003"

    def test_log_state_transition_with_details(self, trail: AuditTrail) -> None:
        entry = trail.log_state_transition("a1", AgentState.SHADOW, AgentState.ACTIVE, details={"version": "1.0"})
        assert entry.details == {"version": "1.0"}

    def test_log_state_transition_with_actor(self, trail: AuditTrail) -> None:
        entry = trail.log_state_transition("a1", AgentState.SHADOW, AgentState.ACTIVE, actor="reviewer-5")
        assert entry.actor == "reviewer-5"

    def test_log_state_transition_default_actor(self, trail: AuditTrail) -> None:
        entry = trail.log_state_transition("a1", None, AgentState.SHADOW)
        assert entry.actor == "system"

    def test_log_state_transition_updates_total(self, trail: AuditTrail) -> None:
        assert trail.total_entries == 0
        trail.log_state_transition("a1", None, AgentState.SHADOW)
        assert trail.total_entries == 1
        trail.log_state_transition("a2", None, AgentState.SHADOW)
        assert trail.total_entries == 2

    # --- log_decision ---

    def test_log_decision_creates_entry(self, trail: AuditTrail) -> None:
        entry = trail.log_decision("a1", "LONG_BIAS")
        assert entry.event_type == "decision"
        assert entry.agent_id == "a1"
        assert entry.new_state == "LONG_BIAS"

    def test_log_decision_with_details(self, trail: AuditTrail) -> None:
        entry = trail.log_decision("a1", "LONG_BIAS", details={"confidence": 0.80})
        assert entry.details == {"confidence": 0.80}

    def test_log_decision_default_actor(self, trail: AuditTrail) -> None:
        entry = trail.log_decision("a1", "LONG_BIAS")
        assert entry.actor == "system"

    def test_log_decision_with_custom_actor(self, trail: AuditTrail) -> None:
        entry = trail.log_decision("a1", "LONG_BIAS", actor="engine-1")
        assert entry.actor == "engine-1"

    # --- log_review ---

    def test_log_review_creates_entry(self, trail: AuditTrail) -> None:
        entry = trail.log_review("a1", "reviewer-1", True)
        assert entry.event_type == "review"
        assert entry.agent_id == "a1"
        assert entry.actor == "reviewer-1"
        assert entry.approved is True

    def test_log_review_rejected(self, trail: AuditTrail) -> None:
        entry = trail.log_review("a1", "reviewer-2", False)
        assert entry.approved is False

    def test_log_review_with_details(self, trail: AuditTrail) -> None:
        entry = trail.log_review("a1", "reviewer-1", True, details={"score": 0.90})
        assert entry.details == {"score": 0.90}

    def test_log_review_default_details_empty(self, trail: AuditTrail) -> None:
        entry = trail.log_review("a1", "reviewer-1", True)
        assert entry.details == {}

    # --- log_promotion ---

    def test_log_promotion_creates_state_transition(self, trail: AuditTrail) -> None:
        entry = trail.log_promotion("a1", AgentState.SHADOW, AgentState.ACTIVE)
        assert entry.event_type == "state_transition"
        assert entry.previous_state == AgentState.SHADOW
        assert entry.new_state == AgentState.ACTIVE
        assert entry.details["event"] == "promotion"

    def test_log_promotion_with_details(self, trail: AuditTrail) -> None:
        entry = trail.log_promotion("a1", AgentState.SHADOW, AgentState.ACTIVE, details={"oos_score": 0.75})
        assert entry.details["event"] == "promotion"
        assert entry.details["oos_score"] == 0.75

    # --- log_quarantine ---

    def test_log_quarantine_creates_state_transition(self, trail: AuditTrail) -> None:
        entry = trail.log_quarantine("a1", "drift_detected")
        assert entry.event_type == "state_transition"
        assert entry.previous_state == "ACTIVE"
        assert entry.new_state == "QUARANTINED"
        assert entry.details["event"] == "quarantine"
        assert entry.details["reason"] == "drift_detected"

    def test_log_quarantine_with_details(self, trail: AuditTrail) -> None:
        entry = trail.log_quarantine("a1", "drift_detected", details={"drift_score": 0.30})
        assert entry.details["drift_score"] == 0.30

    # --- get_agent_history ---

    def test_get_agent_history_returns_matching_entries(self, trail: AuditTrail) -> None:
        trail.log_state_transition("a1", None, AgentState.SHADOW)
        trail.log_state_transition("a1", AgentState.SHADOW, AgentState.ACTIVE)
        trail.log_state_transition("a2", None, AgentState.SHADOW)
        history = trail.get_agent_history("a1")
        assert len(history) == 2
        assert all(e.agent_id == "a1" for e in history)

    def test_get_agent_history_empty_for_unknown(self, trail: AuditTrail) -> None:
        trail.log_state_transition("a1", None, AgentState.SHADOW)
        history = trail.get_agent_history("unknown")
        assert history == []

    def test_get_agent_history_partial(self, trail: AuditTrail) -> None:
        trail.log_state_transition("a1", None, AgentState.SHADOW)
        trail.log_state_transition("a2", None, AgentState.SHADOW)
        trail.log_state_transition("a1", AgentState.SHADOW, AgentState.ACTIVE)
        history = trail.get_agent_history("a1")
        assert len(history) == 2

    # --- total_entries ---

    def test_total_entries_across_agents(self, trail: AuditTrail) -> None:
        trail.log_state_transition("a1", None, AgentState.SHADOW)
        trail.log_decision("a1", "LONG_BIAS")
        trail.log_review("a1", "reviewer-1", True)
        trail.log_state_transition("a2", None, AgentState.SHADOW)
        assert trail.total_entries == 4

    # --- mixed event types ---

    def test_mixed_event_types_preserved(self, trail: AuditTrail) -> None:
        trail.log_state_transition("a1", None, AgentState.SHADOW)
        trail.log_decision("a1", "SHORT_BIAS")
        trail.log_review("a1", "reviewer-1", True)
        trail.log_promotion("a1", AgentState.SHADOW, AgentState.ACTIVE)
        trail.log_quarantine("a1", "drift")
        assert trail.total_entries == 5
        types = [e.event_type for e in trail.entries]
        assert types == ["state_transition", "decision", "review", "state_transition", "state_transition"]

    def test_timestamps_are_utc(self, trail: AuditTrail) -> None:
        trail.log_state_transition("a1", None, AgentState.SHADOW)
        assert trail.entries[0].timestamp.tzinfo == UTC

    def test_entries_ordered_by_creation(self, trail: AuditTrail) -> None:
        trail.log_state_transition("a1", None, AgentState.SHADOW)
        trail.log_decision("a1", "LONG_BIAS")
        trail.log_review("a1", "reviewer-1", True)
        assert trail.entries[0].entry_id == "AUDIT-000001"
        assert trail.entries[1].entry_id == "AUDIT-000002"
        assert trail.entries[2].entry_id == "AUDIT-000003"
