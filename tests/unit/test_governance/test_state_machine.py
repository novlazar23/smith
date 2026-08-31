"""Tests für Governance State Machine — EPIC-11."""

from __future__ import annotations

from datetime import UTC

import pytest
from packages.governance.state_machine import (
    STATE_WEIGHTS,
    VALID_TRANSITIONS,
    AgentRecord,
    AgentState,
    GovernanceStateMachine,
)


class TestAgentState:
    """Testet AgentState-Enum."""

    def test_all_states_present(self) -> None:
        assert AgentState.SHADOW == "SHADOW"
        assert AgentState.ACTIVE == "ACTIVE"
        assert AgentState.DEGRADED == "DEGRADED"
        assert AgentState.QUARANTINED == "QUARANTINED"
        assert AgentState.DISABLED == "DISABLED"

    def test_state_values_are_upper(self) -> None:
        for state in AgentState:
            assert state == state.value.upper()


class TestStateWeights:
    """Testet STATE_WEIGHTS-Konstante."""

    def test_active_has_full_weight(self) -> None:
        assert STATE_WEIGHTS[AgentState.ACTIVE] == 1.0

    def test_degraded_has_reduced_weight(self) -> None:
        assert STATE_WEIGHTS[AgentState.DEGRADED] == 0.5

    def test_shadow_has_no_weight(self) -> None:
        assert STATE_WEIGHTS[AgentState.SHADOW] == 0.0

    def test_quarantined_has_no_weight(self) -> None:
        assert STATE_WEIGHTS[AgentState.QUARANTINED] == 0.0

    def test_disabled_has_no_weight(self) -> None:
        assert STATE_WEIGHTS[AgentState.DISABLED] == 0.0

    def test_all_states_covered(self) -> None:
        for state in AgentState:
            assert state in STATE_WEIGHTS


class TestValidTransitions:
    """Testet VALID_TRANSITIONS-Matrix."""

    def test_shadow_can_go_to_active(self) -> None:
        assert AgentState.ACTIVE in VALID_TRANSITIONS[AgentState.SHADOW]

    def test_shadow_can_go_to_disabled(self) -> None:
        assert AgentState.DISABLED in VALID_TRANSITIONS[AgentState.SHADOW]

    def test_shadow_cannot_go_directly_to_quarantined(self) -> None:
        assert AgentState.QUARANTINED not in VALID_TRANSITIONS[AgentState.SHADOW]

    def test_active_can_go_to_degraded(self) -> None:
        assert AgentState.DEGRADED in VALID_TRANSITIONS[AgentState.ACTIVE]

    def test_active_can_go_to_quarantined(self) -> None:
        assert AgentState.QUARANTINED in VALID_TRANSITIONS[AgentState.ACTIVE]

    def test_active_can_go_to_disabled(self) -> None:
        assert AgentState.DISABLED in VALID_TRANSITIONS[AgentState.ACTIVE]

    def test_active_cannot_go_to_shadow_directly(self) -> None:
        assert AgentState.SHADOW not in VALID_TRANSITIONS[AgentState.ACTIVE]

    def test_degraded_can_recover_to_active(self) -> None:
        assert AgentState.ACTIVE in VALID_TRANSITIONS[AgentState.DEGRADED]

    def test_degraded_can_go_to_quarantined(self) -> None:
        assert AgentState.QUARANTINED in VALID_TRANSITIONS[AgentState.DEGRADED]

    def test_degraded_can_go_to_shadow(self) -> None:
        assert AgentState.SHADOW in VALID_TRANSITIONS[AgentState.DEGRADED]

    def test_quarantined_can_go_to_shadow(self) -> None:
        assert AgentState.SHADOW in VALID_TRANSITIONS[AgentState.QUARANTINED]

    def test_quarantined_can_go_to_disabled(self) -> None:
        assert AgentState.DISABLED in VALID_TRANSITIONS[AgentState.QUARANTINED]

    def test_disabled_can_reinitialize_to_shadow(self) -> None:
        assert AgentState.SHADOW in VALID_TRANSITIONS[AgentState.DISABLED]

    def test_disabled_is_sink_except_shadow(self) -> None:
        for state in AgentState:
            if state != AgentState.SHADOW:
                assert state not in VALID_TRANSITIONS[AgentState.DISABLED]


class TestAgentRecord:
    """Testet AgentRecord-Datenklasse."""

    def test_initial_state_is_shadow(self) -> None:
        record = AgentRecord(agent_id="test-agent")
        assert record.state == AgentState.SHADOW

    def test_initial_version(self) -> None:
        record = AgentRecord(agent_id="test-agent")
        assert record.version == "0.0.0"

    def test_shadow_weight_is_zero(self) -> None:
        record = AgentRecord(agent_id="test-agent")
        assert record.consensus_weight == 0.0

    def test_active_weight_is_one(self) -> None:
        record = AgentRecord(agent_id="test-agent", state=AgentState.ACTIVE)
        assert record.consensus_weight == 1.0

    def test_degraded_custom_weight(self) -> None:
        record = AgentRecord(
            agent_id="test-agent",
            state=AgentState.DEGRADED,
            degraded_weight=0.75,
        )
        assert record.consensus_weight == 0.75

    def test_degraded_default_weight(self) -> None:
        record = AgentRecord(agent_id="test-agent", state=AgentState.DEGRADED)
        assert record.consensus_weight == 0.5

    def test_history_starts_empty(self) -> None:
        record = AgentRecord(agent_id="test-agent")
        assert record.history == []

    def test_created_at_is_present(self) -> None:
        record = AgentRecord(agent_id="test-agent")
        assert record.created_at.tzinfo == UTC


class TestGovernanceStateMachine:
    """Testet GovernanceStateMachine."""

    @pytest.fixture
    def sm(self) -> GovernanceStateMachine:
        return GovernanceStateMachine()

    def test_register_creates_shadow_agent(self, sm: GovernanceStateMachine) -> None:
        record = sm.register("agent-1", version="1.0.0")
        assert record.agent_id == "agent-1"
        assert record.state == AgentState.SHADOW
        assert record.version == "1.0.0"

    def test_register_returns_agent_record(self, sm: GovernanceStateMachine) -> None:
        record = sm.register("agent-1")
        assert isinstance(record, AgentRecord)

    def test_register_logs_audit_entry(self, sm: GovernanceStateMachine) -> None:
        sm.register("agent-1")
        assert sm.audit.total_entries >= 1
        entry = sm.audit.entries[-1]
        assert entry.event_type == "state_transition"
        assert entry.agent_id == "agent-1"
        assert entry.new_state == AgentState.SHADOW

    def test_transition_shadow_to_active(self, sm: GovernanceStateMachine) -> None:
        sm.register("agent-1")
        transition = sm.transition("agent-1", AgentState.ACTIVE, reason="promotion", actor="reviewer-1")
        assert transition.from_state == AgentState.SHADOW
        assert transition.to_state == AgentState.ACTIVE
        assert transition.reason == "promotion"
        assert transition.actor == "reviewer-1"

    def test_transition_updates_agent_state(self, sm: GovernanceStateMachine) -> None:
        sm.register("agent-1")
        sm.transition("agent-1", AgentState.ACTIVE, reason="promotion")
        record = sm.get_agent("agent-1")
        assert record.state == AgentState.ACTIVE

    def test_transition_logs_to_audit(self, sm: GovernanceStateMachine) -> None:
        sm.register("agent-1")
        before = sm.audit.total_entries
        sm.transition("agent-1", AgentState.ACTIVE, reason="promotion")
        assert sm.audit.total_entries == before + 1

    def test_transition_cannot_go_active_to_shadow(self, sm: GovernanceStateMachine) -> None:
        sm.register("agent-1")
        sm.transition("agent-1", AgentState.ACTIVE, reason="promotion")
        with pytest.raises(ValueError, match="not allowed"):
            sm.transition("agent-1", AgentState.SHADOW, reason="invalid")

    def test_transition_cannot_go_shadow_to_quarantined(self, sm: GovernanceStateMachine) -> None:
        sm.register("agent-1")
        with pytest.raises(ValueError, match="not allowed"):
            sm.transition("agent-1", AgentState.QUARANTINED, reason="invalid")

    def test_transition_to_degraded_sets_weight(self, sm: GovernanceStateMachine) -> None:
        sm.register("agent-1")
        sm.transition("agent-1", AgentState.ACTIVE, reason="promotion")
        sm.transition(
            "agent-1",
            AgentState.DEGRADED,
            reason="performance_drop",
            metadata={"weight": 0.3},
        )
        record = sm.get_agent("agent-1")
        assert record.degraded_weight == 0.3
        assert record.consensus_weight == 0.3

    def test_transition_to_degraded_default_weight(self, sm: GovernanceStateMachine) -> None:
        sm.register("agent-1")
        sm.transition("agent-1", AgentState.ACTIVE, reason="promotion")
        sm.transition("agent-1", AgentState.DEGRADED, reason="performance_drop")
        record = sm.get_agent("agent-1")
        assert record.degraded_weight == 0.5

    def test_can_transition_returns_true(self, sm: GovernanceStateMachine) -> None:
        sm.register("agent-1")
        assert sm.can_transition("agent-1", AgentState.ACTIVE) is True

    def test_can_transition_returns_false_for_invalid(self, sm: GovernanceStateMachine) -> None:
        sm.register("agent-1")
        assert sm.can_transition("agent-1", AgentState.QUARANTINED) is False

    def test_can_transition_returns_false_for_unknown_agent(self, sm: GovernanceStateMachine) -> None:
        assert sm.can_transition("unknown", AgentState.ACTIVE) is False

    def test_get_agent_returns_record(self, sm: GovernanceStateMachine) -> None:
        sm.register("agent-1")
        record = sm.get_agent("agent-1")
        assert record is not None
        assert record.agent_id == "agent-1"

    def test_get_agent_returns_none_for_unknown(self, sm: GovernanceStateMachine) -> None:
        assert sm.get_agent("unknown") is None

    def test_get_all_agents(self, sm: GovernanceStateMachine) -> None:
        sm.register("agent-1")
        sm.register("agent-2")
        all_agents = sm.get_all_agents()
        assert len(all_agents) == 2
        assert "agent-1" in all_agents
        assert "agent-2" in all_agents

    def test_get_state_counts(self, sm: GovernanceStateMachine) -> None:
        sm.register("agent-1")
        sm.register("agent-2")
        sm.register("agent-3")
        sm.transition("agent-1", AgentState.ACTIVE, reason="promotion")
        sm.transition("agent-2", AgentState.ACTIVE, reason="promotion")
        counts = sm.get_state_counts()
        assert counts[AgentState.SHADOW] == 1
        assert counts[AgentState.ACTIVE] == 2
        for state in AgentState:
            assert state in counts

    def test_get_consensus_weights(self, sm: GovernanceStateMachine) -> None:
        sm.register("agent-1")
        sm.register("agent-2")
        sm.transition("agent-1", AgentState.ACTIVE, reason="promotion")
        weights = sm.get_consensus_weights(["agent-1", "agent-2"])
        assert weights["agent-1"] == 1.0
        assert weights["agent-2"] == 0.0

    def test_consensus_weights_skips_unknown(self, sm: GovernanceStateMachine) -> None:
        sm.register("agent-1")
        weights = sm.get_consensus_weights(["agent-1", "unknown"])
        assert "agent-1" in weights
        assert "unknown" not in weights

    def test_multiple_transitions(self, sm: GovernanceStateMachine) -> None:
        sm.register("agent-1")
        sm.transition("agent-1", AgentState.ACTIVE, reason="promotion")
        sm.transition("agent-1", AgentState.DEGRADED, reason="drift")
        sm.transition("agent-1", AgentState.QUARANTINED, reason="critical_failure")
        record = sm.get_agent("agent-1")
        assert record.state == AgentState.QUARANTINED
        assert len(record.history) == 3

    def test_full_lifecycle(self, sm: GovernanceStateMachine) -> None:
        """Ganzer Lebenszyklus: SHADOW → ACTIVE → DEGRADED → ACTIVE → QUARANTINED → SHADOW → DISABLED → SHADOW."""
        sm.register("agent-1")
        sm.transition("agent-1", AgentState.ACTIVE, reason="promotion")
        sm.transition("agent-1", AgentState.DEGRADED, reason="drift")
        sm.transition("agent-1", AgentState.ACTIVE, reason="recovery")
        sm.transition("agent-1", AgentState.QUARANTINED, reason="critical_failure")
        sm.transition("agent-1", AgentState.SHADOW, reason="review_reset")
        sm.transition("agent-1", AgentState.DISABLED, reason="permanent_removal")
        record = sm.get_agent("agent-1")
        assert record.state == AgentState.DISABLED
        assert len(record.history) == 6

    def test_audit_entry_contains_details(self, sm: GovernanceStateMachine) -> None:
        sm.register("agent-1", version="2.0.0")
        entry = sm.audit.entries[-1]
        assert entry.agent_id == "agent-1"
        assert entry.new_state == AgentState.SHADOW
        assert entry.previous_state == "N/A"
        assert entry.details.get("reason") == "initial_registration"
        assert entry.details.get("version") == "2.0.0"

    def test_disabled_to_shadow_rollback(self, sm: GovernanceStateMachine) -> None:
        sm.register("agent-1")
        sm.transition("agent-1", AgentState.ACTIVE, reason="promotion")
        sm.transition("agent-1", AgentState.QUARANTINED, reason="failure")
        sm.transition("agent-1", AgentState.DISABLED, reason="permanent")
        assert sm.get_agent("agent-1").state == AgentState.DISABLED
        sm.transition("agent-1", AgentState.SHADOW, reason="reinitialize")
        assert sm.get_agent("agent-1").state == AgentState.SHADOW

    def test_disabled_cannot_go_to_active_directly(self, sm: GovernanceStateMachine) -> None:
        sm.register("agent-1")
        sm.transition("agent-1", AgentState.ACTIVE, reason="promotion")
        sm.transition("agent-1", AgentState.QUARANTINED, reason="failure")
        sm.transition("agent-1", AgentState.DISABLED, reason="permanent")
        with pytest.raises(ValueError, match="not allowed"):
            sm.transition("agent-1", AgentState.ACTIVE, reason="bypass")

    def test_disabled_cannot_go_to_quarantined(self, sm: GovernanceStateMachine) -> None:
        sm.register("agent-1")
        sm.transition("agent-1", AgentState.DISABLED, reason="direct_disable")
        with pytest.raises(ValueError, match="not allowed"):
            sm.transition("agent-1", AgentState.QUARANTINED, reason="invalid")

    def test_disabled_cannot_go_to_degraded(self, sm: GovernanceStateMachine) -> None:
        sm.register("agent-1")
        sm.transition("agent-1", AgentState.DISABLED, reason="direct_disable")
        with pytest.raises(ValueError, match="not allowed"):
            sm.transition("agent-1", AgentState.DEGRADED, reason="invalid")

    def test_disable_then_reenable_lifecycle(self, sm: GovernanceStateMachine) -> None:
        sm.register("agent-1")
        sm.transition("agent-1", AgentState.ACTIVE, reason="promotion")
        sm.transition("agent-1", AgentState.QUARANTINED, reason="critical")
        sm.transition("agent-1", AgentState.DISABLED, reason="removal")
        sm.transition("agent-1", AgentState.SHADOW, reason="reinit")
        sm.transition("agent-1", AgentState.ACTIVE, reason="re-promotion")
        assert sm.get_agent("agent-1").state == AgentState.ACTIVE
        assert len(sm.get_agent("agent-1").history) == 5
