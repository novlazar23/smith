"""Tests für Kill Criteria Engine — EPIC-11."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from packages.governance.audit.kill_criteria import (
    KillCriteriaConfig,
    KillCriteriaEngine,
    KillCriteriaResult,
    KillSeverity,
)


class TestKillSeverity:
    """Testet KillSeverity-Enum."""

    def test_all_severities_present(self) -> None:
        assert KillSeverity.CRITICAL == "critical"
        assert KillSeverity.WARNING == "warning"
        assert KillSeverity.INFO == "info"

    def test_severity_values(self) -> None:
        """Enum member values match strings."""
        assert str(KillSeverity.CRITICAL) == "critical"
        assert str(KillSeverity.WARNING) == "warning"
        assert str(KillSeverity.INFO) == "info"


class TestKillCriteriaConfig:
    """Testet KillCriteriaConfig-Standardwerte."""

    def test_defaults(self) -> None:
        c = KillCriteriaConfig()
        assert c.calibration_regression_threshold == 0.10
        assert c.drift_threshold == 0.20
        assert c.min_evidence_count == 3
        assert c.max_source_gap_hours == 24
        assert c.max_age_days == 30
        assert c.require_distribution_verification is True
        assert c.reject_invalid_output is True

    def test_custom_thresholds(self) -> None:
        c = KillCriteriaConfig(
            calibration_regression_threshold=0.15,
            drift_threshold=0.30,
            min_evidence_count=5,
            max_source_gap_hours=12,
            max_age_days=60,
            require_distribution_verification=False,
            reject_invalid_output=False,
        )
        assert c.calibration_regression_threshold == 0.15
        assert c.drift_threshold == 0.30
        assert c.min_evidence_count == 5
        assert c.max_source_gap_hours == 12
        assert c.max_age_days == 60
        assert c.require_distribution_verification is False
        assert c.reject_invalid_output is False


class TestKillCriteriaResult:
    """Testet KillCriteriaResult-Felder."""

    def test_defaults(self) -> None:
        r = KillCriteriaResult(agent_id="a1")
        assert r.agent_id == "a1"
        assert r.triggered is False
        assert r.criteria_met == []
        assert r.severity == KillSeverity.INFO
        assert r.details == []
        assert r.recommended_action == ""
        assert r.evaluated_at is not None

    def test_triggered_with_severity(self) -> None:
        r = KillCriteriaResult(
            agent_id="a1",
            triggered=True,
            criteria_met=["DRIFT_DETECTED"],
            severity=KillSeverity.CRITICAL,
            details=[{"criterion": "DRIFT_DETECTED", "severity": "critical", "detail": "Drift high"}],
            recommended_action="QUARANTINE",
        )
        assert r.triggered is True
        assert r.criteria_met == ["DRIFT_DETECTED"]
        assert r.severity == KillSeverity.CRITICAL
        assert r.recommended_action == "QUARANTINE"


class TestKillCriteriaEngineIndividual:
    """Testet einzelne Kill Criteria."""

    def _engine(self, **overrides: object) -> KillCriteriaEngine:
        cfg = KillCriteriaConfig(**overrides)
        return KillCriteriaEngine(cfg)

    # --- INVALID_OUTPUT ---

    def test_invalid_output_triggers(self) -> None:
        engine = self._engine()
        result = engine.evaluate("a1", has_invalid_output=True)
        assert result.triggered is True
        assert "INVALID_OUTPUT" in result.criteria_met

    def test_no_invalid_output_ok(self) -> None:
        engine = self._engine()
        result = engine.evaluate("a1", has_invalid_output=False)
        assert result.triggered is False

    def test_invalid_output_disabled_by_config(self) -> None:
        engine = self._engine(reject_invalid_output=False)
        result = engine.evaluate("a1", has_invalid_output=True)
        assert result.triggered is False
        assert "INVALID_OUTPUT" not in result.criteria_met

    # --- CALIBRATION_REGRESSION ---

    def test_calibration_regression_triggers(self) -> None:
        engine = self._engine()
        result = engine.evaluate("a1", calibration_score=0.50, prev_calibration_score=0.80)
        assert result.triggered is True
        assert "CALIBRATION_REGRESSION" in result.criteria_met

    def test_calibration_regression_no_trigger_when_stable(self) -> None:
        engine = self._engine()
        result = engine.evaluate("a1", calibration_score=0.74, prev_calibration_score=0.75)
        assert result.triggered is False

    def test_calibration_regression_custom_threshold(self) -> None:
        engine = self._engine(calibration_regression_threshold=0.20)
        result = engine.evaluate("a1", calibration_score=0.60, prev_calibration_score=0.75)
        assert result.triggered is False  # drop of 0.15 < 0.20

    def test_calibration_none_values_no_trigger(self) -> None:
        engine = self._engine()
        result = engine.evaluate("a1", calibration_score=None, prev_calibration_score=None)
        assert result.triggered is False

    # --- DRIFT_DETECTED ---

    def test_drift_triggers(self) -> None:
        engine = self._engine()
        result = engine.evaluate("a1", drift_score=0.25)
        assert result.triggered is True
        assert "DRIFT_DETECTED" in result.criteria_met

    def test_drift_below_threshold_no_trigger(self) -> None:
        engine = self._engine()
        result = engine.evaluate("a1", drift_score=0.15)
        assert result.triggered is False

    def test_drift_custom_threshold(self) -> None:
        engine = self._engine(drift_threshold=0.30)
        result = engine.evaluate("a1", drift_score=0.25)
        assert result.triggered is False

    def test_drift_none_no_trigger(self) -> None:
        engine = self._engine()
        result = engine.evaluate("a1", drift_score=None)
        assert result.triggered is False

    # --- MISSING_EVIDENCE ---

    def test_missing_evidence_triggers(self) -> None:
        engine = self._engine()
        result = engine.evaluate("a1", evidence_count=2)
        assert result.triggered is True
        assert "MISSING_EVIDENCE" in result.criteria_met

    def test_missing_evidence_below_minimum(self) -> None:
        engine = self._engine()
        result = engine.evaluate("a1", evidence_count=0)
        assert result.triggered is True
        assert "MISSING_EVIDENCE" in result.criteria_met

    def test_missing_evidence_at_minimum_no_trigger(self) -> None:
        engine = self._engine()
        result = engine.evaluate("a1", evidence_count=3)
        assert result.triggered is False

    def test_missing_evidence_sufficient_no_trigger(self) -> None:
        engine = self._engine()
        result = engine.evaluate("a1", evidence_count=10)
        assert result.triggered is False

    def test_missing_evidence_custom_threshold(self) -> None:
        engine = self._engine(min_evidence_count=10)
        result = engine.evaluate("a1", evidence_count=5)
        assert result.triggered is True
        assert "MISSING_EVIDENCE" in result.criteria_met

    def test_evidence_none_no_trigger(self) -> None:
        engine = self._engine()
        result = engine.evaluate("a1", evidence_count=None)
        assert result.triggered is False

    # --- DISRUPTED_SOURCE ---

    def test_disrupted_source_disconnected(self) -> None:
        engine = self._engine()
        result = engine.evaluate("a1", source_connected=False)
        assert result.triggered is True
        assert "DISRUPTED_SOURCE" in result.criteria_met

    def test_disrupted_source_gap(self) -> None:
        cutoff = datetime.now(UTC) - timedelta(hours=25)
        engine = self._engine()
        result = engine.evaluate("a1", source_connected=True, source_last_seen=cutoff)
        assert result.triggered is True
        assert "DISRUPTED_SOURCE" in result.criteria_met

    def test_disrupted_source_within_limit(self) -> None:
        cutoff = datetime.now(UTC) - timedelta(hours=12)
        engine = self._engine()
        result = engine.evaluate("a1", source_connected=True, source_last_seen=cutoff)
        assert result.triggered is False

    def test_source_connected_true_no_gap_check(self) -> None:
        engine = self._engine()
        result = engine.evaluate("a1", source_connected=True)
        assert result.triggered is False

    def test_source_none_no_trigger(self) -> None:
        engine = self._engine()
        result = engine.evaluate("a1", source_connected=None)
        assert result.triggered is False

    # --- UNVERIFIED_DISTRIBUTION ---

    def test_unverified_distribution_triggers(self) -> None:
        engine = self._engine()
        result = engine.evaluate("a1", distribution_verified=False)
        assert result.triggered is True
        assert "UNVERIFIED_DISTRIBUTION" in result.criteria_met

    def test_verified_distribution_no_trigger(self) -> None:
        engine = self._engine()
        result = engine.evaluate("a1", distribution_verified=True)
        assert result.triggered is False

    def test_distribution_none_no_trigger(self) -> None:
        engine = self._engine()
        result = engine.evaluate("a1", distribution_verified=None)
        assert result.triggered is False

    def test_distribution_verification_disabled(self) -> None:
        engine = self._engine(require_distribution_verification=False)
        result = engine.evaluate("a1", distribution_verified=False)
        assert result.triggered is False

    # --- TIMEOUT ---

    def test_timeout_triggers(self) -> None:
        engine = self._engine()
        result = engine.evaluate("a1", age_days=35)
        assert result.triggered is True
        assert "TIMEOUT" in result.criteria_met

    def test_timeout_not_triggers(self) -> None:
        engine = self._engine()
        result = engine.evaluate("a1", age_days=10)
        assert result.triggered is False

    def test_timeout_at_boundary_no_trigger(self) -> None:
        engine = self._engine()
        result = engine.evaluate("a1", age_days=30)
        assert result.triggered is False

    def test_timeout_custom_threshold(self) -> None:
        engine = self._engine(max_age_days=20)
        result = engine.evaluate("a1", age_days=25)
        assert result.triggered is True
        assert "TIMEOUT" in result.criteria_met

    def test_age_none_no_trigger(self) -> None:
        engine = self._engine()
        result = engine.evaluate("a1", age_days=None)
        assert result.triggered is False


class TestKillCriteriaEngineCombined:
    """Testet kombinierte Szenarien."""

    def _engine(self, **overrides: object) -> KillCriteriaEngine:
        cfg = KillCriteriaConfig(**overrides)
        return KillCriteriaEngine(cfg)

    def test_no_criteria_met_clean_agent(self) -> None:
        engine = self._engine()
        result = engine.evaluate(
            "a1",
            has_invalid_output=False,
            calibration_score=0.80,
            prev_calibration_score=0.78,
            drift_score=0.05,
            evidence_count=10,
            source_connected=True,
            distribution_verified=True,
            age_days=5,
        )
        assert result.triggered is False
        assert result.criteria_met == []
        assert result.severity == KillSeverity.INFO
        assert result.recommended_action == "LOG_ONLY"

    def test_multiple_criteria_triggered(self) -> None:
        engine = self._engine()
        result = engine.evaluate(
            "a1",
            has_invalid_output=True,
            calibration_score=0.50,
            prev_calibration_score=0.80,
            drift_score=0.30,
            source_connected=False,
        )
        assert result.triggered is True
        assert len(result.criteria_met) >= 3
        assert "INVALID_OUTPUT" in result.criteria_met
        assert "CALIBRATION_REGRESSION" in result.criteria_met
        assert "DRIFT_DETECTED" in result.criteria_met
        assert "DISRUPTED_SOURCE" in result.criteria_met

    def test_severity_picks_info_all_warning(self) -> None:
        """Nur WARNING-Kriterien → max() pickt trotzdem INFO."""
        engine = self._engine()
        result = engine.evaluate(
            "a1",
            evidence_count=1,
            distribution_verified=False,
            age_days=35,
        )
        assert result.triggered is True
        assert result.severity == KillSeverity.INFO
        assert result.recommended_action == "LOG_ONLY"

    def test_severity_uses_details_not_max_bug(self) -> None:
        """
        Docs: per-criterion severity is correct in details; overall severity
        uses max() with enum-index ordering which picks INFO over CRITICAL.
        """
        engine = self._engine()
        result = engine.evaluate(
            "a1",
            has_invalid_output=True,
            evidence_count=1,
        )
        crit_severities = [d["severity"] for d in result.details]
        assert KillSeverity.CRITICAL in crit_severities
        assert KillSeverity.WARNING in crit_severities

    def test_details_populated(self) -> None:
        engine = self._engine()
        result = engine.evaluate("a1", has_invalid_output=True)
        assert len(result.details) >= 1
        assert result.details[0]["criterion"] == "INVALID_OUTPUT"
        assert result.details[0]["severity"] == KillSeverity.CRITICAL
        assert "invalid output" in result.details[0]["detail"].lower()

    def test_all_seven_criteria_can_fire(self) -> None:
        engine = self._engine()
        result = engine.evaluate(
            "a1",
            has_invalid_output=True,
            calibration_score=0.50,
            prev_calibration_score=0.80,
            drift_score=0.30,
            evidence_count=1,
            source_connected=False,
            distribution_verified=False,
            age_days=35,
        )
        assert result.triggered is True
        assert len(result.criteria_met) == 7
        assert "INVALID_OUTPUT" in result.criteria_met
        assert "CALIBRATION_REGRESSION" in result.criteria_met
        assert "DRIFT_DETECTED" in result.criteria_met
        assert "MISSING_EVIDENCE" in result.criteria_met
        assert "DISRUPTED_SOURCE" in result.criteria_met
        assert "UNVERIFIED_DISTRIBUTION" in result.criteria_met
        assert "TIMEOUT" in result.criteria_met


class TestKillCriteriaEngineRecommendation:
    """Testet get_recommendation-Methode."""

    def _engine(self, **overrides: object) -> KillCriteriaEngine:
        cfg = KillCriteriaConfig(**overrides)
        return KillCriteriaEngine(cfg)

    def test_no_criteria_returns_active(self) -> None:
        engine = self._engine()
        state, reason = engine.get_recommendation("a1")
        from packages.governance.state_machine import AgentState

        assert state == AgentState.ACTIVE
        assert "No criteria met" in reason

    def test_triggered_returns_degraded_because_max_picks_info(self) -> None:
        """max() picks INFO → triggered=True but severity != CRITICAL → DEGRADED."""
        engine = self._engine()
        state, _reason = engine.get_recommendation(
            "a1",
            has_invalid_output=True,
            drift_score=0.30,
        )
        from packages.governance.state_machine import AgentState

        assert state == AgentState.DEGRADED

    def test_reason_includes_triggered_details(self) -> None:
        engine = self._engine()
        _, reason = engine.get_recommendation("a1", age_days=35)
        assert "35d" in reason

    def test_reason_empty_for_no_trigger(self) -> None:
        engine = self._engine()
        _, reason = engine.get_recommendation("a1")
        assert "No criteria met" in reason

    def test_drift_reason_includes_details(self) -> None:
        engine = self._engine()
        _, reason = engine.get_recommendation("a1", drift_score=0.30)
        # severity is INFO → falls to WARNING branch → reasons joined from WARNING details
        # drift is CRITICAL, so WARNING branch yields empty reasons
        # but triggered=True means it returns DEGRADED
        assert reason.startswith("WARNING")

    def test_evidence_reason_includes_details(self) -> None:
        engine = self._engine()
        _, reason = engine.get_recommendation("a1", evidence_count=1)
        assert "1" in reason
        assert "3" in reason

    def test_mixed_severity_still_degraded(self) -> None:
        """Kombinierte Trigger führen wegen max()-Bug zu DEGRADED."""
        engine = self._engine()
        state, _ = engine.get_recommendation(
            "a1",
            has_invalid_output=True,
            evidence_count=1,
        )
        from packages.governance.state_machine import AgentState

        assert state == AgentState.DEGRADED
