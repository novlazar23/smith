"""Tests für Quarantine Engine — EPIC-11."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from packages.governance.quarantine import (
    QuarantineConfig,
    QuarantineEngine,
    QuarantineEvent,
    QuarantineReason,
)


class TestQuarantineEvent:
    """Testet QuarantineEvent-Felder und -Werte."""

    def test_defaults(self) -> None:
        ev = QuarantineEvent(
            agent_id="a1",
            reason=QuarantineReason.INVALID_OUTPUT,
            severity="warning",
            details="test",
        )
        assert ev.agent_id == "a1"
        assert ev.resolved is False
        assert ev.resolved_at is None
        assert ev.timestamp is not None

    def test_severity_values(self) -> None:
        ev = QuarantineEvent(
            agent_id="a1",
            reason=QuarantineReason.DRIFT_DETECTED,
            severity="critical",
            details="drift",
        )
        assert ev.severity in ("warning", "critical")


class TestQuarantineConfig:
    """Testet QuarantineConfig-Standardwerte."""

    def test_defaults(self) -> None:
        c = QuarantineConfig()
        assert c.calibration_regression_threshold == 0.10
        assert c.drift_threshold == 0.20
        assert c.min_evidence_count == 3
        assert c.max_source_gap_hours == 24
        assert c.max_age_days == 30
        assert c.distribution_confidence == 0.90


class TestQuarantineEngineChecks:
    """Testet einzelne Quarantäne-Kriterien."""

    def _engine(self, **overrides) -> QuarantineEngine:
        cfg = QuarantineConfig(**overrides)
        return QuarantineEngine(cfg)

    # --- CALIBRATION_REGRESSION ---

    def test_no_calibration_regression_when_scores_stable(self) -> None:
        engine = self._engine()
        needed, events = engine.check("a1", calibration_score=0.75, prev_calibration_score=0.74)
        assert needed is False
        assert events == []

    def test_calibration_regression_triggers(self) -> None:
        engine = self._engine()
        needed, events = engine.check("a1", calibration_score=0.50, prev_calibration_score=0.75)
        assert needed is True
        assert any(e.reason == QuarantineReason.CALIBRATION_REGRESSION for e in events)

    def test_calibration_regression_threshold_custom(self) -> None:
        engine = self._engine(calibration_regression_threshold=0.30)
        needed, events = engine.check("a1", calibration_score=0.55, prev_calibration_score=0.75)
        assert needed is False

    # --- DRIFT_DETECTED ---

    def test_no_drift_when_below_threshold(self) -> None:
        engine = self._engine()
        needed, events = engine.check("a1", drift_score=0.15)
        assert needed is False

    def test_drift_triggers_when_above_threshold(self) -> None:
        engine = self._engine()
        needed, events = engine.check("a1", drift_score=0.25)
        assert needed is True
        assert any(e.reason == QuarantineReason.DRIFT_DETECTED for e in events)

    def test_drift_custom_threshold(self) -> None:
        engine = self._engine(drift_threshold=0.30)
        needed, events = engine.check("a1", drift_score=0.25)
        assert needed is False

    # --- MISSING_EVIDENCE ---

    def test_no_missing_evidence_when_sufficient(self) -> None:
        engine = self._engine()
        needed, events = engine.check("a1", evidence_count=5)
        assert needed is False

    def test_missing_evidence_triggers(self) -> None:
        engine = self._engine()
        needed, events = engine.check("a1", evidence_count=1)
        assert needed is False  # warning does not trigger quarantine
        assert any(e.reason == QuarantineReason.MISSING_EVIDENCE for e in events)
        assert any(e.severity == "warning" for e in events)

    def test_missing_evidence_custom_threshold(self) -> None:
        engine = self._engine(min_evidence_count=10)
        needed, events = engine.check("a1", evidence_count=5)
        assert any(e.reason == QuarantineReason.MISSING_EVIDENCE for e in events)

    # --- DISRUPTED_SOURCE ---

    def test_disrupted_source_disconnected(self) -> None:
        engine = self._engine()
        needed, events = engine.check("a1", source_connected=False)
        assert needed is True
        assert any(e.reason == QuarantineReason.DISRUPTED_SOURCE for e in events)

    def test_disrupted_source_gap(self) -> None:
        # Gap events are "warning" severity → quarantine_needed=False
        cutoff = datetime.now(UTC) - timedelta(hours=25)
        engine = self._engine()
        needed, events = engine.check("a1", source_connected=True, source_last_seen=cutoff)
        assert needed is False  # warning only, not critical
        assert any(e.reason == QuarantineReason.DISRUPTED_SOURCE for e in events)

    def test_source_gap_within_limit(self) -> None:
        cutoff = datetime.now(UTC) - timedelta(hours=12)
        engine = self._engine()
        needed, events = engine.check("a1", source_connected=True, source_last_seen=cutoff)
        assert needed is False
        assert events == []

    def test_source_connected_true_no_gap_check(self) -> None:
        engine = self._engine()
        needed, events = engine.check("a1", source_connected=True)
        assert needed is False

    # --- UNVERIFIED_DISTRIBUTION ---

    def test_unverified_distribution_triggers(self) -> None:
        engine = self._engine()
        needed, events = engine.check("a1", distribution_verified=False)
        assert needed is False  # warning severity
        assert any(e.reason == QuarantineReason.UNVERIFIED_DISTRIBUTION for e in events)

    def test_verified_distribution_no_trigger(self) -> None:
        engine = self._engine()
        needed, events = engine.check("a1", distribution_verified=True)
        assert needed is False

    def test_distribution_not_specified_no_trigger(self) -> None:
        engine = self._engine()
        needed, events = engine.check("a1")
        assert needed is False

    # --- TIMEOUT ---

    def test_timeout_triggers(self) -> None:
        engine = self._engine()
        needed, events = engine.check("a1", age_days=35)
        assert needed is False  # warning severity
        assert any(e.reason == QuarantineReason.TIMEOUT for e in events)

    def test_timeout_not_triggers(self) -> None:
        engine = self._engine()
        needed, events = engine.check("a1", age_days=10)
        assert needed is False

    def test_timeout_custom_threshold(self) -> None:
        engine = self._engine(max_age_days=20)
        needed, events = engine.check("a1", age_days=25)
        assert any(e.reason == QuarantineReason.TIMEOUT for e in events)

    # --- NO CHECKS SPECIFIED ---

    def test_no_checks_returns_clean(self) -> None:
        engine = self._engine()
        needed, events = engine.check("a1")
        assert needed is False
        assert events == []


class TestQuarantineEngineCombined:
    """Testet kombinierte Szenarien und Metriken."""

    def _engine(self, **overrides) -> QuarantineEngine:
        cfg = QuarantineConfig(**overrides)
        return QuarantineEngine(cfg)

    def test_no_quarantine_clean_agent(self) -> None:
        engine = self._engine()
        needed, events = engine.check(
            "a1",
            calibration_score=0.80,
            prev_calibration_score=0.78,
            drift_score=0.05,
            evidence_count=10,
            source_connected=True,
            distribution_verified=True,
        )
        assert needed is False
        assert events == []

    def test_multiple_events(self) -> None:
        engine = self._engine()
        cutoff = datetime.now(UTC) - timedelta(hours=25)
        needed, events = engine.check(
            "a1",
            calibration_score=0.40,
            prev_calibration_score=0.75,
            drift_score=0.30,
            source_connected=False,
            source_last_seen=cutoff,
            evidence_count=1,
        )
        assert needed is True
        assert len(events) >= 3

    def test_warning_only_no_quarantine(self) -> None:
        engine = self._engine()
        needed, events = engine.check(
            "a1",
            evidence_count=1,
            distribution_verified=False,
            age_days=35,
        )
        assert needed is False
        assert len(events) >= 3

    def test_active_quarantine_count(self) -> None:
        engine = self._engine()
        # Events direkt hinzufügen (check() appendet nicht nach self.events)
        engine.events.append(
            QuarantineEvent(
                agent_id="a1",
                reason=QuarantineReason.CALIBRATION_REGRESSION,
                severity="critical",
                details="calib drop",
            )
        )
        engine.events.append(
            QuarantineEvent(
                agent_id="a2",
                reason=QuarantineReason.DRIFT_DETECTED,
                severity="critical",
                details="drift",
            )
        )
        # Resolve eine Event
        engine.events[0].resolved = True
        engine.events[0].resolved_at = datetime.now(UTC)
        assert engine.active_quarantine_count == 1

    def test_event_has_details(self) -> None:
        engine = self._engine()
        needed, events = engine.check("a1", source_connected=False)
        assert events[0].details == "Source disconnected"

    def test_event_has_correct_agent_id(self) -> None:
        engine = self._engine()
        needed, events = engine.check("my-agent-123", source_connected=False)
        assert events[0].agent_id == "my-agent-123"


class TestQuarantineEventResolved:
    """Testet resolved-Status."""

    def _engine(self, **overrides) -> QuarantineEngine:
        cfg = QuarantineConfig(**overrides)
        return QuarantineEngine(cfg)

    def test_event_not_resolved_initially(self) -> None:
        engine = self._engine()
        needed, events = engine.check("a1", source_connected=False)
        ev = events[0]
        assert ev.resolved is False
        assert ev.resolved_at is None

    def test_resolve_event(self) -> None:
        engine = self._engine()
        needed, events = engine.check("a1", source_connected=False)
        ev = events[0]
        ev.resolved = True
        ev.resolved_at = datetime.now(UTC)
        assert ev.resolved is True
        assert ev.resolved_at is not None