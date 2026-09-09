"""Tests für packages/shadow — Shadow-Mode Engine, Metrics, Comparator (EPIC-16 WIP)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from packages.consensus import ConsensusDecision, ConsensusResult, VoteDirection
from packages.governance.audit import AuditTrail
from packages.orchestrator.pipeline import OrchestratorPipelineResult
from packages.shadow import (
    ShadowBrierScore,
    ShadowCalibration,
    ShadowEngine,
    ShadowMetrics,
    ShadowPaperComparison,
    ShadowPnL,
    ShadowPnLRecord,
)


def _make_consensus(
    decision: ConsensusDecision = ConsensusDecision.NO_TRADE,
    confidence: float = 0.2,
) -> ConsensusResult:
    return ConsensusResult(
        decision=decision,
        vote_distribution={VoteDirection.ABSTAIN: 1.0},
        agent_weights={},
        agent_agreements=[],
        agent_disagreements=[],
        confidence=confidence,
        reason="test",
    )


def _make_pipeline_result(
    decision: str = "NO_TRADE",
    consensus: ConsensusResult | None = None,
) -> OrchestratorPipelineResult:
    return OrchestratorPipelineResult(
        decision=decision,
        consensus=consensus,
        first_round_reports=[],
        seal_records=[],
        second_round_reports=[],
    )


class _FakePipeline:
    """Pipeline-Stub: liefert ein vorgegebenes Ergebnis, führt nichts aus."""

    def __init__(self, result: OrchestratorPipelineResult) -> None:
        self._result = result
        self.calls: list[dict[str, Any]] = []

    def run(self, **kwargs: Any) -> OrchestratorPipelineResult:
        self.calls.append(kwargs)
        return self._result


class TestShadowBrierScore:
    """Testet den Multi-Klassen-Brier-Score."""

    def test_perfect_prediction_scores_zero(self) -> None:
        bs = ShadowBrierScore()
        assert bs.add(1.0, 0.0, 0.0, "up") == pytest.approx(0.0)

    def test_wrong_prediction_scores_positive(self) -> None:
        bs = ShadowBrierScore()
        # (0-1)^2 + (0-0)^2 + (1-0)^2 = 2.0
        assert bs.add(0.0, 0.0, 1.0, "up") == pytest.approx(2.0)

    def test_mean_score_and_count(self) -> None:
        bs = ShadowBrierScore()
        bs.add(1.0, 0.0, 0.0, "up")  # 0.0
        bs.add(0.5, 0.5, 0.0, "up")  # 0.25 + 0.25 = 0.5
        assert bs.count == 2
        assert bs.mean_score == pytest.approx(0.25)
        assert bs.is_valid is True

    def test_empty_state(self) -> None:
        bs = ShadowBrierScore()
        assert bs.count == 0
        assert bs.mean_score == 0.0
        assert bs.is_valid is False


class TestShadowCalibration:
    """Testet das Kalibrierungs-Binning."""

    def test_add_assigns_bin_and_aggregates(self) -> None:
        cal = ShadowCalibration()
        cal.add(0.75, correct=True, direction="up")
        cal.add(0.72, correct=False, direction="up")
        data = cal.calibration_data
        assert data["0.7-0.8"] == {
            "count": 2,
            "avg_confidence": pytest.approx(0.735),
            "accuracy": pytest.approx(0.5),
        }
        assert cal.total_observations == 2
        assert cal.overall_accuracy == pytest.approx(0.5)

    def test_confidence_one_falls_into_last_bin(self) -> None:
        cal = ShadowCalibration()
        cal.add(1.0, correct=True, direction="up")
        assert cal.calibration_data["0.9-1.0"]["count"] == 1

    def test_reliability_diagram_excludes_empty_bins(self) -> None:
        cal = ShadowCalibration()
        cal.add(0.25, correct=True, direction="down")
        diagram = cal.reliability_diagram_data
        assert len(diagram) == 1
        assert diagram[0]["avg_confidence"] == pytest.approx(0.25)
        assert diagram[0]["accuracy"] == pytest.approx(1.0)

    def test_empty_calibration(self) -> None:
        cal = ShadowCalibration()
        assert cal.total_observations == 0
        assert cal.overall_accuracy == 0.0
        assert cal.reliability_diagram_data == []
        # Alle 10 Bins existieren, alle leer
        assert all(v["count"] == 0 for v in cal.calibration_data.values())
        assert len(cal.calibration_data) == 10


class TestShadowPnL:
    """Testet das virtuelle PnL-Tracking."""

    def test_virtual_pnl_by_direction(self) -> None:
        assert (
            ShadowPnLRecord("r1", "LONG", 0.05, 0.03).virtual_pnl == 0.03
        )
        assert (
            ShadowPnLRecord("r2", "SHORT", 0.05, -0.02).virtual_pnl == pytest.approx(0.02)
        )
        assert (
            ShadowPnLRecord("r3", "NO_TRADE", None, 0.05).virtual_pnl == 0.0
        )
        assert (
            ShadowPnLRecord("r4", "LONG", 0.05, None).virtual_pnl is None
        )

    def test_total_and_realized_counts(self) -> None:
        pnl = ShadowPnL()
        pnl.add("r1", "LONG", 0.05, 0.03)
        pnl.add("r2", "SHORT", 0.05, -0.02)
        pnl.add("r3", "LONG", 0.05, None)  # offen
        assert pnl.count == 3
        assert pnl.realized_count == 2
        assert pnl.total_virtual_pnl == pytest.approx(0.05)
        assert len(pnl.records) == 3


class TestShadowMetrics:
    """Testet die zentrale Metrik-Sammlung."""

    def test_record_prediction_updates_brier_and_calibration(self) -> None:
        metrics = ShadowMetrics()
        metrics.record_prediction(
            predicted_up=0.7,
            predicted_down=0.2,
            predicted_range=0.1,
            actual="up",
            confidence=0.7,
            correct=True,
        )
        assert metrics.brier_score.mean_score == pytest.approx(0.14)
        assert metrics.calibration.total_observations == 1
        assert metrics.calibration.overall_accuracy == pytest.approx(1.0)

    def test_record_prediction_without_confidence_skips_calibration(self) -> None:
        metrics = ShadowMetrics()
        metrics.record_prediction(0.5, 0.3, 0.2, "up")
        assert metrics.brier_score.count == 1
        assert metrics.calibration.total_observations == 0

    def test_record_pnl_and_summary(self) -> None:
        metrics = ShadowMetrics()
        metrics.record_prediction(1.0, 0.0, 0.0, "up", confidence=0.9, correct=True)
        metrics.record_pnl("r1", "LONG", 0.05, 0.03)
        summary = metrics.summary
        assert summary["brier_score"]["mean"] == 0.0
        assert summary["calibration"]["total_observations"] == 1
        assert summary["pnl"]["total_virtual_pnl"] == pytest.approx(0.03)
        assert summary["pnl"]["realized_count"] == 1
        assert summary["pnl"]["total_count"] == 1


class TestShadowEngine:
    """Testet ShadowEngine gegen einen Pipeline-Stub (keine echte Pipeline)."""

    def test_run_records_decision_without_execution(self) -> None:
        engine = ShadowEngine(instrument="BTC/USD")
        fake = _FakePipeline(_make_pipeline_result("NO_TRADE", _make_consensus()))
        engine._pipeline = fake

        decision = engine.run(run_id="shadow-001", agents=[], market_data={})

        assert decision.run_id == "shadow-001"
        assert decision.instrument == "BTC/USD"
        assert decision.decision == "NO_TRADE"
        assert decision.is_no_trade is True
        assert decision.consensus is not None
        assert decision.consensus.decision == ConsensusDecision.NO_TRADE
        assert decision.latency_ms >= 0.0
        assert decision.errors == []
        # Pipeline wurde exakt einmal mit den Engine-Parametern aufgerufen
        assert len(fake.calls) == 1
        assert fake.calls[0]["run_id"] == "shadow-001"
        assert fake.calls[0]["instrument"] == "BTC/USD"

    def test_run_writes_audit_entry(self) -> None:
        engine = ShadowEngine(instrument="BTC/USD")
        engine._pipeline = _FakePipeline(
            _make_pipeline_result("LONG_BIAS", _make_consensus(ConsensusDecision.LONG_BIAS, 0.5))
        )
        engine.run(run_id="shadow-002", agents=[], market_data={})

        assert engine.decision_count == 1
        entry = engine.audit_trail.entries[0]
        assert entry.event_type == "decision"
        assert entry.agent_id == "shadow_engine"
        assert entry.actor == "shadow"
        assert entry.details["decision"] == "LONG_BIAS"
        assert entry.details["run_id"] == "shadow-002"
        assert entry.details["consensus"]["confidence"] == 0.5

    def test_custom_audit_trail_is_used(self) -> None:
        trail = AuditTrail()
        engine = ShadowEngine(instrument="ETH/USD", audit_trail=trail)
        assert engine.audit_trail is trail

    def test_run_single_round_records_audit_run_id(self) -> None:
        engine = ShadowEngine(instrument="BTC/USD")
        engine._pipeline = _FakePipeline(
            _make_pipeline_result("NO_TRADE", _make_consensus())
        )
        decision = engine.run_single_round(run_id="shadow-101", agents=[], market_data={})
        assert decision.run_id == "shadow-101"
        assert engine.audit_trail.entries[0].details["run_id"] == "shadow-101"

    def test_brier_score_rejects_unknown_actual_direction(self) -> None:
        bs = ShadowBrierScore()
        with pytest.raises(ValueError, match="Unknown actual direction"):
            bs.add(0.5, 0.3, 0.2, "NO_TRADE")


class TestShadowPaperComparison:
    """Testet den Shadow-vs-Paper-Vergleich."""

    @pytest.fixture
    def ts(self) -> datetime:
        return datetime(2026, 1, 1, tzinfo=UTC)

    def _record_pair(
        self,
        comp: ShadowPaperComparison,
        run_id: str,
        ts: datetime,
        shadow_direction: str = "LONG_BIAS",
        paper_direction: str = "LONG_BIAS",
        shadow_confidence: float = 0.8,
        paper_confidence: float = 0.75,
    ) -> None:
        comp.record_shadow(
            run_id=run_id,
            instrument="BTC/USD",
            timestamp=ts,
            direction=shadow_direction,
            confidence=shadow_confidence,
            consensus={"decision": shadow_direction},
            agent_reports=[],
            latency_ms=12.5,
        )
        comp.record_paper(
            run_id=run_id,
            timestamp=ts,
            direction=paper_direction,
            confidence=paper_confidence,
            instrument="BTC/USD",
        )

    def test_matching_decisions_agree(self, ts: datetime) -> None:
        comp = ShadowPaperComparison()
        self._record_pair(comp, "r1", ts)
        result = comp.compare("r1")
        assert result.direction_match is True
        assert result.trade_match is True
        assert result.overall_agreement is True
        assert result.confidence_diff == pytest.approx(0.05)
        assert result.latency_ms == 12.5

    def test_diverging_directions_counted(self, ts: datetime) -> None:
        comp = ShadowPaperComparison()
        self._record_pair(comp, "r1", ts)
        self._record_pair(comp, "r2", ts, paper_direction="SHORT_BIAS")
        comp.compare("r1")
        comp.compare("r2")

        assert comp.comparison_count == 2
        assert comp.agreement_rate == pytest.approx(0.5)
        assert comp.divergence_count == 1
        divergences = comp.get_divergences()
        assert len(divergences) == 1
        assert divergences[0].run_id == "r2"
        assert comp.avg_confidence_diff == pytest.approx(0.05)
        assert comp.avg_latency_ms == pytest.approx(12.5)

    def test_compare_missing_record_raises(self, ts: datetime) -> None:
        comp = ShadowPaperComparison()
        self._record_pair(comp, "r1", ts, paper_direction="NO_TRADE")
        with pytest.raises(KeyError, match="r2"):
            comp.compare("r2")
        # r1 wurde noch nicht verglichen -> Rates sind 0
        assert comp.agreement_rate == 0.0
        assert comp.direction_agreement_rate == 0.0

    def test_summary_keys(self, ts: datetime) -> None:
        comp = ShadowPaperComparison()
        self._record_pair(comp, "r1", ts)
        comp.compare("r1")
        summary = comp.summary
        assert summary["total_comparisons"] == 1
        assert summary["overall_agreement_rate"] == 1.0
        assert summary["divergence_count"] == 0
        assert summary["shadow_runs"] == 1
        assert summary["paper_runs"] == 1

    def test_paper_decision_properties(self, ts: datetime) -> None:
        from packages.shadow import PaperDecision

        no_trade = PaperDecision(
            run_id="r1", timestamp=ts, direction="NO_TRADE",
            confidence=0.1, instrument="BTC/USD",
        )
        assert no_trade.is_no_trade is True
        assert no_trade.is_trade is False

        trade = PaperDecision(
            run_id="r2", timestamp=ts, direction="RANGE",
            confidence=0.4, instrument="BTC/USD",
        )
        assert trade.is_no_trade is False
        assert trade.is_trade is True
