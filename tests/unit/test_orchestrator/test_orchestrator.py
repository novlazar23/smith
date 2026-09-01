"""Tests fuer die Orchestrator App — Stage-Management, Graphknoten, Audit-Trail.

Abdeckung:
  - graph.py: StageManager, AnalysisStage, AuditEvent, TradingGraphState
  - stages.py: create_run, market_snapshot, validate_data, features, regime
  - agents.py: first_round, seal, second_round, contrarian, multi-timeframe
  - consensus.py: calculate_consensus, generate_strategy, evaluate_portfolio
  - decision.py: risk gates, publish_decision, schedule_evaluation
  - Full pipeline, NO_TRADE scenarios, seal immutability
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from apps.orchestrator.agents import (
    build_multi_timeframe_view,
    run_contrarian_review,
    run_first_round,
    run_second_round,
    seal_first_round,
)
from apps.orchestrator.consensus import (
    calculate_consensus,
    evaluate_portfolio_step,
    generate_strategy_step,
)
from apps.orchestrator.decision import (
    Decision,
    apply_risk_gates,
    publish_decision,
    schedule_evaluation,
)
from apps.orchestrator.graph import (
    VALID_TRANSITIONS,
    AnalysisStage,
    AuditEvent,
    StageManager,
    TradingGraphState,
    transition,
)
from apps.orchestrator.stages import (
    build_market_snapshot,
    classify_regime,
    compute_features,
    create_run,
    validate_data,
)
from packages.consensus import (
    ConsensusDecision,
    ConsensusResult,
    WeightConfig,
    WeightedConsensusEngine,
)
from packages.schemas.agent_report import AgentReport, AgentStatus, EvidenceReference
from packages.schemas.final_decision import FinalDecisionType
from packages.strategy.engine import StrategyEngine

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def manager() -> StageManager:
    return StageManager()


@pytest.fixture
def minimal_state() -> TradingGraphState:
    return TradingGraphState(
        run_id="run-001",
        instrument="BTC/USD",
        request={"instrument": "BTC/USD", "timeframes": ["1h"]},
        analysis_time=datetime.now(UTC),
    )


@pytest.fixture
def agent_reports() -> list[AgentReport]:
    """Echte AgentReport-Objekte fuer Konsens-Tests."""
    return [
        AgentReport(
            report_id="rpt-001",
            run_id="run-001",
            agent_id="indicator-agent",
            agent_version="1.0.0",
            instrument="BTC/USD",
            horizon="1h",
            as_of=datetime.now(UTC),
            hypothesis="Bullish momentum",
            probabilities={"up": 0.65, "down": 0.15, "range": 0.20},
            evidence=[EvidenceReference(
                reference="indicator:rsi",
                feature="rsi",
                value="30",
                direction="positive",
                relevance=0.8,
            )],
            raw_confidence=0.7,
            calibrated_confidence=0.68,
            status=AgentStatus.ACTIVE,
        ),
        AgentReport(
            report_id="rpt-002",
            run_id="run-001",
            agent_id="chart-agent",
            agent_version="1.0.0",
            instrument="BTC/USD",
            horizon="1h",
            as_of=datetime.now(UTC),
            hypothesis="Breakout pattern",
            probabilities={"up": 0.70, "down": 0.10, "range": 0.20},
            evidence=[EvidenceReference(
                reference="chart:structure",
                feature="structure",
                value="bull_flag",
                direction="positive",
                relevance=0.7,
            )],
            raw_confidence=0.75,
            calibrated_confidence=0.72,
            status=AgentStatus.ACTIVE,
        ),
    ]


@pytest.fixture
def sample_reports() -> list[dict[str, Any]]:
    """Beispiel-Reports fuer Agenten-Tests."""
    return [
        {
            "report_id": "rpt-001",
            "agent_id": "indicator-agent",
            "agent_version": "1.0.0",
            "instrument": "BTC/USD",
            "horizon": "1h",
            "hypothesis": "Bullish momentum",
            "probabilities": {"up": 0.65, "down": 0.15, "range": 0.20},
            "evidence": [{"feature": "rsi", "value": "30", "direction": "positive", "relevance": 0.8}],
        },
        {
            "report_id": "rpt-002",
            "agent_id": "chart-agent",
            "agent_version": "1.0.0",
            "instrument": "BTC/USD",
            "horizon": "1h",
            "hypothesis": "Breakout pattern",
            "probabilities": {"up": 0.70, "down": 0.10, "range": 0.20},
            "evidence": [{"feature": "structure", "value": "bull_flag", "direction": "positive", "relevance": 0.7}],
        },
    ]


def _build_stage_pipeline(mgr: StageManager) -> None:
    """Transition through stages 11.1-11.12 (before consensus)."""
    for stage in [
        AnalysisStage.BUILD_MARKET_SNAPSHOT,
        AnalysisStage.VALIDATE_DATA,
        AnalysisStage.COMPUTE_FEATURES,
        AnalysisStage.CLASSIFY_REGIME,
        AnalysisStage.RUN_FIRST_ROUND,
        AnalysisStage.SEAL_FIRST_ROUND,
        AnalysisStage.HISTORICAL_VALIDATE,
        AnalysisStage.MEASURE_DEPENDENCIES,
        AnalysisStage.RUN_SECOND_ROUND,
        AnalysisStage.CONTRARIAN_REVIEW,
        AnalysisStage.MULTI_TIMEFRAME,
    ]:
        mgr.transition(stage)


def _consensus_state(state: TradingGraphState, mgr: StageManager,
                      reports: list[AgentReport]) -> tuple[TradingGraphState, StageManager, ConsensusResult]:
    """Run consensus and stay at CALCULATE_CONSENSUS."""
    state = state.model_copy(update={
        "first_round_reports": [r.model_dump() for r in reports],
        "market_snapshot": {"quality": 1.0, "critical_fields": {"price": True}},
        "data_quality": {"overall_quality": 1.0},
    })
    engine = WeightedConsensusEngine()
    state, mgr, consensus = calculate_consensus(state, mgr, engine)
    return state, mgr, consensus


def _consensus_and_advance(state: TradingGraphState, mgr: StageManager,
                           reports: list[AgentReport]) -> tuple[TradingGraphState, StageManager, ConsensusResult]:
    """Run consensus and advance through GENERATE_STRATEGY → EVALUATE_PORTFOLIO → APPLY_RISK_GATES."""
    state, mgr, consensus = _consensus_state(state, mgr, reports)
    for stage in [
        AnalysisStage.GENERATE_STRATEGY,
        AnalysisStage.EVALUATE_PORTFOLIO,
        AnalysisStage.APPLY_RISK_GATES,
    ]:
        mgr.transition(stage)
    return state, mgr, consensus


def _full_pipeline_state(
    state: TradingGraphState,
    mgr: StageManager,
    extra: dict[str, Any] | None = None,
) -> tuple[TradingGraphState, StageManager, ConsensusResult]:
    """Run full pipeline from CALCULATE_CONSENSUS through APPLY_RISK_GATES."""
    engine = WeightedConsensusEngine()
    state, mgr, consensus = calculate_consensus(state, mgr, engine)
    for stage in [
        AnalysisStage.GENERATE_STRATEGY,
        AnalysisStage.EVALUATE_PORTFOLIO,
        AnalysisStage.APPLY_RISK_GATES,
    ]:
        mgr.transition(stage)
    state = state.model_copy(update={
        "consensus_report": {
            "decision": consensus.decision.value,
            "confidence": consensus.confidence,
            "agreements": consensus.agent_agreements,
            "disagreements": consensus.agent_disagreements,
            "vote_distribution": consensus.vote_distribution,
            "agent_weights": consensus.agent_weights,
        },
        "data_quality": {"overall_quality": 1.0},
    })
    if extra:
        state = state.model_copy(update=extra)
    return state, mgr, consensus


def _pipeline_to(manager: StageManager) -> None:
    """Helper: advance manager through all stages after create_run."""
    for stage in [
        AnalysisStage.BUILD_MARKET_SNAPSHOT,
        AnalysisStage.VALIDATE_DATA,
        AnalysisStage.COMPUTE_FEATURES,
        AnalysisStage.CLASSIFY_REGIME,
        AnalysisStage.RUN_FIRST_ROUND,
        AnalysisStage.SEAL_FIRST_ROUND,
        AnalysisStage.HISTORICAL_VALIDATE,
        AnalysisStage.MEASURE_DEPENDENCIES,
        AnalysisStage.RUN_SECOND_ROUND,
        AnalysisStage.CONTRARIAN_REVIEW,
        AnalysisStage.MULTI_TIMEFRAME,
        AnalysisStage.CALCULATE_CONSENSUS,
        AnalysisStage.GENERATE_STRATEGY,
        AnalysisStage.EVALUATE_PORTFOLIO,
        AnalysisStage.APPLY_RISK_GATES,
    ]:
        manager.transition(stage)


# ═══════════════════════════════════════════════════════════════════
#  TESTS: graph.py — AnalysisStage, AuditEvent, TradingGraphState
# ═══════════════════════════════════════════════════════════════════


class TestAnalysisStage:
    """Testet AnalysisStage Enum und Sequenzen."""

    def test_all_18_stages_present(self) -> None:
        stages = list(AnalysisStage)
        assert len(stages) == 18

    def test_stage_values_match_spec(self) -> None:
        expected = {
            "create_run", "build_market_snapshot", "validate_data",
            "compute_features", "classify_regime", "run_first_round",
            "seal_first_round", "historical_validate", "measure_dependencies",
            "run_second_round", "contrarian_review", "multi_timeframe",
            "calculate_consensus", "generate_strategy", "evaluate_portfolio",
            "apply_risk_gates", "publish_decision", "schedule_evaluation",
        }
        actual = {s.value for s in AnalysisStage}
        assert actual == expected

    def test_default_sequence_returns_all_18(self) -> None:
        seq = AnalysisStage.default_sequence()
        assert len(seq) == 18
        assert seq[0] == AnalysisStage.CREATE_RUN
        assert seq[-1] == AnalysisStage.SCHEDULE_EVALUATION

    def test_default_sequence_order(self) -> None:
        seq = AnalysisStage.default_sequence()
        expected_order = [
            AnalysisStage.CREATE_RUN,
            AnalysisStage.BUILD_MARKET_SNAPSHOT,
            AnalysisStage.VALIDATE_DATA,
            AnalysisStage.COMPUTE_FEATURES,
            AnalysisStage.CLASSIFY_REGIME,
            AnalysisStage.RUN_FIRST_ROUND,
            AnalysisStage.SEAL_FIRST_ROUND,
            AnalysisStage.HISTORICAL_VALIDATE,
            AnalysisStage.MEASURE_DEPENDENCIES,
            AnalysisStage.RUN_SECOND_ROUND,
            AnalysisStage.CONTRARIAN_REVIEW,
            AnalysisStage.MULTI_TIMEFRAME,
            AnalysisStage.CALCULATE_CONSENSUS,
            AnalysisStage.GENERATE_STRATEGY,
            AnalysisStage.EVALUATE_PORTFOLIO,
            AnalysisStage.APPLY_RISK_GATES,
            AnalysisStage.PUBLISH_DECISION,
            AnalysisStage.SCHEDULE_EVALUATION,
        ]
        assert seq == expected_order

    def test_seal_first_round_allows_second_round_or_skip(self) -> None:
        """RUN_FIRST_ROUND kann zu SEAL_FIRST_ROUND oder RUN_SECOND_ROUND."""
        allowed = VALID_TRANSITIONS[AnalysisStage.RUN_FIRST_ROUND]
        assert AnalysisStage.SEAL_FIRST_ROUND in allowed
        assert AnalysisStage.RUN_SECOND_ROUND in allowed


class TestAuditEvent:
    """Testet AuditEvent dataclass."""

    def test_audit_event_creates_id(self) -> None:
        event = AuditEvent(stage="create_run")
        assert event.event_id != ""
        assert len(event.event_id) == 16

    def test_audit_event_timestamp_set(self) -> None:
        event = AuditEvent(stage="create_run")
        assert event.timestamp.tzinfo is not None

    def test_audit_event_default_empty(self) -> None:
        event = AuditEvent(stage="test_stage")
        assert event.inputs == {}
        assert event.outputs == {}
        assert event.duration_ms == 0.0

    def test_audit_event_with_fields(self) -> None:
        event = AuditEvent(
            stage="create_run",
            inputs={"key": "value"},
            outputs={"run_id": "r1"},
            duration_ms=150.0,
        )
        assert event.inputs == {"key": "value"}
        assert event.outputs == {"run_id": "r1"}
        assert event.duration_ms == 150.0

    def test_audit_event_immutability(self) -> None:
        event = AuditEvent(stage="test")
        with pytest.raises(Exception):
            event.stage = "changed"


class TestTradingGraphState:
    """Testet TradingGraphState dataclass."""

    def test_default_fields(self, minimal_state: TradingGraphState) -> None:
        assert minimal_state.run_id == "run-001"
        assert minimal_state.instrument == "BTC/USD"
        assert minimal_state.current_stage == "created"
        assert minimal_state.status == "pending"
        assert minimal_state.first_round_reports is None
        assert minimal_state.errors == []
        assert minimal_state.warnings == []

    def test_model_copy_immutability(self, minimal_state: TradingGraphState) -> None:
        new_state = minimal_state.model_copy(update={"run_id": "run-002"})
        assert new_state.run_id == "run-002"
        assert minimal_state.run_id == "run-001"

    def test_update_with_stage_data(self, minimal_state: TradingGraphState) -> None:
        new_state = minimal_state.model_copy(update={
            "market_snapshot_id": "hash-abc",
            "current_stage": AnalysisStage.BUILD_MARKET_SNAPSHOT.value,
        })
        assert new_state.market_snapshot_id == "hash-abc"
        assert new_state.current_stage == "build_market_snapshot"


# ═══════════════════════════════════════════════════════════════════
#  TESTS: graph.py — StageManager
# ═══════════════════════════════════════════════════════════════════


class TestStageManager:
    """Testet StageManager transitions, audit trail, invalid transitions."""

    def test_initial_state(self, manager: StageManager) -> None:
        assert manager.current_stage is None
        assert manager.stage_count() == 0
        assert manager.audit_events == ()
        assert manager.transition_log == []

    def test_transition_to_create_run(self, manager: StageManager) -> None:
        event = manager.transition(AnalysisStage.CREATE_RUN)
        assert manager.current_stage == AnalysisStage.CREATE_RUN
        assert manager.stage_count() == 1
        assert event.stage == "create_run"

    def test_sequential_transition(self, manager: StageManager) -> None:
        for stage in [
            AnalysisStage.CREATE_RUN,
            AnalysisStage.BUILD_MARKET_SNAPSHOT,
            AnalysisStage.VALIDATE_DATA,
        ]:
            manager.transition(stage)
        assert manager.current_stage == AnalysisStage.VALIDATE_DATA
        assert manager.stage_count() == 3

    def test_full_sequence(self, manager: StageManager) -> None:
        for stage in AnalysisStage.default_sequence():
            manager.transition(stage)
        assert manager.stage_count() == 18
        assert manager.current_stage == AnalysisStage.SCHEDULE_EVALUATION

    def test_invalid_transition_raises(self, manager: StageManager) -> None:
        manager.transition(AnalysisStage.CREATE_RUN)
        # VALIDATE_DATA != BUILD_MARKET_SNAPSHOT -> invalid
        with pytest.raises(ValueError, match="Ungültiger Übergang"):
            manager.transition(AnalysisStage.VALIDATE_DATA)

    def test_invalid_start_raises(self, manager: StageManager) -> None:
        with pytest.raises(ValueError, match="Ungültiger Übergang"):
            manager.transition(AnalysisStage.VALIDATE_DATA)

    def test_transition_log_records_all(self, manager: StageManager) -> None:
        manager.transition(AnalysisStage.CREATE_RUN)
        manager.transition(AnalysisStage.BUILD_MARKET_SNAPSHOT)
        assert len(manager.transition_log) == 2
        assert manager.transition_log[0] == (None, AnalysisStage.CREATE_RUN)

    def test_can_transition_positive(self, manager: StageManager) -> None:
        assert manager.can_transition(AnalysisStage.CREATE_RUN) is True
        manager.transition(AnalysisStage.CREATE_RUN)
        assert manager.can_transition(AnalysisStage.BUILD_MARKET_SNAPSHOT) is True

    def test_can_transition_negative(self, manager: StageManager) -> None:
        manager.transition(AnalysisStage.CREATE_RUN)
        assert manager.can_transition(AnalysisStage.VALIDATE_DATA) is False

    def test_has_completed(self, manager: StageManager) -> None:
        manager.transition(AnalysisStage.CREATE_RUN)
        assert manager.has_completed(AnalysisStage.CREATE_RUN) is True
        assert manager.has_completed(AnalysisStage.VALIDATE_DATA) is False

    def test_transition_with_inputs_outputs(self, manager: StageManager) -> None:
        event = manager.transition(
            AnalysisStage.CREATE_RUN,
            inputs={"instrument": "BTC/USD"},
            outputs={"run_id": "r1"},
            duration_ms=42.0,
        )
        assert event.inputs == {"instrument": "BTC/USD"}
        assert event.outputs == {"run_id": "r1"}
        assert event.duration_ms == 42.0

    def test_run_second_round_skip_after_seal(self, manager: StageManager) -> None:
        """RUN_SECOND_ROUND kann nach SEAL_FIRST_ROUND übersprungen werden."""
        for stage in [
            AnalysisStage.CREATE_RUN,
            AnalysisStage.BUILD_MARKET_SNAPSHOT,
            AnalysisStage.VALIDATE_DATA,
            AnalysisStage.COMPUTE_FEATURES,
            AnalysisStage.CLASSIFY_REGIME,
            AnalysisStage.RUN_FIRST_ROUND,
            AnalysisStage.SEAL_FIRST_ROUND,
        ]:
            manager.transition(stage)
        # Nach SEAL_FIRST_ROUND: HISTORICAL_VALIDATE ist erlaubt
        assert manager.can_transition(AnalysisStage.HISTORICAL_VALIDATE) is True


# ═══════════════════════════════════════════════════════════════════
#  Tests for stages.py
# ═══════════════════════════════════════════════════════════════════


class TestCreateRun:
    """Testet Stage 11.1 — create_run."""

    def test_creates_state_with_run_id(self) -> None:
        state, manager = create_run({
            "run_id": "my-run",
            "instrument": "ETH/USD",
        })
        assert state.run_id == "my-run"
        assert state.instrument == "ETH/USD"
        assert manager.current_stage == AnalysisStage.CREATE_RUN

    def test_generates_run_id_if_missing(self) -> None:
        state, _ = create_run({"instrument": "BTC/USD"})
        assert state.run_id != ""

    def test_records_audit_event(self) -> None:
        _, manager = create_run({"run_id": "r1"})
        assert manager.stage_count() == 1
        assert manager.has_completed(AnalysisStage.CREATE_RUN)

    def test_sets_model_versions(self) -> None:
        state, _ = create_run({
            "run_id": "r1",
            "model_version": "3.0",
            "prompt_version": "2.1",
        })
        assert state.model_version == "3.0"
        assert state.prompt_version == "2.1"

    def test_started_at_is_set(self) -> None:
        state, _ = create_run({"run_id": "r1"})
        assert state.started_at is not None


class TestBuildMarketSnapshot:
    """Testet Stage 11.2 — build_market_snapshot."""

    def test_creates_snapshot_hash(self) -> None:
        state, mgr = create_run({
            "run_id": "run-001",
            "instrument": "BTC/USD",
        })
        state, mgr = build_market_snapshot(state, mgr)
        assert state.market_snapshot_id is not None
        assert state.current_stage == AnalysisStage.BUILD_MARKET_SNAPSHOT.value
        assert mgr.has_completed(AnalysisStage.BUILD_MARKET_SNAPSHOT)

    def test_snapshot_data_stored(self) -> None:
        state, mgr = create_run({"run_id": "run-001", "instrument": "ETH/USD"})
        state, mgr = build_market_snapshot(state, mgr)
        assert state.market_snapshot is not None
        assert "instrument" in state.market_snapshot


class TestValidateData:
    """Testet Stage 11.3 — validate_data."""

    def test_valid_data_passes(self) -> None:
        state, mgr = create_run({"run_id": "r1", "instrument": "BTC/USD"})
        state, mgr = build_market_snapshot(state, mgr)
        state = state.model_copy(update={
            "market_snapshot": {"quality": 1.0, "critical_fields": {"price": True}},
        })
        state, mgr = validate_data(state, mgr)
        assert state.validation_status == "VALID"
        assert state.data_quality["valid"] is True

    def test_invalid_quality_triggers_no_trade(self) -> None:
        state, mgr = create_run({"run_id": "r2", "instrument": "BTC/USD"})
        state, mgr = build_market_snapshot(state, mgr)
        state = state.model_copy(update={
            "market_snapshot": {"quality": 0.5, "critical_fields": {}},
        })
        state, mgr = validate_data(state, mgr)
        assert state.validation_status == "NO_TRADE_DATA_QUALITY"
        assert state.status == "completed"
        assert state.data_quality["below_threshold"] is True

    def test_missing_critical_fields_no_trade(self) -> None:
        state, mgr = create_run({"run_id": "r3", "instrument": "BTC/USD"})
        state, mgr = build_market_snapshot(state, mgr)
        state = state.model_copy(update={
            "market_snapshot": {
                "quality": 1.0,
                "critical_fields": {"price": False, "volume": False},
            },
        })
        state, mgr = validate_data(state, mgr)
        assert state.validation_status == "NO_TRADE_DATA_QUALITY"
        assert len(state.data_quality["missing_critical_fields"]) > 0


class TestComputeFeatures:
    """Testet Stage 11.4 — compute_features."""

    def test_stores_features(self) -> None:
        state, mgr = create_run({"run_id": "r1", "instrument": "BTC/USD"})
        state, mgr = build_market_snapshot(state, mgr)
        state, mgr = validate_data(state, mgr)
        state, mgr = compute_features(state, mgr)
        assert state.current_stage == AnalysisStage.COMPUTE_FEATURES.value
        assert state.feature_snapshot_id is not None
        assert mgr.has_completed(AnalysisStage.COMPUTE_FEATURES)

    def test_features_with_provider(self) -> None:
        class FakeProvider:
            def compute(self, st: TradingGraphState) -> dict[str, Any]:
                return {"atr": 200.0, "volatility": 0.02}

        state, mgr = create_run({"run_id": "r2", "instrument": "BTC/USD"})
        state, mgr = build_market_snapshot(state, mgr)
        state, mgr = validate_data(state, mgr)
        state, mgr = compute_features(state, mgr, feature_provider=FakeProvider())
        assert state.features["atr"] == 200.0


class TestClassifyRegime:
    """Testet Stage 11.5 — classify_regime."""

    def test_stores_regime_report(self) -> None:
        state, mgr = create_run({"run_id": "r1", "instrument": "BTC/USD"})
        for stage in [
            AnalysisStage.BUILD_MARKET_SNAPSHOT,
            AnalysisStage.VALIDATE_DATA,
            AnalysisStage.COMPUTE_FEATURES,
        ]:
            mgr.transition(stage)
        state, mgr = classify_regime(state, mgr)
        assert state.current_stage == AnalysisStage.CLASSIFY_REGIME.value
        assert state.regime_report is not None
        assert mgr.has_completed(AnalysisStage.CLASSIFY_REGIME)

    def test_regime_with_provider(self) -> None:
        class FakeRegime:
            def classify(self, st: TradingGraphState) -> dict[str, Any]:
                return {"primary_regime": "trending", "stability": 0.8}

        state, mgr = create_run({"run_id": "r2", "instrument": "BTC/USD"})
        for stage in [
            AnalysisStage.BUILD_MARKET_SNAPSHOT,
            AnalysisStage.VALIDATE_DATA,
            AnalysisStage.COMPUTE_FEATURES,
        ]:
            mgr.transition(stage)
        state, mgr = classify_regime(state, mgr, regime_provider=FakeRegime())
        assert state.regime_report["primary_regime"] == "trending"


# ═══════════════════════════════════════════════════════════════════
#  Tests for agents.py
# ═══════════════════════════════════════════════════════════════════


class TestRunFirstRound:
    """Testet Stage 11.6 — run_first_round."""

    def test_empty_reports_without_registry(self) -> None:
        state, mgr = create_run({"run_id": "r1", "instrument": "BTC/USD"})
        for stage in [
            AnalysisStage.BUILD_MARKET_SNAPSHOT,
            AnalysisStage.VALIDATE_DATA,
            AnalysisStage.COMPUTE_FEATURES,
            AnalysisStage.CLASSIFY_REGIME,
        ]:
            mgr.transition(stage)
        state, mgr = run_first_round(state, mgr)
        assert state.first_round_reports == []
        assert state.current_stage == AnalysisStage.RUN_FIRST_ROUND.value

    def test_reports_stored(self) -> None:
        fake_reports = [
            {"report_id": "rpt-001", "agent_id": "indicator-agent",
             "agent_version": "1.0.0", "instrument": "BTC/USD", "horizon": "1h",
             "hypothesis": "Bullish momentum",
             "probabilities": {"up": 0.65, "down": 0.15, "range": 0.20},
             "evidence": [{"feature": "rsi", "value": "30", "direction": "positive", "relevance": 0.8}]},
            {"report_id": "rpt-002", "agent_id": "chart-agent",
             "agent_version": "1.0.0", "instrument": "BTC/USD", "horizon": "1h",
             "hypothesis": "Breakout pattern",
             "probabilities": {"up": 0.70, "down": 0.10, "range": 0.20},
             "evidence": [{"feature": "structure", "value": "bull_flag", "direction": "positive", "relevance": 0.7}]},
        ]
        state, mgr = create_run({"run_id": "r2", "instrument": "BTC/USD"})
        for stage in [
            AnalysisStage.BUILD_MARKET_SNAPSHOT,
            AnalysisStage.VALIDATE_DATA,
            AnalysisStage.COMPUTE_FEATURES,
            AnalysisStage.CLASSIFY_REGIME,
        ]:
            mgr.transition(stage)
        state, mgr = run_first_round(state, mgr, agent_registry=type('FakeRegistry', (),
            {'execute_first_round': lambda s, st: fake_reports})())
        assert len(state.first_round_reports) == 2
        assert state.peer_reports is None


class TestSealFirstRound:
    """Testet Stage 11.7 — seal_first_round (Unveränderbarkeit)."""

    def test_creates_seal_hash(self) -> None:
        state, mgr = create_run({"run_id": "r1", "instrument": "BTC/USD"})
        for stage in [
            AnalysisStage.BUILD_MARKET_SNAPSHOT,
            AnalysisStage.VALIDATE_DATA,
            AnalysisStage.COMPUTE_FEATURES,
            AnalysisStage.CLASSIFY_REGIME,
            AnalysisStage.RUN_FIRST_ROUND,
        ]:
            mgr.transition(stage)
        state, mgr = seal_first_round(state, mgr)
        assert state.seal_hash is not None
        assert state.sealed_at is not None
        assert mgr.has_completed(AnalysisStage.SEAL_FIRST_ROUND)

    def test_seal_is_deterministic(self) -> None:
        state_r1, mgr1 = create_run({"run_id": "r1", "instrument": "BTC/USD"})
        for stage in [
            AnalysisStage.BUILD_MARKET_SNAPSHOT,
            AnalysisStage.VALIDATE_DATA,
            AnalysisStage.COMPUTE_FEATURES,
            AnalysisStage.CLASSIFY_REGIME,
            AnalysisStage.RUN_FIRST_ROUND,
        ]:
            mgr1.transition(stage)
        state_r1, mgr1 = seal_first_round(state_r1, mgr1)

        state_r2, mgr2 = create_run({"run_id": "r1", "instrument": "BTC/USD"})
        for stage in [
            AnalysisStage.BUILD_MARKET_SNAPSHOT,
            AnalysisStage.VALIDATE_DATA,
            AnalysisStage.COMPUTE_FEATURES,
            AnalysisStage.CLASSIFY_REGIME,
            AnalysisStage.RUN_FIRST_ROUND,
        ]:
            mgr2.transition(stage)
        state_r2, mgr2 = seal_first_round(state_r2, mgr2)

        raw = f"{state_r1.run_id}|{state_r1.instrument}|{state_r1.analysis_time.isoformat()}"
        assert mgr1.seal(AnalysisStage.SEAL_FIRST_ROUND, raw) == state_r1.seal_hash

    def test_seal_verification(self) -> None:
        state, mgr = create_run({"run_id": "r1", "instrument": "BTC/USD"})
        for stage in [
            AnalysisStage.BUILD_MARKET_SNAPSHOT,
            AnalysisStage.VALIDATE_DATA,
            AnalysisStage.COMPUTE_FEATURES,
            AnalysisStage.CLASSIFY_REGIME,
            AnalysisStage.RUN_FIRST_ROUND,
        ]:
            mgr.transition(stage)
        state, mgr = seal_first_round(state, mgr)
        raw = f"{state.run_id}|{state.instrument}|{state.analysis_time.isoformat()}"
        assert mgr.verify_seal(AnalysisStage.SEAL_FIRST_ROUND, raw, state.seal_hash) is True

    def test_seal_immutability_tampering(self) -> None:
        state, mgr = create_run({"run_id": "r1", "instrument": "BTC/USD"})
        for stage in [
            AnalysisStage.BUILD_MARKET_SNAPSHOT,
            AnalysisStage.VALIDATE_DATA,
            AnalysisStage.COMPUTE_FEATURES,
            AnalysisStage.CLASSIFY_REGIME,
            AnalysisStage.RUN_FIRST_ROUND,
        ]:
            mgr.transition(stage)
        state, mgr = seal_first_round(state, mgr)
        raw = f"{state.run_id}|{state.instrument}|{state.analysis_time.isoformat()}"
        assert mgr.verify_seal(AnalysisStage.SEAL_FIRST_ROUND, raw + "_tampered",
                                   state.seal_hash) is False


class TestRunSecondRound:
    """Testet Stage 11.10 — run_second_round."""

    def test_stores_peer_reports(self) -> None:
        class FakeRegistry:
            def execute_first_round(self, st: TradingGraphState) -> list[dict]:
                return [{"report_id": "r1", "agent_id": "a1"}]
            def execute_second_round(self, st: TradingGraphState) -> list[dict]:
                return [{"report_id": "r2", "agent_id": "a2"}]
        state, mgr = create_run({"run_id": "r1", "instrument": "BTC/USD"})
        for stage in [
            AnalysisStage.BUILD_MARKET_SNAPSHOT,
            AnalysisStage.VALIDATE_DATA,
            AnalysisStage.COMPUTE_FEATURES,
            AnalysisStage.CLASSIFY_REGIME,
        ]:
            mgr.transition(stage)
        state, mgr = run_first_round(state, mgr, agent_registry=FakeRegistry())
        mgr.transition(AnalysisStage.SEAL_FIRST_ROUND)
        mgr.transition(AnalysisStage.HISTORICAL_VALIDATE)
        mgr.transition(AnalysisStage.MEASURE_DEPENDENCIES)
        state, mgr = run_second_round(state, mgr, agent_registry=FakeRegistry())
        assert state.second_round_reports == [{"report_id": "r2", "agent_id": "a2"}]
        assert state.peer_reports is not None

    def test_peer_reports_available(self) -> None:
        state, mgr = create_run({"run_id": "r2", "instrument": "BTC/USD"})
        for stage in [
            AnalysisStage.BUILD_MARKET_SNAPSHOT,
            AnalysisStage.VALIDATE_DATA,
            AnalysisStage.COMPUTE_FEATURES,
            AnalysisStage.CLASSIFY_REGIME,
        ]:
            mgr.transition(stage)
        state, mgr = run_first_round(
            state, mgr,
            agent_registry=type('Fake', (), {
                'execute_first_round': lambda s, st: [
                    {"report_id": "r1", "agent_id": "a1", "probabilities": {"up": 0.7, "down": 0.1, "range": 0.2},
                     "hypothesis": "test", "agent_version": "1.0", "horizon": "1h"},
                ],
            })(),
        )
        mgr.transition(AnalysisStage.SEAL_FIRST_ROUND)
        mgr.transition(AnalysisStage.HISTORICAL_VALIDATE)
        mgr.transition(AnalysisStage.MEASURE_DEPENDENCIES)
        state, mgr = run_second_round(state, mgr)
        assert state.peer_reports is not None
        assert len(state.peer_reports) == 1


class TestContrarianReview:
    """Testet Stage 11.11 — run_contrarian_review."""

    def test_creates_contrarian_report(self) -> None:
        state, mgr = create_run({"run_id": "r1", "instrument": "BTC/USD"})
        for stage in [
            AnalysisStage.BUILD_MARKET_SNAPSHOT,
            AnalysisStage.VALIDATE_DATA,
            AnalysisStage.COMPUTE_FEATURES,
            AnalysisStage.CLASSIFY_REGIME,
            AnalysisStage.RUN_FIRST_ROUND,
            AnalysisStage.SEAL_FIRST_ROUND,
            AnalysisStage.HISTORICAL_VALIDATE,
            AnalysisStage.MEASURE_DEPENDENCIES,
        ]:
            mgr.transition(stage)
        state, mgr = run_contrarian_review(state, mgr)
        assert state.contrarian_report is not None
        assert state.current_stage == AnalysisStage.CONTRARIAN_REVIEW.value

    def test_strongest_hypothesis_detected(self, sample_reports: list[dict[str, Any]]) -> None:
        state, mgr = create_run({"run_id": "r2", "instrument": "BTC/USD"})
        for stage in [
            AnalysisStage.BUILD_MARKET_SNAPSHOT,
            AnalysisStage.VALIDATE_DATA,
            AnalysisStage.COMPUTE_FEATURES,
            AnalysisStage.CLASSIFY_REGIME,
            AnalysisStage.RUN_FIRST_ROUND,
            AnalysisStage.SEAL_FIRST_ROUND,
            AnalysisStage.HISTORICAL_VALIDATE,
            AnalysisStage.MEASURE_DEPENDENCIES,
        ]:
            mgr.transition(stage)
        state = state.model_copy(update={"first_round_reports": sample_reports})
        state, mgr = run_contrarian_review(state, mgr)
        report = state.contrarian_report
        assert report["strongest_hypothesis"] is not None
        assert report["strongest_confidence"] > 0


class TestBuildMultiTimeframe:
    """Testet Stage 11.12 — build_multi_timeframe_view."""

    def test_creates_tf_view(self) -> None:
        state, mgr = create_run({"run_id": "r1", "instrument": "BTC/USD"})
        for stage in [
            AnalysisStage.BUILD_MARKET_SNAPSHOT,
            AnalysisStage.VALIDATE_DATA,
            AnalysisStage.COMPUTE_FEATURES,
            AnalysisStage.CLASSIFY_REGIME,
            AnalysisStage.RUN_FIRST_ROUND,
            AnalysisStage.SEAL_FIRST_ROUND,
            AnalysisStage.HISTORICAL_VALIDATE,
            AnalysisStage.MEASURE_DEPENDENCIES,
            AnalysisStage.RUN_SECOND_ROUND,
            AnalysisStage.CONTRARIAN_REVIEW,
        ]:
            mgr.transition(stage)
        state, mgr = build_multi_timeframe_view(state, mgr)
        assert state.multi_timeframe_report is not None
        assert "timeframes" in state.multi_timeframe_report
        assert len(state.multi_timeframe_report["timeframes"]) == 6
        assert state.current_stage == AnalysisStage.MULTI_TIMEFRAME.value


# ═══════════════════════════════════════════════════════════════════
#  Tests for consensus.py
# ═══════════════════════════════════════════════════════════════════


class TestCalculateConsensus:
    """Testet Stage 11.13 — calculate_consensus."""

    def test_consensus_from_reports(self, agent_reports: list[AgentReport]) -> None:
        state, mgr = create_run({"run_id": "r1", "instrument": "BTC/USD"})
        _build_stage_pipeline(mgr)
        state = state.model_copy(update={
            "first_round_reports": [r.model_dump() for r in agent_reports],
            "market_snapshot": {"quality": 1.0, "critical_fields": {"price": True}},
            "data_quality": {"overall_quality": 1.0},
        })
        engine = WeightedConsensusEngine()
        state, mgr, result = calculate_consensus(state, mgr, engine)
        assert result.decision == ConsensusDecision.LONG_BIAS
        assert result.confidence > 0
        assert state.consensus_report is not None

    def test_empty_reports_no_trade(self) -> None:
        state, mgr = create_run({"run_id": "r2", "instrument": "BTC/USD"})
        _build_stage_pipeline(mgr)
        engine = WeightedConsensusEngine()
        state, mgr, result = calculate_consensus(state, mgr, engine)
        assert result.decision == ConsensusDecision.NO_TRADE

    def test_mixed_votes(self) -> None:
        engine = WeightedConsensusEngine(config=WeightConfig(min_consensus_threshold=0.8))
        mixed = [
            AgentReport(
                report_id="l1", run_id="r3", agent_id="long-agent",
                agent_version="1.0", instrument="BTC/USD", horizon="1h",
                as_of=datetime.now(UTC), hypothesis="long",
                probabilities={"up": 0.7, "down": 0.1, "range": 0.2},
                evidence=[EvidenceReference(reference="e1", feature="f", value="v",
                     direction="positive", relevance=0.5)],
                raw_confidence=0.7, status=AgentStatus.ACTIVE,
            ),
            AgentReport(
                report_id="s1", run_id="r3", agent_id="short-agent",
                agent_version="1.0", instrument="BTC/USD", horizon="1h",
                as_of=datetime.now(UTC), hypothesis="short",
                probabilities={"up": 0.1, "down": 0.7, "range": 0.2},
                evidence=[EvidenceReference(reference="e2", feature="f", value="v",
                     direction="negative", relevance=0.5)],
                raw_confidence=0.7, status=AgentStatus.ACTIVE,
            ),
        ]
        state, mgr = create_run({"run_id": "r3", "instrument": "BTC/USD"})
        _build_stage_pipeline(mgr)
        state = state.model_copy(update={
            "first_round_reports": [r.model_dump() for r in mixed],
            "data_quality": {"overall_quality": 1.0},
        })
        state, _, result = calculate_consensus(state, mgr, engine)
        assert result.decision == ConsensusDecision.NO_TRADE


class TestGenerateStrategy:
    """Testet Stage 11.14 — generate_strategy."""

    def test_generates_variants(self, agent_reports: list[AgentReport]) -> None:
        state, mgr = create_run({"run_id": "r1", "instrument": "BTC/USD"})
        _build_stage_pipeline(mgr)
        state, mgr, consensus = _consensus_state(state, mgr, agent_reports)
        state = state.model_copy(update={
            "consensus_report": {
                "decision": consensus.decision.value,
                "confidence": consensus.confidence,
                "agent_weights": consensus.agent_weights,
                "agreements": consensus.agent_agreements,
                "disagreements": consensus.agent_disagreements,
                "vote_distribution": consensus.vote_distribution,
            },
            "features": {"current_price": 50000.0, "atr": 500.0},
        })
        strategy = StrategyEngine()
        state, _ = generate_strategy_step(state, mgr, strategy, consensus)
        assert state.strategy_report is not None
        assert state.strategy_report["variant_count"] == 3
        assert state.current_stage == AnalysisStage.GENERATE_STRATEGY.value

    def test_no_variants_for_no_trade(self) -> None:
        state, mgr = create_run({"run_id": "r2", "instrument": "BTC/USD"})
        for stage in [
            AnalysisStage.BUILD_MARKET_SNAPSHOT,
            AnalysisStage.VALIDATE_DATA,
            AnalysisStage.COMPUTE_FEATURES,
            AnalysisStage.CLASSIFY_REGIME,
            AnalysisStage.RUN_FIRST_ROUND,
            AnalysisStage.SEAL_FIRST_ROUND,
            AnalysisStage.HISTORICAL_VALIDATE,
            AnalysisStage.MEASURE_DEPENDENCIES,
            AnalysisStage.RUN_SECOND_ROUND,
            AnalysisStage.CONTRARIAN_REVIEW,
            AnalysisStage.MULTI_TIMEFRAME,
        ]:
            mgr.transition(stage)
        mgr.transition(AnalysisStage.CALCULATE_CONSENSUS)
        state = state.model_copy(update={
            "consensus_report": {
                "decision": ConsensusDecision.NO_TRADE.value,
                "confidence": 0.0,
                "agreements": [], "disagreements": [], "vote_distribution": {},
                "agent_weights": {},
            },
            "data_quality": {"overall_quality": 1.0},
        })
        strategy = StrategyEngine()
        no_trade_consensus = ConsensusResult(
            decision=ConsensusDecision.NO_TRADE,
            vote_distribution={}, agent_weights={},
            agent_agreements=[], agent_disagreements=[],
            confidence=0.0, reason="no consensus",
        )
        state, _ = generate_strategy_step(state, mgr, strategy, no_trade_consensus)
        assert state.strategy_report["variant_count"] == 0


class TestEvaluatePortfolio:
    """Testet Stage 11.15 — evaluate_portfolio."""

    def test_stores_portfolio_report(self) -> None:
        state, mgr = create_run({"run_id": "r1", "instrument": "BTC/USD"})
        for stage in [
            AnalysisStage.BUILD_MARKET_SNAPSHOT,
            AnalysisStage.VALIDATE_DATA,
            AnalysisStage.COMPUTE_FEATURES,
            AnalysisStage.CLASSIFY_REGIME,
            AnalysisStage.RUN_FIRST_ROUND,
            AnalysisStage.SEAL_FIRST_ROUND,
            AnalysisStage.HISTORICAL_VALIDATE,
            AnalysisStage.MEASURE_DEPENDENCIES,
            AnalysisStage.RUN_SECOND_ROUND,
            AnalysisStage.CONTRARIAN_REVIEW,
            AnalysisStage.MULTI_TIMEFRAME,
            AnalysisStage.CALCULATE_CONSENSUS,
            AnalysisStage.GENERATE_STRATEGY,
        ]:
            mgr.transition(stage)
        state, _ = evaluate_portfolio_step(state, mgr)
        assert state.portfolio_report is not None
        assert "existing_exposures" in state.portfolio_report
        assert state.current_stage == AnalysisStage.EVALUATE_PORTFOLIO.value

    def test_with_portfolio_manager(self) -> None:
        state, mgr = create_run({"run_id": "r2", "instrument": "BTC/USD"})
        for stage in [
            AnalysisStage.BUILD_MARKET_SNAPSHOT,
            AnalysisStage.VALIDATE_DATA,
            AnalysisStage.COMPUTE_FEATURES,
            AnalysisStage.CLASSIFY_REGIME,
            AnalysisStage.RUN_FIRST_ROUND,
            AnalysisStage.SEAL_FIRST_ROUND,
            AnalysisStage.HISTORICAL_VALIDATE,
            AnalysisStage.MEASURE_DEPENDENCIES,
            AnalysisStage.RUN_SECOND_ROUND,
            AnalysisStage.CONTRARIAN_REVIEW,
            AnalysisStage.MULTI_TIMEFRAME,
            AnalysisStage.CALCULATE_CONSENSUS,
            AnalysisStage.GENERATE_STRATEGY,
        ]:
            mgr.transition(stage)
        class FakePM:
            def get_exposures(self, st: TradingGraphState) -> list[dict]:
                return [{"symbol": "BTC", "size": 0.3}]
            def correlation_score(self, st: TradingGraphState) -> float:
                return 0.7
            def concentration_score(self, st: TradingGraphState) -> float:
                return 0.8
        state, _ = evaluate_portfolio_step(state, mgr, portfolio_manager=FakePM())
        assert state.portfolio_report["max_exposure"] == 0.3
        assert state.portfolio_report["new_trade_allowed"] is True


# ═══════════════════════════════════════════════════════════════════
#  Tests for decision.py
# ═══════════════════════════════════════════════════════════════════


class TestApplyRiskGates:
    """Testet Stage 11.16 — apply_risk_gates."""

    def test_approves_without_manager(self) -> None:
        """apply_risk_gates ohne RiskManager — Default aprobiert."""
        state, mgr = create_run({"run_id": "r1", "instrument": "BTC/USD"})
        _build_stage_pipeline(mgr)
        state, mgr, consensus = _consensus_state(state, mgr, agent_reports_fixture())
        for stage in [
            AnalysisStage.GENERATE_STRATEGY,
            AnalysisStage.EVALUATE_PORTFOLIO,
        ]:
            mgr.transition(stage)
        state = state.model_copy(update={
            "consensus_report": {
                "decision": consensus.decision.value,
                "confidence": consensus.confidence,
                "agreements": consensus.agent_agreements,
                "disagreements": consensus.agent_disagreements,
                "vote_distribution": consensus.vote_distribution,
                "agent_weights": consensus.agent_weights,
            },
            "strategy_report": {"best_proposal": {"expected_return_net": 0.005}},
            "data_quality": {"overall_quality": 1.0},
        })
        state, mgr = apply_risk_gates(state, mgr)
        assert state.risk_report["approved"] is True
        assert state.risk_report["veto"] is False

    def test_blocks_on_low_quality(self) -> None:
        state, mgr = create_run({"run_id": "r2", "instrument": "BTC/USD"})
        _build_stage_pipeline(mgr)
        state, mgr, consensus = _consensus_state(state, mgr, agent_reports_fixture())
        for stage in [
            AnalysisStage.GENERATE_STRATEGY,
            AnalysisStage.EVALUATE_PORTFOLIO,
        ]:
            mgr.transition(stage)
        state = state.model_copy(update={
            "consensus_report": {
                "decision": consensus.decision.value,
                "confidence": consensus.confidence,
                "agreements": consensus.agent_agreements,
                "disagreements": consensus.agent_disagreements,
                "vote_distribution": consensus.vote_distribution,
                "agent_weights": consensus.agent_weights,
            },
            "data_quality": {"overall_quality": 0.5, "valid": False},
        })
        state, mgr = apply_risk_gates(state, mgr)
        assert state.risk_report["approved"] is False
        assert state.risk_report["veto"] is True

    def test_blocks_on_negative_edge(self) -> None:
        state, mgr = create_run({"run_id": "r3", "instrument": "BTC/USD"})
        _build_stage_pipeline(mgr)
        state, mgr, consensus = _consensus_state(state, mgr, agent_reports_fixture())
        for stage in [
            AnalysisStage.GENERATE_STRATEGY,
            AnalysisStage.EVALUATE_PORTFOLIO,
        ]:
            mgr.transition(stage)
        state = state.model_copy(update={
            "consensus_report": {
                "decision": consensus.decision.value,
                "confidence": consensus.confidence,
                "agreements": consensus.agent_agreements,
                "disagreements": consensus.agent_disagreements,
                "vote_distribution": consensus.vote_distribution,
                "agent_weights": consensus.agent_weights,
            },
            "strategy_report": {"best_proposal": {"expected_return_net": -0.01}},
            "data_quality": {"overall_quality": 1.0},
        })
        state, mgr = apply_risk_gates(state, mgr)
        assert state.risk_report["approved"] is False


class TestPublishDecision:
    """Testet Stage 11.17 — publish_decision."""

    def _full_pipeline_state(self, extra: dict[str, Any] | None = None) -> tuple[TradingGraphState, StageManager, ConsensusResult]:
        """Build state through full pipeline (before publish)."""
        state, mgr = create_run({"run_id": "pd-001", "instrument": "BTC/USD"})
        _build_stage_pipeline(mgr)
        state, mgr, consensus = _consensus_state(state, mgr, agent_reports_fixture())
        for stage in [
            AnalysisStage.GENERATE_STRATEGY,
            AnalysisStage.EVALUATE_PORTFOLIO,
            AnalysisStage.APPLY_RISK_GATES,
        ]:
            mgr.transition(stage)
        state = state.model_copy(update={
            "consensus_report": {
                "decision": consensus.decision.value,
                "confidence": consensus.confidence,
                "agreements": consensus.agent_agreements,
                "disagreements": consensus.agent_disagreements,
                "vote_distribution": consensus.vote_distribution,
                "agent_weights": consensus.agent_weights,
            },
            "data_quality": {"overall_quality": 1.0},
        })
        if extra:
            state = state.model_copy(update=extra)
        return state, mgr, consensus

    def test_publishes_long_bias(self, agent_reports: list[AgentReport]) -> None:
        """Pipeline mit agent_reports — publishDecision mit LONG_BIAS."""
        state, mgr = create_run({"run_id": "pd-lb", "instrument": "BTC/USD"})
        _build_stage_pipeline(mgr)
        state, mgr, consensus = _consensus_state(state, mgr, agent_reports)
        for stage in [
            AnalysisStage.GENERATE_STRATEGY,
            AnalysisStage.EVALUATE_PORTFOLIO,
            AnalysisStage.APPLY_RISK_GATES,
        ]:
            mgr.transition(stage)
        state = state.model_copy(update={
            "consensus_report": {
                "decision": consensus.decision.value,
                "confidence": consensus.confidence,
                "agreements": consensus.agent_agreements,
                "disagreements": consensus.agent_disagreements,
                "vote_distribution": consensus.vote_distribution,
                "agent_weights": consensus.agent_weights,
            },
            "data_quality": {"overall_quality": 1.0},
        })
        decision = publish_decision(state, mgr)
        assert decision.direction in (FinalDecisionType.LONG_BIAS, FinalDecisionType.NO_TRADE)
        assert decision.run_id == "pd-lb"
        assert decision.reasoning != ""

    def test_publishes_no_trade_data_quality(self) -> None:
        state, mgr, _ = self._full_pipeline_state({
            "validation_status": "NO_TRADE_DATA_QUALITY",
            "data_quality": {"overall_quality": 0.5},
        })
        decision = publish_decision(state, mgr)
        assert decision.direction == FinalDecisionType.NO_TRADE_DATA_QUALITY

    def test_publishes_no_trade_risk(self) -> None:
        state, mgr, _ = self._full_pipeline_state({
            "data_quality": {"overall_quality": 1.0},
            "risk_report": {"approved": False, "veto": True, "blocking_reasons": ["exposure"]},
            "consensus_report": {
                "decision": "LONG_BIAS", "confidence": 0.8,
                "agreements": [], "disagreements": [], "vote_distribution": {},
                "agent_weights": {},
            },
        })
        decision = publish_decision(state, mgr)
        assert decision.direction == FinalDecisionType.NO_TRADE_RISK

    def test_sets_completed_status(self) -> None:
        state, mgr, _ = self._full_pipeline_state({
            "data_quality": {"overall_quality": 1.0},
            "consensus_report": {
                "decision": "LONG_BIAS", "confidence": 0.8,
                "agreements": [], "disagreements": [], "vote_distribution": {},
                "agent_weights": {},
            },
        })
        decision = publish_decision(state, mgr)
        assert decision is not None
        assert decision.run_id == "pd-001"
        assert decision.direction is not None


class TestScheduleEvaluation:
    """Testet Stage 11.18 — schedule_evaluation."""

    def test_schedules_eval(self) -> None:
        state, mgr = create_run({"run_id": "se-001", "instrument": "BTC/USD"})
        _build_stage_pipeline(mgr)
        state, mgr, consensus = _consensus_state(state, mgr, agent_reports_fixture())
        state = state.model_copy(update={
            "consensus_report": {
                "decision": consensus.decision.value,
                "confidence": consensus.confidence,
                "agreements": consensus.agent_agreements,
                "disagreements": consensus.agent_disagreements,
                "vote_distribution": consensus.vote_distribution,
                "agent_weights": consensus.agent_weights,
            },
            "data_quality": {"overall_quality": 1.0},
            "strategy_report": {
                "best_proposal": {"entry_price": 50000, "stop_loss": 49000, "targets": [51000]},
            },
        })
        for stage in [
            AnalysisStage.GENERATE_STRATEGY,
            AnalysisStage.EVALUATE_PORTFOLIO,
            AnalysisStage.APPLY_RISK_GATES,
            AnalysisStage.PUBLISH_DECISION,
        ]:
            mgr.transition(stage)
        state, mgr = schedule_evaluation(state, mgr)
        assert state.evaluation_schedule is not None
        assert state.evaluation_schedule["horizon"] == "1h"
        assert state.current_stage == AnalysisStage.SCHEDULE_EVALUATION.value

    def test_custom_horizon(self) -> None:
        state, mgr = create_run({"run_id": "se-002", "instrument": "BTC/USD"})
        _build_stage_pipeline(mgr)
        state, mgr, consensus = _consensus_state(state, mgr, agent_reports_fixture())
        state = state.model_copy(update={
            "consensus_report": {
                "decision": consensus.decision.value,
                "confidence": consensus.confidence,
                "agreements": consensus.agent_agreements,
                "disagreements": consensus.agent_disagreements,
                "vote_distribution": consensus.vote_distribution,
                "agent_weights": consensus.agent_weights,
            },
            "data_quality": {"overall_quality": 1.0},
            "strategy_report": {
                "best_proposal": {"entry_price": 50000, "stop_loss": 49000, "targets": [51000]},
            },
        })
        for stage in [
            AnalysisStage.GENERATE_STRATEGY,
            AnalysisStage.EVALUATE_PORTFOLIO,
            AnalysisStage.APPLY_RISK_GATES,
            AnalysisStage.PUBLISH_DECISION,
        ]:
            mgr.transition(stage)
        state, mgr = schedule_evaluation(state, mgr, evaluation_horizon="4h")
        assert state.evaluation_schedule["horizon"] == "4h"


# ═══════════════════════════════════════════════════════════════════
#  TESTS: Full pipeline
# ═══════════════════════════════════════════════════════════════════


class TestFullPipeline:
    """Testet den gesamten Pipeline-Flow von run → publish_decision."""

    def test_full_pipeline_long_bias(self, agent_reports: list[AgentReport]) -> None:
        """Vollstaendliche Pipeline: create_run → publish_decision."""
        state, manager = create_run({
            "run_id": "pipeline-001",
            "instrument": "BTC/USD",
            "timeframes": ["1h"],
        })

        # Stages 1-5
        for stage in [
            AnalysisStage.BUILD_MARKET_SNAPSHOT,
            AnalysisStage.VALIDATE_DATA,
            AnalysisStage.COMPUTE_FEATURES,
            AnalysisStage.CLASSIFY_REGIME,
        ]:
            manager.transition(stage)

        # First round mit Report
        state = state.model_copy(update={
            "first_round_reports": [r.model_dump() for r in agent_reports],
            "market_snapshot": {"quality": 1.0, "critical_fields": {"price": True}},
            "data_quality": {"overall_quality": 1.0},
        })

        # Seal
        manager.transition(AnalysisStage.RUN_FIRST_ROUND)
        manager.transition(AnalysisStage.SEAL_FIRST_ROUND)

        # Remaining pre-consensus stages
        for stage in [
            AnalysisStage.HISTORICAL_VALIDATE,
            AnalysisStage.MEASURE_DEPENDENCIES,
            AnalysisStage.RUN_SECOND_ROUND,
            AnalysisStage.CONTRARIAN_REVIEW,
            AnalysisStage.MULTI_TIMEFRAME,
        ]:
            manager.transition(stage)

        # Consensus
        engine = WeightedConsensusEngine()
        state, manager, consensus = calculate_consensus(state, manager, engine)
        assert consensus.decision == ConsensusDecision.LONG_BIAS

        # Strategy
        state = state.model_copy(update={
            "consensus_report": {
                "decision": consensus.decision.value,
                "confidence": consensus.confidence,
                "agreements": consensus.agent_agreements,
                "disagreements": consensus.agent_disagreements,
                "vote_distribution": consensus.vote_distribution,
                "agent_weights": consensus.agent_weights,
            },
            "features": {"current_price": 50000.0, "atr": 500.0},
            "data_quality": {"overall_quality": 1.0},
        })
        strategy = StrategyEngine()
        state, manager = generate_strategy_step(state, manager, strategy, consensus)

        # Portfolio
        state, manager = evaluate_portfolio_step(state, manager)

        # Risk — default approved (quality=1.0, positive edge from strategy engine)
        state, manager = apply_risk_gates(state, manager)

        # Publish
        decision = publish_decision(state, manager)
        assert decision.run_id == "pipeline-001"
        assert decision.direction in (
            FinalDecisionType.LONG_BIAS,
            FinalDecisionType.NO_TRADE,
        )
        assert manager.stage_count() > 0

    def test_full_pipeline_skips_second_round(self, agent_reports: list[AgentReport]) -> None:
        """Pipeline mit SKIP von RUN_SECOND_ROUND."""
        _, manager = create_run({"run_id": "skip-001", "instrument": "ETH/USD"})
        for stage in [
            AnalysisStage.BUILD_MARKET_SNAPSHOT,
            AnalysisStage.VALIDATE_DATA,
            AnalysisStage.COMPUTE_FEATURES,
            AnalysisStage.CLASSIFY_REGIME,
            AnalysisStage.RUN_FIRST_ROUND,
            AnalysisStage.SEAL_FIRST_ROUND,
            AnalysisStage.HISTORICAL_VALIDATE,
            AnalysisStage.MEASURE_DEPENDENCIES,
            AnalysisStage.CONTRARIAN_REVIEW,
            AnalysisStage.MULTI_TIMEFRAME,
            AnalysisStage.CALCULATE_CONSENSUS,
            AnalysisStage.GENERATE_STRATEGY,
            AnalysisStage.EVALUATE_PORTFOLIO,
            AnalysisStage.APPLY_RISK_GATES,
            AnalysisStage.PUBLISH_DECISION,
            AnalysisStage.SCHEDULE_EVALUATION,
        ]:
            manager.transition(stage)
        assert manager.stage_count() == 17  # RUN_SECOND_ROUND uebersprungen


# ═══════════════════════════════════════════════════════════════════
#  TESTS: NO_TRADE scenarios
# ═══════════════════════════════════════════════════════════════════


class TestNoTradeScenarios:
    """Testet NO_TRADE Szenarien."""

    def test_no_trade_data_quality(self) -> None:
        """Validation NO_TRADE_DATA_QUALITY → publish_decision mit NO_TRADE_DATA_QUALITY."""
        state, manager = create_run({"run_id": "nt-dq", "instrument": "BTC/USD"})
        for stage in [
            AnalysisStage.BUILD_MARKET_SNAPSHOT,
        ]:
            manager.transition(stage)
        state = state.model_copy(update={
            "market_snapshot": {"quality": 0.3},
        })
        state, manager = validate_data(state, manager)
        assert state.validation_status == "NO_TRADE_DATA_QUALITY"

        for stage in [
            AnalysisStage.COMPUTE_FEATURES,
            AnalysisStage.CLASSIFY_REGIME,
            AnalysisStage.RUN_FIRST_ROUND,
            AnalysisStage.SEAL_FIRST_ROUND,
            AnalysisStage.HISTORICAL_VALIDATE,
            AnalysisStage.MEASURE_DEPENDENCIES,
            AnalysisStage.RUN_SECOND_ROUND,
            AnalysisStage.CONTRARIAN_REVIEW,
            AnalysisStage.MULTI_TIMEFRAME,
            AnalysisStage.CALCULATE_CONSENSUS,
            AnalysisStage.GENERATE_STRATEGY,
            AnalysisStage.EVALUATE_PORTFOLIO,
            AnalysisStage.APPLY_RISK_GATES,
        ]:
            manager.transition(stage)

        decision = publish_decision(state, manager)
        assert decision.direction == FinalDecisionType.NO_TRADE_DATA_QUALITY

    def test_no_trade_insufficient_edge(self) -> None:
        """Insufficient edge → NO_TRADE_INSUFFICIENT_EDGE."""
        state, manager = create_run({"run_id": "nt-ie", "instrument": "BTC/USD"})
        for stage in [
            AnalysisStage.BUILD_MARKET_SNAPSHOT,
            AnalysisStage.VALIDATE_DATA,
            AnalysisStage.COMPUTE_FEATURES,
            AnalysisStage.CLASSIFY_REGIME,
            AnalysisStage.RUN_FIRST_ROUND,
            AnalysisStage.SEAL_FIRST_ROUND,
            AnalysisStage.HISTORICAL_VALIDATE,
            AnalysisStage.MEASURE_DEPENDENCIES,
            AnalysisStage.RUN_SECOND_ROUND,
            AnalysisStage.CONTRARIAN_REVIEW,
            AnalysisStage.MULTI_TIMEFRAME,
        ]:
            manager.transition(stage)
        engine = WeightedConsensusEngine(config=WeightConfig(min_consensus_threshold=0.8))
        state, _, consensus = calculate_consensus(
            state, manager, engine
        )
        assert consensus.decision == ConsensusDecision.NO_TRADE

    def test_no_trade_risk_veto(self) -> None:
        state, mgr = create_run({"run_id": "nt-rv", "instrument": "BTC/USD"})
        _build_stage_pipeline(mgr)
        state, mgr, consensus = _consensus_state(state, mgr, agent_reports_fixture())
        for stage in [
            AnalysisStage.GENERATE_STRATEGY,
            AnalysisStage.EVALUATE_PORTFOLIO,
            AnalysisStage.APPLY_RISK_GATES,
        ]:
            mgr.transition(stage)
        state = state.model_copy(update={
            "consensus_report": {
                "decision": consensus.decision.value,
                "confidence": consensus.confidence,
                "agreements": consensus.agent_agreements,
                "disagreements": consensus.agent_disagreements,
                "vote_distribution": consensus.vote_distribution,
                "agent_weights": consensus.agent_weights,
            },
            "data_quality": {"overall_quality": 1.0},
            "risk_report": {"approved": False, "veto": True, "blocking_reasons": ["drawdown"]},
        })
        decision = publish_decision(state, mgr)
        assert decision.direction == FinalDecisionType.NO_TRADE_RISK

    def test_no_trade_portfolio(self) -> None:
        state, mgr = create_run({"run_id": "nt-pp", "instrument": "BTC/USD"})
        _build_stage_pipeline(mgr)
        state, mgr, consensus = _consensus_state(state, mgr, agent_reports_fixture())
        for stage in [
            AnalysisStage.GENERATE_STRATEGY,
            AnalysisStage.EVALUATE_PORTFOLIO,
            AnalysisStage.APPLY_RISK_GATES,
        ]:
            mgr.transition(stage)
        state = state.model_copy(update={
            "consensus_report": {
                "decision": consensus.decision.value,
                "confidence": consensus.confidence,
                "agreements": consensus.agent_agreements,
                "disagreements": consensus.agent_disagreements,
                "vote_distribution": consensus.vote_distribution,
                "agent_weights": consensus.agent_weights,
            },
            "data_quality": {"overall_quality": 1.0},
            "risk_report": {"approved": False, "veto": True, "blocking_reasons": ["drawdown"]},
        })
        decision = publish_decision(state, mgr)
        assert decision.direction == FinalDecisionType.NO_TRADE_RISK

    def test_no_trade_model_uncertainty(self) -> None:
        state, mgr = create_run({"run_id": "nt-mu", "instrument": "BTC/USD"})
        _build_stage_pipeline(mgr)
        state, mgr, _ = _consensus_state(state, mgr, agent_reports_fixture())
        for stage in [
            AnalysisStage.GENERATE_STRATEGY,
            AnalysisStage.EVALUATE_PORTFOLIO,
            AnalysisStage.APPLY_RISK_GATES,
        ]:
            mgr.transition(stage)
        state = state.model_copy(update={
            "consensus_report": {
                "decision": "NO_TRADE", "confidence": 0.0,
                "agreements": [], "disagreements": [],
                "vote_distribution": {},
                "agent_weights": {},
            },
            "data_quality": {"overall_quality": 1.0},
        })
        decision = publish_decision(state, mgr)
        assert decision.direction == FinalDecisionType.NO_TRADE


# ═══════════════════════════════════════════════════════════════════
#  TESTS: Seal immutability
# ═══════════════════════════════════════════════════════════════════


class TestSealImmutability:
    """Testet die Unveränderbarkeit des Seals."""

    def test_seal_hash_format(self) -> None:
        state, mgr = create_run({"run_id": "si-001", "instrument": "BTC/USD"})
        for stage in [
            AnalysisStage.BUILD_MARKET_SNAPSHOT,
            AnalysisStage.VALIDATE_DATA,
            AnalysisStage.COMPUTE_FEATURES,
            AnalysisStage.CLASSIFY_REGIME,
            AnalysisStage.RUN_FIRST_ROUND,
        ]:
            mgr.transition(stage)
        state, mgr = seal_first_round(state, mgr)
        # SHA256 hex = 64 chars
        assert len(state.seal_hash) == 64

    def test_seal_changes_with_data(self) -> None:
        state, mgr = create_run({"run_id": "si-002", "instrument": "BTC/USD"})
        for stage in [
            AnalysisStage.BUILD_MARKET_SNAPSHOT,
            AnalysisStage.VALIDATE_DATA,
            AnalysisStage.COMPUTE_FEATURES,
            AnalysisStage.CLASSIFY_REGIME,
            AnalysisStage.RUN_FIRST_ROUND,
        ]:
            mgr.transition(stage)
        state, mgr = seal_first_round(state, mgr)
        hash1 = state.seal_hash
        # Neue Stage mit gleicher Stage aber anderer Daten
        hash2 = mgr.seal(AnalysisStage.SEAL_FIRST_ROUND, "different_data")
        assert hash1 != hash2

    def test_seal_unchanged_with_same_data(self) -> None:
        state, mgr = create_run({"run_id": "si-003", "instrument": "BTC/USD"})
        for stage in [
            AnalysisStage.BUILD_MARKET_SNAPSHOT,
            AnalysisStage.VALIDATE_DATA,
            AnalysisStage.COMPUTE_FEATURES,
            AnalysisStage.CLASSIFY_REGIME,
            AnalysisStage.RUN_FIRST_ROUND,
        ]:
            mgr.transition(stage)
        state, mgr = seal_first_round(state, mgr)
        raw = f"{state.run_id}|{state.instrument}|{state.analysis_time.isoformat()}"
        assert mgr.seal(AnalysisStage.SEAL_FIRST_ROUND, raw) == state.seal_hash

    def test_verify_seal_positive(self) -> None:
        state, mgr = create_run({"run_id": "si-004", "instrument": "BTC/USD"})
        for stage in [
            AnalysisStage.BUILD_MARKET_SNAPSHOT,
            AnalysisStage.VALIDATE_DATA,
            AnalysisStage.COMPUTE_FEATURES,
            AnalysisStage.CLASSIFY_REGIME,
            AnalysisStage.RUN_FIRST_ROUND,
        ]:
            mgr.transition(stage)
        state, mgr = seal_first_round(state, mgr)
        raw = f"{state.run_id}|{state.instrument}|{state.analysis_time.isoformat()}"
        assert mgr.verify_seal(AnalysisStage.SEAL_FIRST_ROUND, raw, state.seal_hash) is True

    def test_verify_seal_negative_tampered_data(self) -> None:
        state, mgr = create_run({"run_id": "si-005", "instrument": "BTC/USD"})
        for stage in [
            AnalysisStage.BUILD_MARKET_SNAPSHOT,
            AnalysisStage.VALIDATE_DATA,
            AnalysisStage.COMPUTE_FEATURES,
            AnalysisStage.CLASSIFY_REGIME,
            AnalysisStage.RUN_FIRST_ROUND,
        ]:
            mgr.transition(stage)
        state, mgr = seal_first_round(state, mgr)
        assert mgr.verify_seal(
            AnalysisStage.SEAL_FIRST_ROUND, "tampered_data", state.seal_hash
        ) is False

    def test_verify_seal_negative_wrong_hash(self) -> None:
        state, mgr = create_run({"run_id": "si-006", "instrument": "BTC/USD"})
        for stage in [
            AnalysisStage.BUILD_MARKET_SNAPSHOT,
            AnalysisStage.VALIDATE_DATA,
            AnalysisStage.COMPUTE_FEATURES,
            AnalysisStage.CLASSIFY_REGIME,
            AnalysisStage.RUN_FIRST_ROUND,
        ]:
            mgr.transition(stage)
        state, mgr = seal_first_round(state, mgr)
        raw = f"{state.run_id}|{state.instrument}|{state.analysis_time.isoformat()}"
        assert mgr.verify_seal(
            AnalysisStage.SEAL_FIRST_ROUND, raw, "wrong_hash_000000000000000000000000000000000000000000000000000000000000000"
        ) is False


# ═══════════════════════════════════════════════════════════════════
#  TESTS: Edge cases
# ═══════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Testet Edge Cases und Randbedingungen."""

    def test_multiple_sequential_states(self, manager: StageManager) -> None:
        """Mehrere unabhaengige State-Übergänge."""
        for stage in AnalysisStage.default_sequence():
            manager.transition(stage)
        assert manager.stage_count() == 18

    def test_audit_events_are_frozen(self, manager: StageManager) -> None:
        manager.transition(AnalysisStage.CREATE_RUN)
        events = manager.audit_events
        # AuditEvents is frozenset → kein append
        with pytest.raises(AttributeError):
            events.append("tampered")

    def test_transition_function(self, manager: StageManager) -> None:
        result = transition(AnalysisStage.CREATE_RUN, manager)
        assert result.current_stage == AnalysisStage.CREATE_RUN

    def test_decision_has_all_fields(self) -> None:
        state, mgr = create_run({"run_id": "ef-001", "instrument": "BTC/USD"})
        _build_stage_pipeline(mgr)
        state, mgr, consensus = _consensus_state(state, mgr, agent_reports_fixture())
        for stage in [
            AnalysisStage.GENERATE_STRATEGY,
            AnalysisStage.EVALUATE_PORTFOLIO,
            AnalysisStage.APPLY_RISK_GATES,
        ]:
            mgr.transition(stage)
        state = state.model_copy(update={
            "consensus_report": {
                "decision": consensus.decision.value,
                "confidence": consensus.confidence,
                "agreements": consensus.agent_agreements,
                "disagreements": consensus.agent_disagreements,
                "vote_distribution": consensus.vote_distribution,
                "agent_weights": consensus.agent_weights,
            },
            "data_quality": {"overall_quality": 1.0},
        })
        decision = publish_decision(state, mgr)
        assert isinstance(decision, Decision)
        assert decision.run_id == "ef-001"
        assert decision.reasoning != ""


# ── Helper ────────────────────────────────────────────────────────


def agent_reports_fixture() -> list[AgentReport]:
    """Return sample AgentReports (avoids pytest fixture in non-test context)."""
    return [
        AgentReport(
            report_id="rpt-001",
            run_id="run-001",
            agent_id="indicator-agent",
            agent_version="1.0.0",
            instrument="BTC/USD",
            horizon="1h",
            as_of=datetime.now(UTC),
            hypothesis="Bullish momentum",
            probabilities={"up": 0.65, "down": 0.15, "range": 0.20},
            evidence=[EvidenceReference(
                reference="indicator:rsi",
                feature="rsi",
                value="30",
                direction="positive",
                relevance=0.8,
            )],
            raw_confidence=0.7,
            calibrated_confidence=0.68,
            status=AgentStatus.ACTIVE,
        ),
        AgentReport(
            report_id="rpt-002",
            run_id="run-001",
            agent_id="chart-agent",
            agent_version="1.0.0",
            instrument="BTC/USD",
            horizon="1h",
            as_of=datetime.now(UTC),
            hypothesis="Breakout pattern",
            probabilities={"up": 0.70, "down": 0.10, "range": 0.20},
            evidence=[EvidenceReference(
                reference="chart:structure",
                feature="structure",
                value="bull_flag",
                direction="positive",
                relevance=0.7,
            )],
            raw_confidence=0.75,
            calibrated_confidence=0.72,
            status=AgentStatus.ACTIVE,
        ),
    ]
