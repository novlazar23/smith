"""Tests für Governance/Decision-Engine."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from packages.governance.decision_engine import (
    BlockingRules,
    DecisionEngine,
    DecisionRule,
    FinalDecisionData,
    FinalDecisionType,
    GovernanceConfig,
)

# ---------------------------------------------------------------------------
# Base types
# ---------------------------------------------------------------------------

class TestFinalDecisionType:
    """Testet FinalDecisionType-Enum."""

    def test_all_types_present(self) -> None:
        assert FinalDecisionType.LONG_BIAS == "LONG_BIAS"
        assert FinalDecisionType.SHORT_BIAS == "SHORT_BIAS"
        assert FinalDecisionType.RANGE == "RANGE"
        assert FinalDecisionType.NO_TRADE == "NO_TRADE"
        assert FinalDecisionType.NO_TRADE_DATA_QUALITY == "NO_TRADE_DATA_QUALITY"
        assert FinalDecisionType.NO_TRADE_INSUFFICIENT_EDGE == "NO_TRADE_INSUFFICIENT_EDGE"
        assert FinalDecisionType.NO_TRADE_PORTFOLIO == "NO_TRADE_PORTFOLIO"
        assert FinalDecisionType.NO_TRADE_RISK == "NO_TRADE_RISK"
        assert FinalDecisionType.NO_TRADE_MODEL_UNCERTAINTY == "NO_TRADE_MODEL_UNCERTAINTY"


class TestDecisionRule:
    """Testet DecisionRule-Datenklasse."""

    def test_creation(self) -> None:
        rule = DecisionRule(
            rule_id="r1",
            condition="consensus_long_above_threshold",
            action="approve",
            threshold=0.65,
            blocking=False,
        )
        assert rule.rule_id == "r1"
        assert rule.blocking is False
        assert rule.threshold == 0.65


class TestGovernanceConfig:
    """Testet GovernanceConfig-Standardwerte."""

    def test_defaults(self) -> None:
        config = GovernanceConfig()
        assert config.consensus_long_threshold == 0.65
        assert config.consensus_short_threshold == 0.65
        assert config.consensus_range_threshold == 0.50
        assert config.min_confidence == 0.50
        assert config.max_uncertainty == 0.30
        assert config.required_agents == 2
        assert config.rules == []

    def test_custom_values(self) -> None:
        config = GovernanceConfig(
            consensus_long_threshold=0.70,
            min_confidence=0.60,
            required_agents=3,
        )
        assert config.consensus_long_threshold == 0.70
        assert config.min_confidence == 0.60
        assert config.required_agents == 3


# ---------------------------------------------------------------------------
# Engine — hard blocks
# ---------------------------------------------------------------------------

def _sample_params(**overrides: Any) -> dict[str, Any]:
    """Erstellt ein Standard-Parameter-Wörterbuch für Engine-Tests."""
    base: dict[str, Any] = {
        "consensus_decision": "LONG_BIAS",
        "consensus_confidence": 0.80,
        "consensus_vote_distribution": {"up": 0.7, "down": 0.2, "range": 0.1},
        "risk_approved": True,
        "risk_veto": False,
        "risk_reduction_factor": 1.0,
        "risk_blocking_reasons": [],
        "num_active_agents": 3,
        "instrument": "AAPL",
        "run_id": "test-run",
    }
    base.update(overrides)
    return base


class TestEngineHardBlocks:
    """Testet harte Blockier-Bedingungen der Engine."""

    def test_risk_veto_blocks(self) -> None:
        engine = DecisionEngine()
        params = _sample_params(risk_veto=True, risk_blocking_reasons=["VaR exceeded"])
        result = engine.evaluate(**params)
        assert result.decision == FinalDecisionType.NO_TRADE_RISK
        assert "risk_veto" in result.blocking_reasons[0]

    def test_risk_not_approved_blocks(self) -> None:
        engine = DecisionEngine()
        params = _sample_params(risk_approved=False, risk_blocking_reasons=["Drawdown limit"])
        result = engine.evaluate(**params)
        assert result.decision == FinalDecisionType.NO_TRADE_RISK
        assert "risk_not_approved" in result.blocking_reasons[0]

    def test_insufficient_agents_blocks(self) -> None:
        engine = DecisionEngine()
        params = _sample_params(num_active_agents=1)
        result = engine.evaluate(**params)
        assert result.decision == FinalDecisionType.NO_TRADE_DATA_QUALITY
        assert "insufficient_agents" in result.blocking_reasons[0]


# ---------------------------------------------------------------------------
# Engine — soft blocks
# ---------------------------------------------------------------------------

class TestEngineSoftBlocks:
    """Testet weiche Blockier-Bedingungen der Engine."""

    def test_low_confidence_blocks(self) -> None:
        engine = DecisionEngine()
        params = _sample_params(consensus_confidence=0.30)
        result = engine.evaluate(**params)
        assert result.decision == FinalDecisionType.NO_TRADE_MODEL_UNCERTAINTY
        assert "low_confidence" in result.blocking_reasons[0]

    def test_high_uncertainty_blocks(self) -> None:
        engine = DecisionEngine()
        params = _sample_params(consensus_confidence=0.80, max_uncertainty=0.50)
        result = engine.evaluate(**params)
        assert result.decision == FinalDecisionType.NO_TRADE_MODEL_UNCERTAINTY
        assert "high_uncertainty" in result.blocking_reasons[0]

    def test_insufficient_edge_blocks(self) -> None:
        engine = DecisionEngine()
        params = _sample_params(consensus_confidence=0.50)
        result = engine.evaluate(**params)
        assert result.decision == FinalDecisionType.NO_TRADE_INSUFFICIENT_EDGE
        assert "INSUFFICIENT_EDGE" in str(result.decision)


class TestEngineApproval:
    """Testet genehmigte Entscheidungen."""

    def test_long_bias_approved(self) -> None:
        engine = DecisionEngine()
        params = _sample_params(consensus_decision="LONG_BIAS", consensus_confidence=0.80)
        result = engine.evaluate(**params)
        assert result.decision == FinalDecisionType.LONG_BIAS
        assert "Long bias approved" in result.reason

    def test_short_bias_approved(self) -> None:
        engine = DecisionEngine()
        params = _sample_params(consensus_decision="SHORT_BIAS", consensus_confidence=0.80)
        result = engine.evaluate(**params)
        assert result.decision == FinalDecisionType.SHORT_BIAS
        assert "Short bias approved" in result.reason

    def test_range_approved(self) -> None:
        engine = DecisionEngine(config=GovernanceConfig(consensus_range_threshold=0.50))
        params = _sample_params(
            consensus_decision="RANGE",
            consensus_confidence=0.60,
            consensus_vote_distribution={"range": 0.6, "up": 0.2, "down": 0.2},
        )
        result = engine.evaluate(**params)
        assert result.decision == FinalDecisionType.RANGE
        assert "Range bias approved" in result.reason

    def test_all_defaults_pass_long(self) -> None:
        """Standardkonfiguration mit Long-Bias und 0.80 Confidence sollte genehmigen."""
        engine = DecisionEngine()
        params = _sample_params(consensus_decision="LONG_BIAS", consensus_confidence=0.80)
        result = engine.evaluate(**params)
        assert result.decision == FinalDecisionType.LONG_BIAS


# ---------------------------------------------------------------------------
# Custom rules
# ---------------------------------------------------------------------------

class TestCustomRules:
    """Testet benutzerdefinierte Regeln."""

    def test_custom_rule_blocking(self) -> None:
        rule = DecisionRule(
            rule_id="custom_block",
            condition="consensus_long_above_threshold",
            action="block",
            threshold=0.0,
            blocking=True,
        )
        engine = DecisionEngine(config=GovernanceConfig(rules=[rule]))
        params = _sample_params(consensus_decision="LONG_BIAS", consensus_confidence=0.80)
        result, rule_hits = engine.evaluate_with_rules(**params)
        assert "custom_block" in rule_hits
        assert any("custom_block" in br for br in result.blocking_reasons)

    def test_rules_returned_as_tuple(self) -> None:
        rule = DecisionRule(
            rule_id="long_rule",
            condition="consensus_long_above_threshold",
            action="approve",
            threshold=0.65,
            blocking=False,
        )
        engine = DecisionEngine(config=GovernanceConfig(rules=[rule]))
        params = _sample_params(consensus_decision="LONG_BIAS", consensus_confidence=0.80)
        result, rule_hits = engine.evaluate_with_rules(**params)
        assert isinstance(result, FinalDecisionData)
        assert isinstance(rule_hits, list)
        assert "long_rule" in rule_hits


# ---------------------------------------------------------------------------
# BlockingRules
# ---------------------------------------------------------------------------

class TestBlockingRules:
    """Testet die BlockingRules-Klasse."""

    def test_empty_on_approval(self) -> None:
        rules = BlockingRules()
        blockers = rules.check_blocking_conditions(
            consensus_decision="LONG_BIAS",
            consensus_confidence=0.80,
            risk_approved=True,
            risk_veto=False,
            risk_blocking_reasons=[],
            num_active_agents=3,
        )
        assert blockers == []

    def test_risk_veto_blocker(self) -> None:
        rules = BlockingRules()
        blockers = rules.check_blocking_conditions(
            consensus_decision="LONG_BIAS",
            consensus_confidence=0.80,
            risk_approved=True,
            risk_veto=True,
            risk_blocking_reasons=["VaR breached"],
            num_active_agents=3,
        )
        assert any("risk_veto" in b for b in blockers)

    def test_priority_order(self) -> None:
        """Hard blocks sollten vor soft blocks kommen."""
        rules = BlockingRules()
        blockers = rules.check_blocking_conditions(
            consensus_decision="LONG_BIAS",
            consensus_confidence=0.40,  # low confidence
            risk_approved=False,
            risk_veto=False,
            risk_blocking_reasons=[],
            num_active_agents=1,  # also insufficient
        )
        priority = rules.get_priority_blockers(blockers)
        # risk_not_approved (hard) before low_confidence (soft)
        hard_idx = next(i for i, b in enumerate(priority) if "risk_not_approved" in b)
        soft_idx = next(i for i, b in enumerate(priority) if "low_confidence" in b)
        assert hard_idx < soft_idx

    def test_multiple_blockers(self) -> None:
        rules = BlockingRules()
        blockers = rules.check_blocking_conditions(
            consensus_decision="LONG_BIAS",
            consensus_confidence=0.30,  # low confidence
            risk_approved=False,
            risk_veto=False,
            risk_blocking_reasons=[],
            num_active_agents=1,  # insufficient
            max_uncertainty=0.50,  # high uncertainty
        )
        assert len(blockers) >= 3
        assert any("risk_not_approved" in b for b in blockers)
        assert any("insufficient_agents" in b for b in blockers)
        assert any("low_confidence" in b for b in blockers)

    def test_get_priority_blockers_hard_first(self) -> None:
        rules = BlockingRules()
        reasons = [
            "low_confidence: 0.30 < 0.50",
            "risk_not_approved",
            "insufficient_agents: 1 < 2",
        ]
        priority = rules.get_priority_blockers(reasons)
        assert priority[0] == "risk_not_approved"
        assert priority[1] == "insufficient_agents: 1 < 2"
        assert priority[2] == "low_confidence: 0.30 < 0.50"


# ---------------------------------------------------------------------------
# Decision details
# ---------------------------------------------------------------------------

class TestDecisionDetails:
    """Testet get_decision_details."""

    def test_details_has_expected_keys(self) -> None:
        engine = DecisionEngine()
        params = _sample_params(consensus_decision="LONG_BIAS", consensus_confidence=0.80)
        result = engine.evaluate(**params)
        details = engine.get_decision_details(result)
        assert "decision" in details
        assert "reason" in details
        assert "blocking_reasons" in details
        assert "confidence" in details
        assert "timestamp" in details
        assert details["decision"] == FinalDecisionType.LONG_BIAS


# ---------------------------------------------------------------------------
# Audit hash
# ---------------------------------------------------------------------------

class TestAuditHash:
    """Testet audit_hash-Computierung."""

    def test_hash_computed(self) -> None:
        data = FinalDecisionData(
            run_id="r1",
            instrument="AAPL",
            horizons=["1D", "1W"],
            analysis_time=datetime.now(UTC),
            decision=FinalDecisionType.LONG_BIAS,
            reason="Test reason",
        )
        h = data.compute_hash()
        assert h is not None
        assert len(h) == 64  # SHA-256 hex length
        assert data.audit_hash == h

    def test_hash_deterministic(self) -> None:
        data1 = FinalDecisionData(
            run_id="r1",
            instrument="AAPL",
            horizons=["1D"],
            analysis_time=datetime.now(UTC),
            decision=FinalDecisionType.LONG_BIAS,
            reason="Reason",
        )
        data2 = FinalDecisionData(
            run_id="r1",
            instrument="AAPL",
            horizons=["1D"],
            analysis_time=datetime.now(UTC),
            decision=FinalDecisionType.LONG_BIAS,
            reason="Reason",
        )
        h1 = data1.compute_hash()
        h2 = data2.compute_hash()
        assert h1 == h2
