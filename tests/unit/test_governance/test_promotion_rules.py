"""Tests für Promotion Rules — EPIC-11."""

from __future__ import annotations

from packages.governance.promotion_rules import (
    ActiveAgent,
    AgentState,
    DegradedAgent,
    PromotionCriteria,
    PromotionRuleEngine,
    ShadowAgent,
)


class TestAgentState:
    """Testet AgentState aus promotion_rules."""

    def test_shadow_weight(self) -> None:
        agent = ShadowAgent(agent_id="s1", version="1.0")
        assert agent.consensus_weight == 0.0

    def test_active_weight(self) -> None:
        agent = ActiveAgent(agent_id="a1", version="1.0")
        assert agent.consensus_weight == 1.0
        assert not agent.is_degraded

    def test_degraded_weight(self) -> None:
        agent = DegradedAgent(agent_id="d1", version="1.0")
        assert agent.consensus_weight == 0.5
        assert agent.is_degraded

    def test_degraded_state_set(self) -> None:
        agent = DegradedAgent(agent_id="d1", version="1.0")
        assert agent.state == AgentState.DEGRADED


class TestPromotionCriteria:
    """Testet PromotionCriteria-Standardwerte."""

    def test_defaults(self) -> None:
        c = PromotionCriteria()
        assert c.min_oos_score == 0.60
        assert c.min_calibration_score == 0.70
        assert c.min_marginal_contribution == 0.01
        assert c.min_shadow_days == 5
        assert c.required_reviews == 1


class TestPromotionRuleEngine:
    """Testet PromotionRuleEngine."""

    def _shadow_agent(
        self,
        oos: float = 0.70,
        calibration: float = 0.75,
        marginal: float = 0.02,
    ) -> ShadowAgent:
        return ShadowAgent(
            agent_id="test-agent",
            version="1.0",
            oos_score=oos,
            calibration_score=calibration,
            marginal_contribution=marginal,
        )

    def test_promotion_all_criteria_met(self) -> None:
        engine = PromotionRuleEngine()
        agent = self._shadow_agent()
        can_promote, reasons = engine.evaluate_promotion(agent, "approved")
        assert can_promote is True
        assert any("✅" in r for r in reasons)

    def test_promotion_oos_below_threshold(self) -> None:
        engine = PromotionRuleEngine()
        agent = self._shadow_agent(oos=0.50)
        can_promote, reasons = engine.evaluate_promotion(agent, "approved")
        assert can_promote is False
        assert any("oos_score" in r and "✅" not in r for r in reasons)

    def test_promotion_calibration_below_threshold(self) -> None:
        engine = PromotionRuleEngine()
        agent = self._shadow_agent(calibration=0.60)
        can_promote, reasons = engine.evaluate_promotion(agent, "approved")
        assert can_promote is False
        assert any("calibration_score" in r and "✅" not in r for r in reasons)

    def test_promotion_marginal_below_threshold(self) -> None:
        engine = PromotionRuleEngine()
        agent = self._shadow_agent(marginal=0.005)
        can_promote, reasons = engine.evaluate_promotion(agent, "approved")
        assert can_promote is False
        assert any("marginal_contribution" in r and "✅" not in r for r in reasons)

    def test_promotion_review_not_approved(self) -> None:
        engine = PromotionRuleEngine()
        agent = self._shadow_agent()
        can_promote, reasons = engine.evaluate_promotion(agent, "rejected")
        assert can_promote is False

    def test_promotion_review_pending(self) -> None:
        engine = PromotionRuleEngine()
        agent = self._shadow_agent()
        can_promote, reasons = engine.evaluate_promotion(agent, "pending")
        assert can_promote is False

    def test_promotion_all_criteria_fail(self) -> None:
        engine = PromotionRuleEngine()
        agent = self._shadow_agent(oos=0.40, calibration=0.50, marginal=0.001)
        can_promote, reasons = engine.evaluate_promotion(agent, "pending")
        assert can_promote is False
        failed = [r for r in reasons if "✅" not in r]
        assert len(failed) == 4

    def test_promotion_custom_criteria(self) -> None:
        criteria = PromotionCriteria(
            min_oos_score=0.50,
            min_calibration_score=0.60,
        )
        engine = PromotionRuleEngine(criteria)
        agent = self._shadow_agent(oos=0.55, calibration=0.65)
        can_promote, reasons = engine.evaluate_promotion(agent, "approved")
        assert can_promote is True

    def test_promotion_returns_tuple(self) -> None:
        engine = PromotionRuleEngine()
        agent = self._shadow_agent()
        result = engine.evaluate_promotion(agent, "approved")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], list)

    def test_promotion_reasons_include_passed(self) -> None:
        engine = PromotionRuleEngine()
        agent = self._shadow_agent()
        can_promote, reasons = engine.evaluate_promotion(agent, "approved")
        passed = [r for r in reasons if "✅" in r]
        assert len(passed) == 4

    def test_promotion_reasons_include_failed(self) -> None:
        engine = PromotionRuleEngine()
        agent = self._shadow_agent(oos=0.40)
        can_promote, reasons = engine.evaluate_promotion(agent, "approved")
        failed = [r for r in reasons if "✅" not in r and r.strip()]
        assert len(failed) >= 1

    def test_promotion_oos_exactly_at_threshold(self) -> None:
        engine = PromotionRuleEngine()
        agent = self._shadow_agent(oos=0.60)  # exactly at default threshold
        can_promote, reasons = engine.evaluate_promotion(agent, "approved")
        assert can_promote is True

    def test_promotion_calibration_exactly_at_threshold(self) -> None:
        engine = PromotionRuleEngine()
        agent = self._shadow_agent(calibration=0.70)  # exactly at default threshold
        can_promote, reasons = engine.evaluate_promotion(agent, "approved")
        assert can_promote is True

    def test_promotion_marginal_exactly_at_threshold(self) -> None:
        engine = PromotionRuleEngine()
        agent = self._shadow_agent(marginal=0.01)  # exactly at default threshold
        can_promote, reasons = engine.evaluate_promotion(agent, "approved")
        assert can_promote is True

    def test_promotion_reasons_format_includes_values(self) -> None:
        engine = PromotionRuleEngine()
        agent = self._shadow_agent(oos=0.50)
        _, reasons = engine.evaluate_promotion(agent, "approved")
        failing = [r for r in reasons if "✅" not in r]
        assert any("0.50" in r for r in failing)

    def test_promotion_empty_reasons_list_on_failure(self) -> None:
        engine = PromotionRuleEngine()
        agent = self._shadow_agent()
        can_promote, reasons = engine.evaluate_promotion(agent, "rejected")
        assert can_promote is False
        assert len(reasons) > 0

    def test_promotion_custom_min_shadow_days_no_effect(self) -> None:
        """min_shadow_days ist in PromotionCriteria aber wird nicht in evaluate_promotion geprüft."""
        criteria = PromotionCriteria(min_shadow_days=20)
        engine = PromotionRuleEngine(criteria)
        agent = self._shadow_agent()
        can_promote, reasons = engine.evaluate_promotion(agent, "approved")
        assert can_promote is True

    def test_shadow_agent_has_started_at(self) -> None:
        engine = PromotionRuleEngine()
        agent = self._shadow_agent()
        assert agent.started_at.tzinfo is not None
