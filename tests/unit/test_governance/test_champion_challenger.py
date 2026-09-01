"""Tests für Champion-Challenger Engine — EPIC-11."""

from __future__ import annotations

from datetime import UTC, datetime

from packages.governance.champion_challenger import (
    AgentVersion,
    ChampionChallengerConfig,
    ChampionChallengerEngine,
    EvaluationResult,
)


class TestAgentVersion:
    """Testet AgentVersion-Felder."""

    def test_defaults(self) -> None:
        av = AgentVersion(agent_id="a1", version="v1")
        assert av.oos_score == 0.0
        assert av.calibration_score == 0.0
        assert av.stability_score == 1.0
        assert av.marginal_contribution == 0.0
        assert av.shadow_days == 0
        assert av.samples == 0

    def test_is_champion_true(self) -> None:
        av = AgentVersion(agent_id="a1", version="champion")
        assert av.is_champion is True
        assert av.is_challenger is False

    def test_is_challenger_true(self) -> None:
        av = AgentVersion(agent_id="a1", version="challenger")
        assert av.is_challenger is True
        assert av.is_champion is False

    def test_neither_champion_nor_challenger(self) -> None:
        av = AgentVersion(agent_id="a1", version="v1")
        assert av.is_champion is False
        assert av.is_challenger is False

    def test_created_at_present(self) -> None:
        av = AgentVersion(agent_id="a1", version="v1")
        assert av.created_at is not None


class TestEvaluationResult:
    """Testet EvaluationResult-Felder."""

    def test_defaults(self) -> None:
        now = datetime.now(UTC)
        er = EvaluationResult(
            agent_id="a1",
            champion_version="v1",
            challenger_version="v2",
            evaluation_window_start=now,
            evaluation_window_end=now,
            champion_oos=0.70,
            challenger_oos=0.75,
            champion_stability=0.95,
            challenger_stability=0.94,
            champion_marginal=0.01,
            challenger_marginal=0.02,
        )
        assert er.champion_new_risks == []
        assert er.champion_new_risks_empty is True
        assert er.shadow_success is True
        assert er.promoted is False
        assert er.promotion_reason == ""


class TestChampionChallengerConfig:
    """Testet ChampionChallengerConfig-Standardwerte."""

    def test_defaults(self) -> None:
        c = ChampionChallengerConfig()
        assert c.min_oos_improvement == 0.02
        assert c.max_stability_degradation == 0.05
        assert c.shadow_evaluation_days == 10
        assert c.min_evaluation_samples == 50


class TestChampionChallengerEngine:
    """Testet Champion-Challenger-Promotion-Logik."""

    def _champion(self, **overrides: object) -> AgentVersion:
        defaults = {
            "agent_id": "a1",
            "version": "champion",
            "oos_score": 0.70,
            "calibration_score": 0.80,
            "stability_score": 0.95,
            "marginal_contribution": 0.01,
            "shadow_days": 10,
            "samples": 100,
        }
        defaults.update(overrides)
        return AgentVersion(**defaults)

    def _challenger(self, **overrides: object) -> AgentVersion:
        defaults = {
            "agent_id": "a1",
            "version": "challenger",
            "oos_score": 0.72,
            "calibration_score": 0.81,
            "stability_score": 0.94,
            "marginal_contribution": 0.02,
            "shadow_days": 10,
            "samples": 100,
        }
        defaults.update(overrides)
        return AgentVersion(**defaults)

    # --- Promoted: all criteria met ---

    def test_promotion_all_criteria_met(self) -> None:
        engine = ChampionChallengerEngine()
        champion = self._champion()
        challenger = self._challenger()
        result = engine.evaluate("a1", champion, challenger)
        assert result.promoted is True
        assert result.champion_new_risks == []
        assert "OOS improved" in result.promotion_reason

    def test_promotion_no_stability_degradation_ok(self) -> None:
        engine = ChampionChallengerEngine()
        challenger = self._challenger(stability_score=0.96)  # better stability
        result = engine.evaluate("a1", self._champion(), challenger)
        assert result.promoted is True

    # --- Not promoted: OOS improvement insufficient ---

    def test_no_promotion_oos_below_threshold(self) -> None:
        engine = ChampionChallengerEngine()
        challenger = self._challenger(oos_score=0.71)  # only +0.01, need >=0.02
        result = engine.evaluate("a1", self._champion(), challenger)
        assert result.promoted is False
        assert "OOS improvement" in result.promotion_reason

    def test_no_promotion_oos_lower(self) -> None:
        engine = ChampionChallengerEngine()
        challenger = self._challenger(oos_score=0.65)
        result = engine.evaluate("a1", self._champion(), challenger)
        assert result.promoted is False

    # --- Not promoted: stability degradation too high ---

    def test_no_promotion_stability_too_degraded(self) -> None:
        engine = ChampionChallengerEngine()
        challenger = self._challenger(stability_score=0.88)  # 0.07 degradation > 0.05
        result = engine.evaluate("a1", self._champion(), challenger)
        assert result.promoted is False
        assert "Stability degradation" in result.promotion_reason

    def test_promotion_stability_degradation_ok(self) -> None:
        engine = ChampionChallengerEngine()
        challenger = self._challenger(stability_score=0.91)  # 0.04 ≤ 0.05
        result = engine.evaluate("a1", self._champion(), challenger)
        assert result.promoted is True

    # --- Not promoted: marginal contribution zero/negative ---

    def test_no_promotion_zero_marginal(self) -> None:
        engine = ChampionChallengerEngine()
        challenger = self._challenger(marginal_contribution=0.0)
        result = engine.evaluate("a1", self._champion(), challenger)
        assert result.promoted is False
        assert "Marginal contribution" in result.promotion_reason

    def test_no_promotion_negative_marginal(self) -> None:
        engine = ChampionChallengerEngine()
        challenger = self._challenger(marginal_contribution=-0.01)
        result = engine.evaluate("a1", self._champion(), challenger)
        assert result.promoted is False

    # --- Not promoted: new risks ---

    def test_no_promotion_new_risks(self) -> None:
        engine = ChampionChallengerEngine()
        result = engine.evaluate(
            "a1",
            self._champion(),
            self._challenger(),
            champion_new_risks=["risk_a", "risk_b"],
        )
        assert result.promoted is False
        assert "New risks" in result.promotion_reason
        assert "risk_a" in result.promotion_reason

    def test_promotion_no_new_risks(self) -> None:
        engine = ChampionChallengerEngine()
        result = engine.evaluate(
            "a1",
            self._champion(),
            self._challenger(),
            champion_new_risks=[],
        )
        assert result.promoted is True
        assert result.champion_new_risks_empty is True

    def test_result_contains_risks(self) -> None:
        engine = ChampionChallengerEngine()
        result = engine.evaluate(
            "a1",
            self._champion(),
            self._challenger(),
            champion_new_risks=["risk_x"],
        )
        assert "risk_x" in result.champion_new_risks
        assert result.champion_new_risks_empty is False

    # --- Not promoted: shadow failure ---

    def test_no_promotion_shadow_failed(self) -> None:
        engine = ChampionChallengerEngine()
        result = engine.evaluate(
            "a1",
            self._champion(),
            self._challenger(),
            shadow_success=False,
        )
        assert result.promoted is False
        assert "Shadow operation not successful" in result.promotion_reason

    def test_promotion_shadow_success(self) -> None:
        engine = ChampionChallengerEngine()
        result = engine.evaluate(
            "a1",
            self._champion(),
            self._challenger(),
            shadow_success=True,
        )
        assert result.promoted is True

    # --- Return type structure ---

    def test_evaluation_returns_evaluation_result(self) -> None:
        engine = ChampionChallengerEngine()
        result = engine.evaluate("a1", self._champion(), self._challenger())
        assert isinstance(result, EvaluationResult)

    def test_result_contains_all_metrics(self) -> None:
        engine = ChampionChallengerEngine()
        result = engine.evaluate("a1", self._champion(), self._challenger())
        assert result.champion_oos == 0.70
        assert result.challenger_oos == 0.72
        assert result.champion_stability == 0.95
        assert result.challenger_stability == 0.94
        assert result.champion_marginal == 0.01
        assert result.challenger_marginal == 0.02

    def test_result_agent_id_matches(self) -> None:
        engine = ChampionChallengerEngine()
        result = engine.evaluate("my-agent", self._champion(), self._challenger())
        assert result.agent_id == "my-agent"

    def test_result_champion_version(self) -> None:
        engine = ChampionChallengerEngine()
        champion = self._champion(version="v1.0")
        challenger = self._challenger(version="v1.1")
        result = engine.evaluate("a1", champion, challenger)
        assert result.champion_version == "v1.0"
        assert result.challenger_version == "v1.1"

    # --- Custom config ---

    def test_custom_oos_threshold(self) -> None:
        config = ChampionChallengerConfig(min_oos_improvement=0.01)
        engine = ChampionChallengerEngine(config)
        challenger = self._challenger(oos_score=0.705)  # +0.005, below default 0.02
        result = engine.evaluate("a1", self._champion(), challenger)
        assert result.promoted is False  # 0.005 < 0.01

    def test_custom_stability_threshold(self) -> None:
        config = ChampionChallengerConfig(max_stability_degradation=0.10)
        engine = ChampionChallengerEngine(config)
        challenger = self._challenger(stability_score=0.83)  # 0.12 degradation
        # 0.12 > 0.10 → still fails
        result = engine.evaluate("a1", self._champion(), challenger)
        assert result.promoted is False

    def test_evaluate_with_none_risks(self) -> None:
        engine = ChampionChallengerEngine()
        result = engine.evaluate(
            "a1",
            self._champion(),
            self._challenger(),
            champion_new_risks=None,
        )
        assert result.promoted is True
        assert result.champion_new_risks_empty is True
