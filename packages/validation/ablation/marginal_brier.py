"""Marginal Brier Score Analysis.

Measures each agent's individual Brier score contribution
compared to the ensemble baseline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .base import AblationAnalyzer, AblationResult


@dataclass(frozen=True)
class AgentPredictions:
    """Predictions from a single agent.

    Attributes:
        agent_id: The agent's identifier.
        predictions: List of probability dicts per sample.
        actuals: True outcomes (set externally).
    """

    agent_id: str
    predictions: list[dict[str, float]]
    actuals: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.predictions:
            raise ValueError("predictions must not be empty")


class MarginalBrierAnalyzer(AblationAnalyzer):
    """Marginal Brier score analysis.

    Computes:
    - Individual Brier score for each agent
    - Ensemble Brier score (average of all agents)
    - Marginal = agent_brier - ensemble_brier

    Positive marginal means the agent is worse than ensemble average.
    Negative marginal means the agent helps improve ensemble.
    """

    @property
    def method_name(self) -> str:
        return "marginal_brier"

    @staticmethod
    def _brier_score(
        predictions: list[dict[str, float]], actuals: list[str]
    ) -> float:
        """Compute mean Brier score for multi-class. Lower is better."""
        total = 0.0
        n = len(predictions)
        if n == 0:
            return 0.0
        for pred, actual in zip(predictions, actuals, strict=True):
            score = 0.0
            for cls in ("UP", "DOWN", "RANGE"):
                predicted = pred.get(cls, 0.0)
                expected = 1.0 if actual == cls else 0.0
                score += (predicted - expected) ** 2
            total += score
        return total / n

    def run(
        self,
        full_score: float,
        ablated_score: float,
        agent_id: str,
        confidence: float = 1.0,
    ) -> AblationResult:
        """Run a single marginal Brier comparison.

        full_score = ensemble Brier
        ablated_score = agent Brier (this agent is the "ablation" of interest)

        For Brier score, lower is better.
        marginal = full_score - ablated_score
        Positive = agent is better than ensemble (helpful)
        Negative = agent is worse than ensemble (harmful)
        """
        marginal = full_score - ablated_score

        if marginal > 1e-9:
            direction = "helpful"
        elif marginal < -1e-9:
            direction = "harmful"
        else:
            direction = "neutral"

        return AblationResult(
            method=self.method_name,
            agent_id=agent_id,
            full_score=full_score,
            ablated_score=ablated_score,
            marginal_contribution=marginal,
            direction=direction,
            confidence=confidence,
        )

    def analyze(
        self,
        agents: Sequence[AgentPredictions],
    ) -> list[AblationResult]:
        """Analyze marginal Brier scores for all agents.

        Args:
            agents: All agents with predictions and shared actuals.

        Returns:
            List of AblationResult sorted by marginal contribution (most helpful first).
        """
        if not agents:
            return []

        n_samples = len(agents[0].predictions)

        def _ensemble_pred(idx: int) -> dict[str, float]:
            agg: dict[str, float] = {"UP": 0.0, "DOWN": 0.0, "RANGE": 0.0}
            for agent in agents:
                if idx < len(agent.predictions):
                    for cls in ("UP", "DOWN", "RANGE"):
                        agg[cls] += agent.predictions[idx].get(cls, 0.0)
            for cls in agg:
                agg[cls] /= len(agents)
            return agg

        actuals = agents[0].actuals
        ensemble_predictions = [_ensemble_pred(i) for i in range(n_samples)]
        ensemble_brier = self._brier_score(ensemble_predictions, actuals)

        results: list[AblationResult] = []
        for agent in agents:
            agent_brier = self._brier_score(agent.predictions, actuals)
            results.append(self.run(ensemble_brier, agent_brier, agent.agent_id))

        # Sort by marginal contribution (most helpful/negative first)
        results.sort(key=lambda r: r.marginal_contribution)

        return results
