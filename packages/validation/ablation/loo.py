"""Leave-One-Out (LOO) Ablation.

Removes one agent at a time from the full ensemble and measures
the change in overall score (e.g. Brier score, accuracy).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .base import AblationAnalyzer, AblationResult


@dataclass(frozen=True)
class AgentEnsemble:
    """An agent's predictions in the ensemble.

    Attributes:
        agent_id: The agent's identifier.
        predictions: List of prediction dicts per sample.
        sample_ids: Corresponding sample IDs.
    """

    agent_id: str
    predictions: list[dict[str, float]]
    sample_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.predictions:
            raise ValueError("agent must have at least one prediction")
        n = len(self.predictions)
        if len(self.sample_ids) != n:
            object.__setattr__(self, "sample_ids", [f"s{i}" for i in range(n)])


class LeaveOneOutAblation(AblationAnalyzer):
    """Leave-one-out ablation for agent ensembles.

    For each agent in the ensemble:
    1. Score the full ensemble (all agents combined).
    2. Score the ensemble without that agent.
    3. marginal_contribution = full_score - ablated_score.

    Uses Brier score by default: lower is better.
    A POSITIVE marginal contribution means the agent HELPS (higher score without it).
    A NEGATIVE marginal contribution means the agent HURTS (lower score without it).

    Note: For Brier score, a higher score means worse performance, so
    a positive marginal (full > ablated) means removing the agent worsened performance
    -> the agent was HELPING. This is counterintuitive but correct for Brier score.
    """

    def __init__(
        self,
        score_fn: Callable[[list[dict[str, float]], list[str]], float] | None = None,
        higher_is_better: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        self.score_fn = score_fn or self._default_brier_score
        self.higher_is_better = higher_is_better

    @property
    def method_name(self) -> str:
        return "loo"

    @staticmethod
    def _default_brier_score(
        predictions: list[dict[str, float]], actuals: list[str]
    ) -> float:
        """Mean Brier score for multi-class. Lower is better.

        Brier score = mean of sum((p_class - actual_class)^2) per sample.
        """
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
        """Run a single LOO ablation comparison.

        For Brier score: marginal = ablated - full (positive = ablation helped).
        For accuracy-like: marginal = full - ablated (positive = agent helped).
        """
        if self.higher_is_better:
            marginal = full_score - ablated_score
        else:
            # For Brier score, lower is better
            marginal = ablated_score - full_score

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

    def run_ensemble(
        self,
        agents: Sequence[AgentEnsemble],
        actuals: list[str],
    ) -> list[AblationResult]:
        """Run LOO ablation on a full ensemble of agents.

        Args:
            agents: All agents in the ensemble.
            actuals: True outcomes for each sample.

        Returns:
            List of AblationResult, one per agent.
        """
        if not agents:
            return []

        n_samples = len(agents[0].predictions)

        def _aggregate(
            agents_list: Sequence[AgentEnsemble], idx: int
        ) -> dict[str, float]:
            agg: dict[str, float] = {"UP": 0.0, "DOWN": 0.0, "RANGE": 0.0}
            for agent in agents_list:
                if idx < len(agent.predictions):
                    for cls in ("UP", "DOWN", "RANGE"):
                        agg[cls] += agent.predictions[idx].get(cls, 0.0)
            for cls in agg:
                agg[cls] /= len(agents_list)
            return agg

        # Score full ensemble
        full_predictions = [_aggregate(agents, i) for i in range(n_samples)]
        full_score = self.score_fn(full_predictions, actuals)

        results: list[AblationResult] = []
        for agent in agents:
            remaining = [a for a in agents if a.agent_id != agent.agent_id]
            if not remaining:
                continue

            ablated_predictions = [
                _aggregate(remaining, i) for i in range(n_samples)
            ]
            ablated_score = self.score_fn(ablated_predictions, actuals)

            results.append(self.run(full_score, ablated_score, agent.agent_id))

        return results
