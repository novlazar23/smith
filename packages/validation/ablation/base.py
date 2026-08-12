"""Base class for ablation analysis methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class AblationResult:
    """Result of an ablation study.

    Attributes:
        method: Ablation method name.
        agent_id: Agent that was ablated (or "all" for baseline).
        full_score: Score with all agents/features.
        ablated_score: Score with this agent/feature removed.
        marginal_contribution: full_score - ablated_score (positive = helpful).
        direction: "positive", "negative", or "neutral".
        confidence: Confidence in this estimate.
    """

    method: str
    agent_id: str
    full_score: float
    ablated_score: float
    marginal_contribution: float
    direction: str = "positive"
    confidence: float = 1.0

    @property
    def is_helpful(self) -> bool:
        """Whether this agent contributes positively."""
        return self.marginal_contribution > 1e-9

    @property
    def is_harmful(self) -> bool:
        """Whether this agent hurts the ensemble."""
        return self.marginal_contribution < -1e-9

    @property
    def _effective_direction(self) -> str:
        """Internal direction used for counting (helpful/harmful/neutral)."""
        if self.is_helpful:
            return "helpful"
        elif self.is_harmful:
            return "harmful"
        return "neutral"


class AblationAnalyzer(ABC):
    """Abstract base for ablation analyzers.

    Subclasses implement `run()` which compares full model score
    against ablated score and returns an AblationResult.
    """

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Return the ablation method name."""

    @abstractmethod
    def run(
        self,
        full_score: float,
        ablated_score: float,
        agent_id: str,
        confidence: float = 1.0,
    ) -> AblationResult:
        """Run a single ablation comparison.

        Args:
            full_score: Score with all agents/features included.
            ablated_score: Score with one agent/feature removed.
            agent_id: Identifier of the ablated agent/feature.
            confidence: Confidence in the ablated score estimate.

        Returns:
            AblationResult with marginal contribution analysis.
        """
