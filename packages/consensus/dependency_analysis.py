"""Dependency analysis for agent consensus.

Detects correlated and redundant agents by comparing their evidence features
and feature_snapshot_id. Correlated agents receive reduced weights to avoid
double-counting the same information.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from packages.schemas.agent_report import AgentReport

from .base import WeightConfig


@dataclass(frozen=True)
class AgentDependency:
    """Dependency relationship between two agents.

    Attributes:
        agent_a: First agent ID.
        agent_b: Second agent ID.
        correlation: Correlation score in [0.0, 1.0].
        shared_features: Number of shared evidence features.
        total_features: Total unique features across both agents.
        reduced_agent: ID of the agent whose weight should be reduced.
    """

    agent_a: str
    agent_b: str
    correlation: float
    shared_features: int
    total_features: int
    reduced_agent: str


@dataclass
class DependencyAnalysisResult:
    """Result of a dependency analysis run.

    Attributes:
        dependencies: Detected correlated agent pairs.
        dependency_matrix: Pairwise correlation scores.
        adjusted_weights: Agent weights after dependency reduction.
    """

    dependencies: list[AgentDependency] = field(default_factory=list)
    dependency_matrix: dict[tuple[str, str], float] = field(
        default_factory=dict
    )
    adjusted_weights: dict[str, float] = field(default_factory=dict)


class DependencyAnalyzer:
    """Analyzes agent dependency and redundancy in consensus inputs.

    Compares agents' evidence features and feature_snapshot_id to detect
    correlated agents. When two agents share >70% evidence features they
    are treated as redundant and the later one receives a reduced weight.

    Attributes:
        config: Weight configuration for base weight computation.
        correlation_threshold: Shared feature ratio above which agents
            are considered correlated (default 0.7).
    """

    def __init__(
        self,
        config: WeightConfig | None = None,
        correlation_threshold: float = 0.7,
    ) -> None:
        self.config = config or WeightConfig()
        self.correlation_threshold = correlation_threshold

    def _compute_base_weight(self, report: AgentReport) -> float:
        """Compute the base weight for an agent report."""
        base = self.config.base_weight
        status = report.status.value
        multiplier = self.config.status_multiplier.get(status, 1.0)
        return base * multiplier

    def _extract_features(self, report: AgentReport) -> set[str]:
        """Extract a set of feature names from an agent's evidence."""
        features: set[str] = set()
        for ev in report.evidence:
            features.add(ev.feature)
        if report.feature_snapshot_id:
            features.add(f"snapshot:{report.feature_snapshot_id}")
        return features

    def _compute_correlation(
        self, features_a: set[str], features_b: set[str]
    ) -> tuple[float, int, int]:
        """Compute correlation between two feature sets.

        Returns:
            Tuple of (correlation_score, shared_count, total_count).
        """
        shared = features_a & features_b
        total = features_a | features_b
        if not total:
            return 0.0, 0, 0
        shared_count = len(shared)
        total_count = len(total)
        correlation = shared_count / total_count if total_count > 0 else 0.0
        return correlation, shared_count, total_count

    def analyze(
        self, reports: list[AgentReport]
    ) -> DependencyAnalysisResult:
        """Analyze dependency structure among a set of agent reports.

        Compares all pairs of agents, builds a correlation matrix, and
        computes adjusted weights that penalise redundant agents.

        Args:
            reports: Agent reports to analyse.

        Returns:
            DependencyAnalysisResult with correlations and adjusted weights.
        """
        if not reports:
            return DependencyAnalysisResult()

        # Compute base weights and feature sets
        weights: dict[str, float] = {}
        feature_sets: dict[str, set[str]] = {}
        for report in reports:
            weights[report.agent_id] = self._compute_base_weight(report)
            feature_sets[report.agent_id] = self._extract_features(report)

        # Build pairwise correlation matrix
        dependency_matrix: dict[tuple[str, str], float] = {}
        dependencies: list[AgentDependency] = []
        agent_ids = [r.agent_id for r in reports]

        for i, aid_a in enumerate(agent_ids):
            for j, aid_b in enumerate(agent_ids):
                if i >= j:
                    continue
                corr, shared, total = self._compute_correlation(
                    feature_sets[aid_a], feature_sets[aid_b]
                )
                key = (aid_a, aid_b)
                dependency_matrix[key] = corr

                if corr >= self.correlation_threshold:
                    # The later agent gets reduced weight
                    dependencies.append(
                        AgentDependency(
                            agent_a=aid_a,
                            agent_b=aid_b,
                            correlation=corr,
                            shared_features=shared,
                            total_features=total,
                            reduced_agent=aid_b,
                        )
                    )

        # Compute adjusted weights
        adjusted_weights = dict(weights)
        for dep in dependencies:
            current = adjusted_weights.get(dep.reduced_agent, 1.0)
            # Reduce by correlation factor
            adjusted_weights[dep.reduced_agent] = current * (
                1.0 - dep.correlation
            )

        return DependencyAnalysisResult(
            dependencies=dependencies,
            dependency_matrix=dependency_matrix,
            adjusted_weights=adjusted_weights,
        )

    def reduce_weights(
        self, reports: list[AgentReport]
    ) -> dict[str, float]:
        """Convenience wrapper: analyse and return only adjusted weights.

        Shadow agents (status == 'shadow') always get weight 0.0.

        Args:
            reports: Agent reports to process.

        Returns:
            Dict mapping agent_id to adjusted weight.
        """
        result = self.analyze(reports)

        # Shadow agents always get weight 0.0
        for report in reports:
            if report.status.value == "shadow":
                result.adjusted_weights[report.agent_id] = 0.0

        return result.adjusted_weights

    def get_correlation(self, agent_a: str, agent_b: str) -> float:
        """Return the pairwise correlation score for two agents.

        Args:
            agent_a: First agent ID.
            agent_b: Second agent ID.

        Returns:
            Correlation score, or 0.0 if not found.
        """
        return 0.0
