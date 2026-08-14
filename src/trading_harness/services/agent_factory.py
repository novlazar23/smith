from __future__ import annotations

import copy
import random
from datetime import UTC, datetime
from uuid import uuid4

from trading_harness.models import (
    AgentGenome,
    AgentStatus,
    GenomeMutation,
    MutationType,
)

STRATEGIES = {
    "technical": [
        "RSI",
        "MACD",
        "Bollinger Bands",
        "VWAP",
        "EMA Cross",
        "Stochastic",
        "ADX",
    ],
    "chart_pattern": [
        "Head and Shoulders",
        "Double Top/Bottom",
        "Triangle",
        "Flag",
        "Wedge",
        "Cup and Handle",
    ],
    "orderflow": [
        "Order Book Imbalance",
        "Trade Flow Toxicity",
        "Volume Profile",
        "Footprint Analysis",
        "Delta Divergence",
    ],
    "macro": [
        "Yield Curve",
        "CPI Surprises",
        "Employment Data",
        "FX Rates",
        "Credit Spreads",
    ],
    "news": [
        "Sentiment Analysis",
        "News Frequency",
        "Entity Recognition",
        "Topic Modeling",
        "Contrarian Signal",
    ],
}

RISK_ATTITUDES = ["conservative", "moderate", "aggressive"]
CONFIDENCE_CALIBRATIONS = ["default", "isotonic", "platt", "temperature"]
WEIGHTING_STRATEGIES = ["default", "equal", "performance_weighted", "entropy_weighted"]
CONTEXT_STRATEGIES = ["bounded", "sliding", "expanding", "regime_aware"]
SCHEMAS = ["signal-v1", "signal-v2", "structured-v1"]
MODEL_PROFILES = ["local-main", "local-fast", "local-critic"]


class MutationError(Exception):
    pass


def _random_choice(items: list, n: int) -> list:
    n = min(n, len(items))
    return random.sample(items, n) if n > 0 else []


def _random_float(low: float, high: float) -> float:
    return round(random.uniform(low, high), 4)


class MutationStrategy:
    def mutate(self, parent: AgentGenome, description: str = "") -> AgentGenome:
        raise NotImplementedError


class IndicatorMutation(MutationStrategy):
    def __init__(self, add: bool = True, n_changes: int = 1) -> None:
        self.add = add
        self.n_changes = n_changes

    def mutate(self, parent: AgentGenome, description: str = "") -> AgentGenome:
        child = copy.deepcopy(parent)
        indicators = list(parent.indicators)
        if self.add:
            category = parent.category
            available = STRATEGIES.get(category, STRATEGIES.get("technical", []))
            new = [i for i in available if i not in indicators]
            to_add = _random_choice(new, self.n_changes)
            indicators.extend(to_add)
        else:
            if len(indicators) > 1:
                remove = random.sample(indicators, min(self.n_changes, len(indicators) - 1))
                indicators = [i for i in indicators if i not in remove]
        child.indicators = sorted(set(indicators))
        child.prompt_version = str(int(parent.prompt_version) + 1)
        child.generation = parent.generation + 1
        return child


class TimeframeMutation(MutationStrategy):
    def __init__(self) -> None:
        self.timeframes = ["1m", "5m", "15m", "1h", "4h", "1d", "1w"]

    def mutate(self, parent: AgentGenome, description: str = "") -> AgentGenome:
        child = copy.deepcopy(parent)
        current = set(parent.timeframes) if parent.timeframes else {"1h"}
        action = random.choice(["add", "remove", "replace"])
        if action == "add":
            available = [t for t in self.timeframes if t not in current]
            if available:
                current.add(random.choice(available))
        elif action == "remove" and len(current) > 1:
            current.discard(random.choice(list(current)))
        elif action == "replace":
            available = [t for t in self.timeframes if t not in current]
            if available:
                current.discard(random.choice(list(current))) if current else None
                current.add(random.choice(available))
        child.timeframes = sorted(current)
        child.prompt_version = str(int(parent.prompt_version) + 1)
        child.generation = parent.generation + 1
        return child


class ParameterMutation(MutationStrategy):
    def mutate(self, parent: AgentGenome, description: str = "") -> AgentGenome:
        child = copy.deepcopy(parent)
        attr = random.choice(
            ["risk_attitude", "confidence_calibration", "weighting_strategy",
             "context_window_strategy", "output_schema", "model_profile"]
        )
        value_map = {
            "risk_attitude": RISK_ATTITUDES,
            "confidence_calibration": CONFIDENCE_CALIBRATIONS,
            "weighting_strategies": WEIGHTING_STRATEGIES,
            "context_window_strategy": CONTEXT_STRATEGIES,
            "output_schema": SCHEMAS,
            "model_profile": MODEL_PROFILES,
        }
        values = value_map.get(attr, [getattr(parent, attr, "default")])
        current = getattr(child, attr, "default")
        new_val = random.choice([v for v in values if v != current]) if len(values) > 1 else current
        if new_val != current:
            setattr(child, attr, new_val)
            child.prompt_version = str(int(parent.prompt_version) + 1)
            child.generation = parent.generation + 1
        return child


class TemperatureMutation(MutationStrategy):
    def __init__(self, delta_range: float = 0.1) -> None:
        self.delta_range = delta_range

    def mutate(self, parent: AgentGenome, description: str = "") -> AgentGenome:
        child = copy.deepcopy(parent)
        delta = random.uniform(-self.delta_range, self.delta_range)
        child.temperature = round(max(0.05, min(1.0, parent.temperature + delta)), 4)
        child.prompt_version = str(int(parent.prompt_version) + 1)
        child.generation = parent.generation + 1
        return child


class RecombinationStrategy(MutationStrategy):
    def __init__(self, n_crossovers: int = 3) -> None:
        self.n_crossovers = n_crossovers

    def mutate(self, parent: AgentGenome, description: str = "") -> AgentGenome:
        raise MutationError("Recombination requires two parents — use AgentFactory.recombine()")


class SpecializationStrategy(MutationStrategy):
    def __init__(self, target_regime: str | None = None) -> None:
        self.target_regime = target_regime

    def mutate(self, parent: AgentGenome, description: str = "") -> AgentGenome:
        child = copy.deepcopy(parent)
        child.generation = parent.generation + 1
        child.prompt_version = str(int(parent.prompt_version) + 1)
        if self.target_regime:
            child.feature_preferences = list(
                set(parent.feature_preferences + [f"regime_{self.target_regime}"])
            )
        else:
            parent_prefs = set(parent.feature_preferences)
            new_pref = random.choice(["trend_strength", "mean_reversion", "momentum",
                                       "volatility_regime", "liquidity_score"])
            parent_prefs.add(new_pref)
            child.feature_preferences = sorted(parent_prefs)
        return child


class SimplificationStrategy(MutationStrategy):
    def __init__(self, max_indicators: int = 3) -> None:
        self.max_indicators = max_indicators

    def mutate(self, parent: AgentGenome, description: str = "") -> AgentGenome:
        child = copy.deepcopy(parent)
        if len(parent.indicators) > self.max_indicators:
            child.indicators = _random_choice(sorted(parent.indicators), self.max_indicators)
        if len(parent.timeframes) > 2:
            child.timeframes = _random_choice(sorted(parent.timeframes), 2)
        child.generation = parent.generation + 1
        child.prompt_version = str(int(parent.prompt_version) + 1)
        return child


class DiversityInjectionStrategy(MutationStrategy):
    def __init__(self) -> None:
        self.timeframes = ["1m", "5m", "15m", "1h", "4h", "1d", "1w"]

    def mutate(self, parent: AgentGenome, description: str = "") -> AgentGenome:
        child = copy.deepcopy(parent)
        child.generation = parent.generation + 1
        child.prompt_version = str(int(parent.prompt_version) + 1)
        if len(parent.indicators) < 3:
            category = parent.category
            available = STRATEGIES.get(category, STRATEGIES.get("technical", []))
            child.indicators = _random_choice(available, min(3, len(available)))
        if not parent.timeframes:
            child.timeframes = _random_choice(self.timeframes, 2)
        if parent.risk_attitude == "conservative":
            child.risk_attitude = random.choice(["moderate", "aggressive"])
        child.temperature = _random_float(0.1, 0.5)
        return child


class AgentFactory:
    """Generates new agent genomes via mutation and recombination of existing agents."""

    def __init__(self, population_policy: dict | None = None) -> None:
        self.population_policy = population_policy or {}
        self.mutation_budget = self.population_policy.get(
            "evolution_budget",
            {"exploitation": 0.70, "exploration": 0.20, "radical_exploration": 0.10},
        )

    def generate_from_parent(
        self,
        parent: AgentGenome,
        mutation_type: MutationType = MutationType.INDICATOR_ADD,
        hypothesized_advantage: str = "",
        expected_failure_modes: list[str] | None = None,
    ) -> tuple[AgentGenome, GenomeMutation]:
        mutation_record = GenomeMutation(
            agent_id=parent.id,
            generation=parent.generation + 1,
            mutation_type=mutation_type,
            description=f"{mutation_type.value} on agent {parent.id}",
            hypothesized_advantage=hypothesized_advantage,
            expected_failure_modes=expected_failure_modes or [],
        )
        child = self._apply_mutation(parent, mutation_type)
        child.status = AgentStatus.CANDIDATE
        child.parent_agents = [parent.id]
        return child, mutation_record

    def recombine(
        self,
        parent_a: AgentGenome,
        parent_b: AgentGenome,
        hypothesized_advantage: str = "",
        expected_failure_modes: list[str] | None = None,
    ) -> tuple[AgentGenome, GenomeMutation]:
        if parent_a.category != parent_b.category:
            raise MutationError(
                f"Cannot recombine different categories: {parent_a.category} vs {parent_b.category}"
            )
        child = copy.deepcopy(parent_a)
        crossover_fields = ["indicators", "timeframes", "feature_preferences", "statistical_methods"]
        for field in crossover_fields:
            pa_set = set(getattr(parent_a, field, []))
            pb_set = set(getattr(parent_b, field, []))
            union = sorted(pa_set | pb_set)
            half = max(1, len(union) // 2)
            child_list = _random_choice(union, half)
            setattr(child, field, child_list)
        if random.random() < 0.5:
            child.risk_attitude = parent_b.risk_attitude
            child.weighting_strategy = parent_b.weighting_strategy
        child.generation = max(parent_a.generation, parent_b.generation) + 1
        child.prompt_version = str(int(parent_a.prompt_version) + 1)
        child.status = AgentStatus.CANDIDATE
        child.parent_agents = [parent_a.id, parent_b.id]
        mutation_record = GenomeMutation(
            agent_id=child.id,
            generation=child.generation,
            mutation_type=MutationType.RECOMBINATION,
            description=f"Recombination of {parent_a.id} and {parent_b.id}",
            hypothesized_advantage=hypothesized_advantage,
            expected_failure_modes=expected_failure_modes or [],
            created_at=datetime.now(UTC),
        )
        return child, mutation_record

    def specialize(
        self,
        parent: AgentGenome,
        target_regime: str,
        hypothesized_advantage: str = "",
        expected_failure_modes: list[str] | None = None,
    ) -> tuple[AgentGenome, GenomeMutation]:
        strategy = SpecializationStrategy(target_regime)
        child = strategy.mutate(parent)
        child.status = AgentStatus.CANDIDATE
        child.parent_agents = [parent.id]
        mutation_record = GenomeMutation(
            agent_id=child.id,
            generation=child.generation,
            mutation_type=MutationType.CONTEXT_WINDOW,
            description=f"Regime specialization ({target_regime}) of {parent.id}",
            hypothesized_advantage=hypothesized_advantage,
            expected_failure_modes=expected_failure_modes or [],
            created_at=datetime.now(UTC),
        )
        return child, mutation_record

    def simplify(
        self,
        parent: AgentGenome,
        max_indicators: int = 3,
        hypothesized_advantage: str = "",
        expected_failure_modes: list[str] | None = None,
    ) -> tuple[AgentGenome, GenomeMutation]:
        strategy = SimplificationStrategy(max_indicators)
        child = strategy.mutate(parent)
        child.status = AgentStatus.CANDIDATE
        child.parent_agents = [parent.id]
        mutation_record = GenomeMutation(
            agent_id=child.id,
            generation=child.generation,
            mutation_type=MutationType.INDICATOR_REMOVE,
            description=f"Simplification of {parent.id} (max {max_indicators} indicators)",
            hypothesized_advantage=hypothesized_advantage,
            expected_failure_modes=expected_failure_modes or [],
            created_at=datetime.now(UTC),
        )
        return child, mutation_record

    def inject_diversity(
        self,
        parent: AgentGenome,
        hypothesized_advantage: str = "",
        expected_failure_modes: list[str] | None = None,
    ) -> tuple[AgentGenome, GenomeMutation]:
        strategy = DiversityInjectionStrategy()
        child = strategy.mutate(parent)
        child.status = AgentStatus.CANDIDATE
        child.parent_agents = [parent.id]
        mutation_record = GenomeMutation(
            agent_id=child.id,
            generation=child.generation,
            mutation_type=MutationType.STATISTICAL_METHOD_MODIFY,
            description=f"Diversity injection into {parent.id}",
            hypothesized_advantage=hypothesized_advantage,
            expected_failure_modes=expected_failure_modes or [],
            created_at=datetime.now(UTC),
        )
        return child, mutation_record

    def _apply_mutation(
        self, parent: AgentGenome, mutation_type: MutationType
    ) -> AgentGenome:
        strategies: dict[MutationType, MutationStrategy] = {
            MutationType.INDICATOR_ADD: IndicatorMutation(add=True),
            MutationType.INDICATOR_REMOVE: IndicatorMutation(add=False),
            MutationType.TIMEFRAME_MODIFY: TimeframeMutation(),
            MutationType.FEATURE_PREFERENCE_MODIFY: ParameterMutation(),
            MutationType.STATISTICAL_METHOD_MODIFY: ParameterMutation(),
            MutationType.WEIGHTING_STRATEGY: ParameterMutation(),
            MutationType.CONFIDENCE_CALIBRATION: ParameterMutation(),
            MutationType.RISK_ATTITUDE: ParameterMutation(),
            MutationType.CONTEXT_WINDOW: ParameterMutation(),
            MutationType.OUTPUT_SCHEMA: ParameterMutation(),
            MutationType.MODEL_PROFILE: ParameterMutation(),
            MutationType.TEMPERATURE_MODIFY: TemperatureMutation(),
            MutationType.RECOMBINATION: RecombinationStrategy(),
        }
        strategy = strategies.get(mutation_type, IndicatorMutation(add=True))
        return strategy.mutate(parent)

    def generate_random(
        self, category: str, generation: int = 1, parent_agents: list[str] | None = None
    ) -> AgentGenome:
        category_strategies = STRATEGIES.get(category, STRATEGIES.get("technical", []))
        child = AgentGenome(
            id=f"agent-{uuid4()}",
            generation=generation,
            parent_agents=parent_agents or [],
            category=category,
            status=AgentStatus.CANDIDATE,
            prompt_version="1",
            indicators=_random_choice(category_strategies, random.randint(1, 4)),
            timeframes=_random_choice(["1m", "5m", "15m", "1h", "4h", "1d"], random.randint(1, 3)),
            risk_attitude=random.choice(RISK_ATTITUDES),
            confidence_calibration=random.choice(CONFIDENCE_CALIBRATIONS),
            weighting_strategy=random.choice(WEIGHTING_STRATEGIES),
            context_window_strategy=random.choice(CONTEXT_STRATEGIES),
            output_schema=random.choice(SCHEMAS),
            model_profile=random.choice(MODEL_PROFILES),
            temperature=_random_float(0.1, 0.5),
        )
        return child