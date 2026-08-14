from __future__ import annotations

import logging
from typing import Any

from trading_harness.models import (
    AgentGenome,
    AgentStatus,
    ChampionChallenger,
    EvolutionRun,
    GenomeMutation,
    PromotionDecision,
    RollbackEntry,
)
from trading_harness.services.agent_factory import AgentFactory
from trading_harness.services.agent_genome_store import AgentGenomeStore
from trading_harness.services.challenger_pool import ChallengerPool
from trading_harness.services.evolution import PromotionPolicy
from trading_harness.services.graveyard import Graveyard
from trading_harness.services.hall_of_fame import HallOfFame

logger = logging.getLogger(__name__)


class EvolutionService:
    """Orchestrates agent evolution: generation, evaluation, promotion, retirement, rollback."""

    def __init__(
        self,
        genome_store: AgentGenomeStore,
        promotion_policy: PromotionPolicy,
        population_policy: dict,
        factory: AgentFactory | None = None,
    ) -> None:
        self._store = genome_store
        self._policy = promotion_policy
        self._pop_policy = population_policy
        self._factory = factory or AgentFactory(population_policy)
        self._challenger_pool = ChallengerPool(
            genome_store, promotion_policy, population_policy
        )
        self._hall_of_fame = HallOfFame()
        self._graveyard = Graveyard()
        self._rollbacks: list[RollbackEntry] = []

    @property
    def hall_of_fame(self) -> HallOfFame:
        return self._hall_of_fame

    @property
    def graveyard(self) -> Graveyard:
        return self._graveyard

    def generate_mutant(
        self,
        parent: AgentGenome,
        mutation_type: str = "INDICATOR_ADD",
        hypothesized_advantage: str = "",
        expected_failure_modes: list[str] | None = None,
    ) -> tuple[AgentGenome, GenomeMutation]:
        from trading_harness.models import MutationType

        mt = MutationType(mutation_type)
        child, record = self._factory.generate_from_parent(
            parent,
            mutation_type=mt,
            hypothesized_advantage=hypothesized_advantage,
            expected_failure_modes=expected_failure_modes or [],
        )
        self._store.add(child)
        logger.info("Generated mutant %s (gen %d) from %s via %s",
                     child.id, child.generation, parent.id, mutation_type)
        return child, record

    def recombine(
        self,
        parent_a: AgentGenome,
        parent_b: AgentGenome,
        hypothesized_advantage: str = "",
        expected_failure_modes: list[str] | None = None,
    ) -> tuple[AgentGenome, GenomeMutation]:
        child, record = self._factory.recombine(
            parent_a, parent_b,
            hypothesized_advantage=hypothesized_advantage,
            expected_failure_modes=expected_failure_modes or [],
        )
        self._store.add(child)
        logger.info("Recombined %s and %s into %s (gen %d)",
                     parent_a.id, parent_b.id, child.id, child.generation)
        return child, record

    def add_challenger(self, agent: AgentGenome) -> AgentGenome:
        agent.status = AgentStatus.CHALLENGER
        return self._challenger_pool.add_challenger(agent)

    def evaluate_challenger(
        self,
        challenger_id: str,
        champion_id: str,
        category: str,
        challenger_score: float,
        incumbent_score: float,
        observations: int,
        out_of_sample_pass: bool = True,
        walk_forward_pass: bool = True,
        shadow_pass: bool = True,
        ensemble_contribution: float = 0.0,
        security_pass: bool = True,
    ) -> PromotionDecision:
        return self._challenger_pool.evaluate_promotion(
            champion_id=champion_id,
            challenger_id=challenger_id,
            category=category,
            challenger_score=challenger_score,
            incumbent_score=incumbent_score,
            observations=observations,
            out_of_sample_pass=out_of_sample_pass,
            walk_forward_pass=walk_forward_pass,
            shadow_pass=shadow_pass,
            ensemble_contribution=ensemble_contribution,
            security_pass=security_pass,
        )

    def promote_challenger(
        self,
        challenger_id: str,
        incumbent_id: str,
        category: str,
    ) -> EvolutionRun:
        run = self._challenger_pool.promote_challenger(
            challenger_id, incumbent_id, category,
        )
        logger.info("Promoted %s to ACTIVE, demoted %s to PROBATION in %s",
                     challenger_id, incumbent_id, category)
        return run

    def retire_agent(
        self,
        agent: AgentGenome,
        reason: str,
        final_score: float = 0.0,
    ) -> None:
        previous_status = agent.status
        agent.status = AgentStatus.RETIRED
        self._store.update(agent)
        self._graveyard.add(
            agent_id=agent.id,
            category=agent.category,
            final_score=final_score,
            reason=reason,
        )
        self._rollback(
            agent_id=agent.id,
            previous_status=previous_status,
            new_status=AgentStatus.RETIRED,
            reason=reason,
        )
        logger.info("Retired agent %s (was %s) in category %s",
                     agent.id, previous_status, agent.category)

    def reject_challenger(
        self,
        agent: AgentGenome,
        reason: str,
    ) -> None:
        agent.status = AgentStatus.REJECTED
        self._store.update(agent)
        self._graveyard.add(
            agent_id=agent.id,
            category=agent.category,
            final_score=0.0,
            reason=reason,
        )
        logger.info("Rejected challenger %s: %s", agent.id, reason)

    def demote_to_probation(self, agent_id: str, reason: str = "PROMOTION_FAILED") -> None:
        agent = self._store.get(agent_id)
        if agent is None:
            raise ValueError(f"Agent {agent_id} not found")
        previous_status = agent.status
        agent.status = AgentStatus.PROBATION
        self._store.update(agent)
        self._rollback(
            agent_id=agent_id,
            previous_status=previous_status,
            new_status=AgentStatus.PROBATION,
            reason=reason,
        )
        logger.info("Demoted agent %s to PROBATION: %s", agent_id, reason)

    def get_challenger_pairs(self, category: str) -> list[ChampionChallenger]:
        return self._challenger_pool.get_challenger_pairs(category)

    def get_population_stats(self, category: str) -> dict[str, Any]:
        return self._challenger_pool.get_category_size(category)

    def get_promotion_history(self, category: str) -> list[EvolutionRun]:
        policy_categories = self._pop_policy.get("categories", {})
        policy_categories.get(category, {"active": 5, "challengers": 10})
        return []

    def get_rollbacks(self) -> list[RollbackEntry]:
        return list(self._rollbacks)

    def get_rollbacks_for_agent(self, agent_id: str) -> list[RollbackEntry]:
        return [r for r in self._rollbacks if r.agent_id == agent_id]

    def _rollback(
        self,
        agent_id: str,
        previous_status: AgentStatus,
        new_status: AgentStatus,
        reason: str,
    ) -> None:
        entry = RollbackEntry(
            agent_id=agent_id,
            previous_status=previous_status,
            new_status=new_status,
            reason=reason,
        )
        self._rollbacks.append(entry)

    def rollback_agent_status(
        self,
        agent_id: str,
        target_status: AgentStatus,
        reason: str,
    ) -> RollbackEntry:
        agent = self._store.get(agent_id)
        if agent is None:
            raise ValueError(f"Agent {agent_id} not found")
        previous_status = agent.status
        agent.status = target_status
        self._store.update(agent)
        entry = RollbackEntry(
            agent_id=agent_id,
            previous_status=previous_status,
            new_status=target_status,
            reason=reason,
        )
        self._rollbacks.append(entry)
        logger.info("Rolled back agent %s from %s to %s: %s",
                     agent_id, previous_status, target_status, reason)
        return entry