from __future__ import annotations

import logging

from trading_harness.models import (
    AgentGenome,
    AgentStatus,
    ChampionChallenger,
    EvolutionRun,
    PromotionDecision,
)
from trading_harness.services.agent_genome_store import AgentGenomeStore
from trading_harness.services.evolution import PromotionPolicy

logger = logging.getLogger(__name__)


def _score_for_selection(
    agent: AgentGenome,
    score_cache: dict[str, float],
) -> float:
    return score_cache.get(agent.id, 0.0)


class ChallengerPool:
    """Manages challenger pools per category with champion/challenger selection."""

    def __init__(
        self,
        genome_store: AgentGenomeStore,
        promotion_policy: PromotionPolicy,
        population_policy: dict,
    ) -> None:
        self._store = genome_store
        self._policy = promotion_policy
        self._pop_policy = population_policy
        self._champion_candidates: dict[str, AgentGenome] = {}
        self._challenger_pairs: dict[str, list[ChampionChallenger]] = {}

    def add_challenger(self, agent: AgentGenome) -> AgentGenome:
        agent.status = AgentStatus.CHALLENGER
        self._store.update(agent)
        return agent

    def get_challengers(self, category: str) -> list[AgentGenome]:
        return self._store.list_challengers(category)

    def get_active_agents(self, category: str) -> list[AgentGenome]:
        return self._store.list_active(category)

    def pair_for_evaluation(
        self,
        category: str,
        score_cache: dict[str, float] | None = None,
    ) -> ChampionChallenger | None:
        score_cache = score_cache or {}
        challengers = self._store.list_challengers(category)
        if not challengers:
            return None
        active = self._store.list_active(category)
        if not active:
            return None
        best_challenger = max(
            challengers,
            key=lambda a: _score_for_selection(a, score_cache),
        )
        best_champion = max(
            active,
            key=lambda a: _score_for_selection(a, score_cache),
        )
        pair = ChampionChallenger(
            champion_id=best_champion.id,
            challenger_id=best_challenger.id,
            category=category,
            champion_score=_score_for_selection(best_champion, score_cache),
            challenger_score=_score_for_selection(best_challenger, score_cache),
            observations=max(
                self._get_observations(best_champion.id, score_cache),
                self._get_observations(best_challenger.id, score_cache),
            ),
        )
        self._challenger_pairs.setdefault(category, []).append(pair)
        return pair

    def _get_observations(self, agent_id: str, score_cache: dict[str, float]) -> int:
        perf = score_cache.get(agent_id, 0.0)
        return int(perf) if perf > 0 else 0

    def evaluate_promotion(
        self,
        champion_id: str,
        challenger_id: str,
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
        from trading_harness.models import ChallengerEvaluation

        evaluation = ChallengerEvaluation(
            challenger_id=challenger_id,
            incumbent_id=champion_id,
            category=category,
            incumbent_category=category,
            observations=observations,
            incumbent_score=incumbent_score,
            challenger_score=challenger_score,
            out_of_sample_pass=out_of_sample_pass,
            walk_forward_pass=walk_forward_pass,
            shadow_pass=shadow_pass,
            ensemble_contribution=ensemble_contribution,
            security_pass=security_pass,
        )
        return self._policy.evaluate(evaluation)

    def promote_challenger(
        self,
        challenger_id: str,
        incumbent_id: str,
        category: str,
        promotion_reason: str = "PROMOTION_APPROVED",
    ) -> EvolutionRun:
        challenger = self._store.get(challenger_id)
        incumbent = self._store.get(incumbent_id)
        if challenger is None or incumbent is None:
            raise ValueError("Challenger or incumbent not found")
        challenger.status = AgentStatus.ACTIVE
        incumbent.status = AgentStatus.PROBATION
        self._store.update(challenger)
        self._store.update(incumbent)
        run = EvolutionRun(
            id=f"evo-promote-{challenger_id}",
            category=category,
            method="champion_challenger_promotion",
            new_agent_ids=[challenger_id],
            parent_agent_ids=[incumbent_id],
            score_delta=1.0,
            observations=1,
        )
        return run

    def demote_to_probation(
        self,
        agent_id: str,
        reason: str = "PROMOTION_FAILED",
    ) -> None:
        agent = self._store.get(agent_id)
        if agent is None:
            raise ValueError(f"Agent {agent_id} not found")
        agent.status = AgentStatus.PROBATION
        self._store.update(agent)

    def get_challenger_pairs(self, category: str) -> list[ChampionChallenger]:
        return self._challenger_pairs.get(category, [])

    def get_category_size(self, category: str) -> dict[str, object]:
        policy_categories = self._pop_policy.get("categories", {})
        cat_config = policy_categories.get(
            category, {"active": 5, "challengers": 10}
        )
        active = len(self._store.list_active(category))
        challengers = len(self._store.list_challengers(category))
        return {
            "category": category,
            "active": active,
            "challengers": challengers,
            "active_limit": int(cat_config["active"]),
            "challenger_limit": int(cat_config["challengers"]),
        }