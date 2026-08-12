"""Governance — Agent Lifecycle State Machine, Promotion, Quarantine, Audit.

EPIC-11 (Governance & Agent Lifecycle) — State Machine, Promotion Rules,
Quarantine, Champion-Challenger, Audit Trail, Kill Criteria.

EPIC-09 Decision-Engine ist in decision_engine.py (FinalDecision logic).
"""

from __future__ import annotations

from .audit import AuditEntry, AuditTrail
from .audit.kill_criteria import (
    KillCriteriaConfig,
    KillCriteriaEngine,
    KillCriteriaResult,
    KillSeverity,
)
from .champion_challenger import (
    AgentVersion,
    ChampionChallengerConfig,
    ChampionChallengerEngine,
    EvaluationResult,
)
from .decision_engine import (
    BlockingRules,
    DecisionEngine,
    DecisionRule,
    FinalDecisionData,
    FinalDecisionType,
    GovernanceConfig,
)
from .promotion_rules import (
    ActiveAgent,
    DegradedAgent,
    PromotionCriteria,
    PromotionRuleEngine,
    ShadowAgent,
)
from .quarantine import QuarantineEngine, QuarantineEvent, QuarantineReason
from .state_machine import (
    STATE_WEIGHTS,
    VALID_TRANSITIONS,
    AgentState,
    GovernanceStateMachine,
    StateTransition,
)

__all__ = [
    "STATE_WEIGHTS",
    "VALID_TRANSITIONS",
    "ActiveAgent",
    "AgentState",
    "AgentVersion",
    "AuditEntry",
    "AuditTrail",
    "BlockingRules",
    "ChampionChallengerConfig",
    "ChampionChallengerEngine",
    "DecisionEngine",
    "DecisionRule",
    "DegradedAgent",
    "EvaluationResult",
    "FinalDecisionData",
    "FinalDecisionType",
    "GovernanceConfig",
    "GovernanceStateMachine",
    "KillCriteriaConfig",
    "KillCriteriaEngine",
    "KillCriteriaResult",
    "KillSeverity",
    "PromotionCriteria",
    "PromotionRuleEngine",
    "QuarantineEngine",
    "QuarantineEvent",
    "QuarantineReason",
    "ShadowAgent",
    "StateTransition",
]
