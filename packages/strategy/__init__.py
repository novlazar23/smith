"""Strategy module — entry/stop/target/exit logic.

Defines StrategyDirection, StrategyProposal, StrategyConfig,
StrategyVariant, entry signals, target levels, EV calculation,
and the StrategyEngine orchestration.
"""

from __future__ import annotations

from .engine import StrategyEngine
from .entry import EntryCondition, EntrySignal, EntryType, evaluate_entry
from .evaluation import (
    apply_gates,
    calculate_expected_return,
    calculate_risk_reward,
    evaluate_variant,
)
from .models import (
    StrategyConfig,
    StrategyDirection,
    StrategyProposal,
    StrategyVariant,
)
from .targets import (
    TargetLevel,
    TargetType,
    calculate_prob_target_before_stop,
    calculate_targets,
    estimate_mfe_mae,
)

__all__ = [
    "EntryCondition",
    "EntrySignal",
    "EntryType",
    "StrategyConfig",
    "StrategyDirection",
    "StrategyEngine",
    "StrategyProposal",
    "StrategyVariant",
    "TargetLevel",
    "TargetType",
    "apply_gates",
    "calculate_expected_return",
    "calculate_prob_target_before_stop",
    "calculate_risk_reward",
    "calculate_targets",
    "estimate_mfe_mae",
    "evaluate_entry",
    "evaluate_variant",
]
