"""Phased Rollout Controller — gradual live-trading deployment.

This package implements the controlled rollout of a live-trading strategy
through five ordered phases (SHADOW → PAPER → LIVE_SMALL → LIVE_MEDIUM →
LIVE_FULL) with automated promotion, demotion, kill-switch, and circuit-
breaker logic.

Public API
----------
- PhasedRolloutController — main state machine
- RolloutPhase — ordered enum of rollout phases
- RolloutDecision — outcome of a single evaluation cycle
- RolloutControllerState — persistent state snapshot
- RolloutThresholds — all configurable threshold values
- CircuitBreaker — exchange error-rate circuit breaker
- CircuitState — circuit breaker state ("closed" / "open")
- KillSwitch — immediate halt mechanism
- KillSwitchState — kill switch state ("disabled" / "activated")
"""

from __future__ import annotations

from .circuit_breaker import CircuitBreaker, CircuitState
from .controller import (
    PhasedRolloutController,
    RolloutControllerState,
    RolloutDecision,
    RolloutPhase,
)
from .kill_switch import KillSwitch, KillSwitchState
from .thresholds import RolloutThresholds

__all__ = [
    "CircuitBreaker",
    "CircuitState",
    "KillSwitch",
    "KillSwitchState",
    "PhasedRolloutController",
    "RolloutControllerState",
    "RolloutDecision",
    "RolloutPhase",
    "RolloutThresholds",
]
