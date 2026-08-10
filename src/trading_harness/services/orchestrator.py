from __future__ import annotations

from enum import StrEnum


class RunState(StrEnum):
    CREATED = "CREATED"
    DATA_READY = "DATA_READY"
    ANALYSIS_RUNNING = "ANALYSIS_RUNNING"
    ADVERSARIAL_REVIEW = "ADVERSARIAL_REVIEW"
    CONSENSUS = "CONSENSUS"
    RISK_REVIEW = "RISK_REVIEW"
    DECISION = "DECISION"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


ALLOWED_TRANSITIONS: dict[RunState, set[RunState]] = {
    RunState.CREATED: {RunState.DATA_READY, RunState.FAILED},
    RunState.DATA_READY: {RunState.ANALYSIS_RUNNING, RunState.FAILED},
    RunState.ANALYSIS_RUNNING: {RunState.ADVERSARIAL_REVIEW, RunState.FAILED},
    RunState.ADVERSARIAL_REVIEW: {RunState.CONSENSUS, RunState.FAILED},
    RunState.CONSENSUS: {RunState.RISK_REVIEW, RunState.FAILED},
    RunState.RISK_REVIEW: {RunState.DECISION, RunState.FAILED},
    RunState.DECISION: {RunState.COMPLETE, RunState.FAILED},
    RunState.COMPLETE: set(),
    RunState.FAILED: set(),
}


def transition(current: RunState, target: RunState) -> RunState:
    if target not in ALLOWED_TRANSITIONS[current]:
        raise ValueError(f"Invalid transition {current} -> {target}")
    return target
