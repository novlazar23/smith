from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(UTC)


class AgentStatus(StrEnum):
    GENERATED = "GENERATED"
    CANDIDATE = "CANDIDATE"
    CHALLENGER = "CHALLENGER"
    ACTIVE = "ACTIVE"
    CHAMPION = "CHAMPION"
    PROBATION = "PROBATION"
    RETIRED = "RETIRED"
    REJECTED = "REJECTED"


class AgentGenome(BaseModel):
    id: str = Field(default_factory=lambda: f"agent-{uuid4()}")
    generation: int = 0
    parent_agents: list[str] = Field(default_factory=list)
    category: str
    status: AgentStatus = AgentStatus.GENERATED
    prompt_version: str = "1"
    reasoning_style: str = "systematic"
    indicators: list[str] = Field(default_factory=list)
    timeframes: list[str] = Field(default_factory=list)
    feature_preferences: list[str] = Field(default_factory=list)
    statistical_methods: list[str] = Field(default_factory=list)
    weighting_strategy: str = "default"
    confidence_calibration: str = "default"
    risk_attitude: str = "conservative"
    context_window_strategy: str = "bounded"
    output_schema: str = "signal-v1"
    model_profile: str = "local-main"
    temperature: float = 0.2
    created_at: datetime = Field(default_factory=utcnow)


class MarketSnapshot(BaseModel):
    id: str = Field(default_factory=lambda: f"snap-{uuid4()}")
    symbol: str
    timestamp: datetime = Field(default_factory=utcnow)
    data: dict[str, Any]
    content_hash: str | None = None


class TradeProposal(BaseModel):
    decision_id: str
    symbol: str
    side: str
    equity: float = Field(gt=0)
    entry_price: float = Field(gt=0)
    stop_price: float = Field(gt=0)
    target_price: float = Field(gt=0)
    requested_leverage: float = Field(default=1.0, gt=0)
    open_positions: int = Field(default=0, ge=0)
    current_daily_loss_fraction: float = Field(default=0.0, ge=0)
    current_portfolio_risk_fraction: float = Field(default=0.0, ge=0)
    expected_slippage_bps: float = Field(default=0.0, ge=0)


class RiskDecision(BaseModel):
    approved: bool
    reason: str
    max_position_size: float = 0.0
    risk_amount: float = 0.0
    risk_fraction: float = 0.0
    risk_reward: float = 0.0


class ChallengerEvaluation(BaseModel):
    challenger_id: str
    incumbent_id: str
    category: str
    incumbent_category: str
    observations: int
    incumbent_score: float
    challenger_score: float
    out_of_sample_pass: bool
    walk_forward_pass: bool
    shadow_pass: bool
    ensemble_contribution: float
    security_pass: bool


class PromotionDecision(BaseModel):
    promote: bool
    reason: str
    relative_improvement: float = 0.0
