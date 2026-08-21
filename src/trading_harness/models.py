from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(UTC)


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


class RunOutcome(StrEnum):
    LONG = "LONG"
    SHORT = "SHORT"
    WAIT = "WAIT"
    NO_TRADE = "NO_TRADE"


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
    requested_quantity: float = Field(default=1.0, gt=0)


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


class TradingRun(BaseModel):
    id: str = Field(default_factory=lambda: f"run-{uuid4()}")
    snapshot_id: str
    state: RunState = RunState.CREATED
    decisions: list[dict[str, Any]] = Field(default_factory=list)
    outcome: RunOutcome | None = None
    outcome_reason: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class PerformanceRecord(BaseModel):
    id: str = Field(default_factory=lambda: f"perf-{uuid4()}")
    run_id: str
    agent_id: str
    snapshot_id: str
    direction: str
    confidence: float
    outcome: str | None = None
    realized_pnl: float = 0.0
    mfe: float = 0.0
    mae: float = 0.0
    timestamp: datetime = Field(default_factory=utcnow)


class AuditEntry(BaseModel):
    id: str = Field(default_factory=lambda: f"audit-{uuid4()}")
    actor: str
    action: str
    entity_type: str
    entity_id: str
    previous_state: str | None = None
    new_state: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=utcnow)


class MarketRegime(StrEnum):
    STRONG_BULL = "strong_bull"
    WEAK_BULL = "weak_bull"
    RANGE = "range"
    WEAK_BEAR = "weak_bear"
    STRONG_BEAR = "strong_bear"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRASH = "crash"
    RECOVERY = "recovery"
    UNKNOWN = "unknown"


class OutcomeRecord(BaseModel):
    """Actual market outcome for a prediction, enabling evaluation."""

    id: str = Field(default_factory=lambda: f"outcome-{uuid4()}")
    prediction_id: str
    agent_id: str
    run_id: str
    snapshot_id: str
    symbol: str
    direction_predicted: str
    direction_actual: str
    confidence_predicted: float
    entry_price: float
    exit_price: float
    mfe: float = 0.0
    mae: float = 0.0
    holding_period_bars: int = 0
    realized_pnl: float = 0.0
    regime: MarketRegime = MarketRegime.UNKNOWN
    timestamp: datetime = Field(default_factory=utcnow)


class EvaluationResult(BaseModel):
    """Result of evaluating predictions against outcomes."""

    id: str = Field(default_factory=lambda: f"eval-{uuid4()}")
    run_id: str
    agent_id: str
    metric_name: str
    metric_value: float
    observations: int
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=utcnow)


class WalkForwardResult(BaseModel):
    """Result of a single walk-forward window evaluation."""

    window_id: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    metric_name: str
    train_metric: float
    test_metric: float
    stability: float = 0.0
    timestamp: datetime = Field(default_factory=utcnow)


class AgentSignal(BaseModel):
    """Structured analysis output from an agent's LLM analysis."""

    id: str = Field(default_factory=lambda: f"signal-{uuid4()}")
    run_id: str
    agent_id: str
    snapshot_id: str
    category: str
    direction: str  # LONG, SHORT, NO_TRADE
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    signals: list[dict[str, Any]] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=utcnow)


class AgentAnalysisResult(BaseModel):
    """Complete result of running an agent's analysis on a snapshot."""

    run_id: str
    agent_id: str
    signal: AgentSignal
    prompt_version: str
    model_profile: str
    raw_response: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Phase 3 — Evolution
# ---------------------------------------------------------------------------


class MutationType(StrEnum):
    INDICATOR_ADD = "INDICATOR_ADD"
    INDICATOR_REMOVE = "INDICATOR_REMOVE"
    TIMEFRAME_MODIFY = "TIMEFRAME_MODIFY"
    FEATURE_PREFERENCE_MODIFY = "FEATURE_PREFERENCE_MODIFY"
    STATISTICAL_METHOD_MODIFY = "STATISTICAL_METHOD_MODIFY"
    WEIGHTING_STRATEGY = "WEIGHTING_STRATEGY"
    CONFIDENCE_CALIBRATION = "CONFIDENCE_CALIBRATION"
    RISK_ATTITUDE = "RISK_ATTITUDE"
    CONTEXT_WINDOW = "CONTEXT_WINDOW"
    OUTPUT_SCHEMA = "OUTPUT_SCHEMA"
    MODEL_PROFILE = "MODEL_PROFILE"
    TEMPERATURE_MODIFY = "TEMPERATURE_MODIFY"
    RECOMBINATION = "RECOMBINATION"


class GenomeMutation(BaseModel):
    agent_id: str
    generation: int
    mutation_type: MutationType
    description: str
    hypothesized_advantage: str = ""
    expected_failure_modes: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)


class HallOfFameRecord(BaseModel):
    agent_id: str
    category: str
    score: float
    observations: int
    added_at: datetime = Field(default_factory=utcnow)
    reason: str = ""


class GraveyardRecord(BaseModel):
    agent_id: str
    category: str
    final_score: float
    reason: str
    retired_at: datetime = Field(default_factory=utcnow)


class ChampionChallenger(BaseModel):
    champion_id: str
    challenger_id: str
    category: str
    champion_score: float
    challenger_score: float
    observations: int
    created_at: datetime = Field(default_factory=utcnow)


class EvolutionRun(BaseModel):
    id: str = Field(default_factory=lambda: f"evo-{uuid4()}")
    category: str
    method: str
    new_agent_ids: list[str] = Field(default_factory=list)
    parent_agent_ids: list[str] = Field(default_factory=list)
    score_delta: float = 0.0
    observations: int = 0
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None


class RollbackEntry(BaseModel):
    agent_id: str
    previous_status: AgentStatus
    new_status: AgentStatus
    reason: str
    timestamp: datetime = Field(default_factory=utcnow)


# ---------------------------------------------------------------------------
# Phase 4 — Paper Trading
# ---------------------------------------------------------------------------


class PaperTradeStatus(StrEnum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


class PaperPositionStatus(StrEnum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    STOPPED_OUT = "STOPPED_OUT"
    TARGET_HIT = "TARGET_HIT"


class PaperTrade(BaseModel):
    """Represents a simulated order in the paper trading system."""

    id: str = Field(default_factory=lambda: f"paper-trade-{uuid4()}")
    trade_id: str  # Original decision_id from TradingRun
    run_id: str
    symbol: str
    side: str  # LONG / SHORT
    equity: float = Field(gt=0)
    entry_price: float = Field(gt=0)
    requested_leverage: float = Field(default=1.0, gt=0)
    requested_quantity: float = Field(default=1.0, gt=0)
    actual_quantity: float = 0.0
    actual_price: float = 0.0
    stop_price: float = Field(gt=0)
    target_price: float = 0.0
    fill_rate: float = Field(default=0.8, ge=0.0, le=1.0)
    slippage_bps: float = Field(default=0.0, ge=0.0)
    fees: float = 0.0
    status: PaperTradeStatus = PaperTradeStatus.PENDING
    partial_fills: list[dict[str, float]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    filled_at: datetime | None = None
    closed_at: datetime | None = None
    reject_reason: str | None = None


class PaperPosition(BaseModel):
    """Represents an open or closed paper trading position."""

    id: str = Field(default_factory=lambda: f"paper-pos-{uuid4()}")
    trade_id: str
    run_id: str
    symbol: str
    side: str  # LONG / SHORT
    entry_price: float = Field(gt=0)
    quantity: float = Field(gt=0)
    fees: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    status: PaperPositionStatus = PaperPositionStatus.OPEN
    open_timestamp: datetime = Field(default_factory=utcnow)
    close_timestamp: datetime | None = None
    close_price: float | None = None
    close_reason: str | None = None  # MANUAL, STOP_LOSS, TARGET_HIT


class PortfolioState(BaseModel):
    """Aggregated portfolio state for a paper trading run."""

    id: str = Field(default_factory=lambda: f"portfolio-{uuid4()}")
    run_id: str
    start_equity: float = 100000.0
    current_equity: float = 100000.0
    total_realized_pnl: float = 0.0
    total_unrealized_pnl: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    peak_equity: float = 100000.0
    positions: dict[str, float] = Field(default_factory=dict)
    symbols: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=utcnow)


class AgentPerformanceSummary(BaseModel):
    """Per-agent performance summary for paper trading."""

    agent_id: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    total_realized_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    timestamp: datetime = Field(default_factory=utcnow)


# ---------------------------------------------------------------------------
# Shadow Trading — Loop-Modelle (Epic, Spec §6.1)
# ---------------------------------------------------------------------------


class ShadowLoopStatus(StrEnum):
    """Lifecycle status of the shadow trading loop (ST.8)."""

    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"


class ShadowTradingStatus(StrEnum):
    """Status of a single shadow trading decision (ST.7)."""

    NO_TRADE = "NO_TRADE"
    REJECTED = "REJECTED"
    FILLED = "FILLED"


class ShadowSessionState(BaseModel):
    """Persistent state of the shadow trading loop (ST.8, data/shadow_trading_state.json).

    state_checksum: SHA-256 (hex) des kanonisierten JSONs dieses States, berechnet
    OHNE das state_checksum-Feld selbst (F5). Mismatch beim Load => korrupt (Quarantäne).
    Berechnung/Verifikation erfolgt in der ShadowTradingStateStore (WI-ST-02).
    """

    session_id: str = Field(default_factory=lambda: f"shadow-{uuid4()}")
    status: ShadowLoopStatus = ShadowLoopStatus.STOPPED
    symbols: list[str] = Field(default_factory=list)
    interval_seconds: int = 900
    started_at: datetime | None = None
    stopped_at: datetime | None = None
    last_iteration_at: datetime | None = None
    iteration_count: int = 0
    decisions_today: int = 0
    budget_date: str = ""  # "YYYY-MM-DD" (UTC)
    start_equity: float = 100000.0
    current_equity: float = 100000.0
    open_positions: int = 0
    error_count: int = 0
    last_error: str | None = None
    restart_required: bool = False
    state_checksum: str = ""  # SHA-256 (hex), siehe Class-Docstring (F5)


class ShadowTradingRecord(BaseModel):
    """One record per loop decision (ST.7); last audit-link for FILLED is trade_id (E8).

    trade_id: Paper-Trade-ID, nur bei status=FILLED gesetzt, sonst None —
    schließt die Audit-Kette snapshot_id -> run_id -> decision_id -> record_id -> trade_id.
    Bewusst anders als ShadowModeLogger.ShadowTradeRecord (Phase 5, Spec-Z3).
    """

    id: str = Field(default_factory=lambda: f"shadow-rec-{uuid4()}")
    timestamp: datetime = Field(default_factory=utcnow)
    symbol: str
    side: str  # "BUY" | "SELL" | "NONE"
    direction: str  # "LONG" | "SHORT" | "NO_TRADE"
    status: ShadowTradingStatus
    decision_id: str
    run_id: str
    snapshot_id: str
    trade_id: str | None = None  # Paper-Trade-ID (FILLED), E8-Audit-Kette
    risk_reason: str = ""
    agent_ids: list[str] = Field(default_factory=list)
    quantity: float = 0.0
    entry_price: float = 0.0
    mark_price: float = 0.0
    slippage: float = 0.0
    commission: float = 0.0
    pnl_estimate: float = 0.0
    slippage_rate: float = 0.0
    commission_rate: float = 0.0
    config_version: str = ""  # sha256(risk-policy + settings)


class SignalAggregation(BaseModel):
    """Deterministic aggregation of agent signals for one symbol (ST.6)."""

    direction: str  # "LONG" | "SHORT" | "NO_TRADE"
    confidence: float
    signal_count: int
    no_trade_count: int
    agent_ids: list[str] = Field(default_factory=list)


class ShadowEvaluationSummary(BaseModel):
    """Phase-2 metrics computed over accumulated ShadowTradingRecords (ST.12/O4)."""

    brier_score: float
    directional_accuracy: float
    expectancy: float
    max_drawdown: float
    sample_size: int
