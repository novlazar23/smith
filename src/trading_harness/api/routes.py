from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from trading_harness.api.security import require_read_key, require_trade_key
from trading_harness.config import get_settings
from trading_harness.models import (
    AgentAnalysisResult,
    AgentGenome,
    AgentStatus,
    ChallengerEvaluation,
    MarketRegime,
    MarketSnapshot,
    OutcomeRecord,
    PerformanceRecord,
    RunOutcome,
    TradeProposal,
)
from trading_harness.services.agent_analysis_store import PersistedAgentAnalysisStore
from trading_harness.services.agent_registry import AgentRegistry
from trading_harness.services.agent_runtime import AgentRuntime
from trading_harness.services.db import Database
from trading_harness.services.evaluation import EvaluationService
from trading_harness.services.evaluation_result_store import (
    PersistedEvaluationResultStore,
)
from trading_harness.services.evolution import PromotionPolicy
from trading_harness.services.execution_gateway import ExecutionGateway
from trading_harness.services.orchestrator import TradingRunService
from trading_harness.services.outcome_store import PersistedOutcomeStore
from trading_harness.services.performance_store import PersistedPerformanceStore
from trading_harness.services.persisted_agent_registry import PersistedAgentRegistry
from trading_harness.services.persisted_snapshot_store import PersistedSnapshotStore
from trading_harness.services.policy_loader import load_yaml
from trading_harness.services.risk_engine import RiskEngine
from trading_harness.services.snapshot_store import SnapshotStore

router = APIRouter()
settings = get_settings()

_db = Database(settings.database_url)
agent_store: AgentRegistry | PersistedAgentRegistry = PersistedAgentRegistry(_db)
snapshot_store_inst: SnapshotStore | PersistedSnapshotStore = PersistedSnapshotStore(_db)
risk_engine = RiskEngine(load_yaml(settings.risk_policy_path))
promotion_policy = PromotionPolicy(load_yaml(settings.population_policy_path))
execution_gateway = ExecutionGateway(settings.live_execution_enabled)
kill_switch = settings.kill_switch_default
trading_run_service = TradingRunService()
performance_store = PersistedPerformanceStore(_db)
outcome_store: PersistedOutcomeStore = PersistedOutcomeStore(_db)
result_store = PersistedEvaluationResultStore(_db)
evaluation_service = EvaluationService(outcome_store, performance_store, result_store)
analysis_store: PersistedAgentAnalysisStore = PersistedAgentAnalysisStore(_db)
agent_runtime = AgentRuntime(analysis_store)

# ---------------------------------------------------------------------------
# Phase 3 — Evolution services
# ---------------------------------------------------------------------------

from trading_harness.services.agent_factory import AgentFactory
from trading_harness.services.agent_genome_store import AgentGenomeStore
from trading_harness.services.evolution_service import EvolutionService

evolution_genome_store = AgentGenomeStore()
evolution_factory = AgentFactory(load_yaml(settings.population_policy_path))
evolution_service = EvolutionService(
    evolution_genome_store,
    promotion_policy,
    load_yaml(settings.population_policy_path),
    factory=evolution_factory,
)


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "live_execution_enabled": settings.live_execution_enabled,
        "kill_switch": kill_switch,
    }


@router.get("/agents")
def list_agents() -> list[AgentGenome]:
    return agent_store.list()


@router.post("/agents", response_model=AgentGenome)
def create_agent(agent: AgentGenome) -> AgentGenome:
    try:
        return agent_store.add(agent)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/agents/{agent_id}", response_model=AgentGenome)
def get_agent(agent_id: str) -> AgentGenome:
    agent = agent_store.get(agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


@router.post("/snapshots", response_model=MarketSnapshot)
def create_snapshot(snapshot: MarketSnapshot) -> MarketSnapshot:
    return snapshot_store_inst.add(snapshot)


@router.get("/snapshots/{snapshot_id}", response_model=MarketSnapshot)
def get_snapshot(snapshot_id: str) -> MarketSnapshot:
    snapshot = snapshot_store_inst.get(snapshot_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return snapshot


@router.post("/risk/evaluate")
def evaluate_risk(proposal: TradeProposal) -> dict:
    return risk_engine.evaluate(proposal, kill_switch=kill_switch).model_dump()


@router.post("/evolution/evaluate-promotion")
def evaluate_promotion(item: ChallengerEvaluation) -> dict:
    return promotion_policy.evaluate(item).model_dump()


@router.post("/execution/orders/{decision_id}")
def submit_execution(decision_id: str) -> dict:
    return execution_gateway.submit(decision_id)


@router.post("/kill-switch/{enabled}")
def set_kill_switch(enabled: bool) -> dict:
    global kill_switch
    kill_switch = enabled
    return {"kill_switch": kill_switch}


@router.post("/runs", response_model=dict)
def create_run(payload: dict) -> dict:
    snapshot_id = payload.get("snapshot_id")
    if not snapshot_id:
        raise HTTPException(status_code=400, detail="snapshot_id is required")
    run = trading_run_service.create(snapshot_id)
    return run.model_dump()


@router.get("/runs", response_model=list[dict])
def list_runs() -> list[dict]:
    return [r.model_dump() for r in trading_run_service.all()]


@router.get("/runs/{run_id}", response_model=dict)
def get_run(run_id: str) -> dict:
    run = trading_run_service.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run.model_dump()


@router.post("/runs/{run_id}/transition/{target}")
def transition_run(run_id: str, target: str, actor: str = "system") -> dict:
    from trading_harness.models import RunState

    try:
        run = trading_run_service.transition(run_id, RunState(target), actor=actor)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return run.model_dump()


@router.post("/runs/{run_id}/decision")
def add_decision(run_id: str, payload: dict) -> dict:
    try:
        run = trading_run_service.add_decision(run_id, payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return run.model_dump()


@router.post("/runs/{run_id}/complete")
def complete_run(run_id: str, payload: dict) -> dict:
    try:
        outcome = RunOutcome(payload.get("outcome", "NO_TRADE"))
        reason = payload.get("reason", "")
        run = trading_run_service.complete(run_id, outcome, reason)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return run.model_dump()


@router.post("/runs/{run_id}/fail")
def fail_run(run_id: str, payload: dict) -> dict:
    error = payload.get("error", "unknown error")
    try:
        run = trading_run_service.fail(run_id, error)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return run.model_dump()


@router.post("/performance", response_model=dict)
def record_performance(record: PerformanceRecord) -> dict:
    return performance_store.add(record).model_dump()


@router.get("/performance", response_model=list[dict])
def list_performance() -> list[dict]:
    return [r.model_dump() for r in performance_store.all()]


@router.get("/performance/summary/run/{run_id}", response_model=list[dict])
def performance_by_run(run_id: str) -> list[dict]:
    return [r.model_dump() for r in performance_store.by_run(run_id)]


@router.get("/performance/summary/agent/{agent_id}", response_model=list[dict])
def performance_by_agent(agent_id: str) -> list[dict]:
    return [r.model_dump() for r in performance_store.by_agent(agent_id)]


@router.get("/performance/summary/snapshot/{snapshot_id}", response_model=list[dict])
def performance_by_snapshot(snapshot_id: str) -> list[dict]:
    return [r.model_dump() for r in performance_store.by_snapshot(snapshot_id)]


@router.get("/audit")
def get_audit_log() -> list[dict]:
    return [e.model_dump() for e in trading_run_service.get_audit_log()]


@router.get("/audit/{entity_id}")
def get_audit_log_for_entity(entity_id: str) -> list[dict]:
    return [e.model_dump() for e in trading_run_service.get_audit_log(entity_id)]


@router.post("/outcomes", response_model=dict)
def create_outcome(record: OutcomeRecord) -> dict:
    return outcome_store.add(record).model_dump()


@router.get("/outcomes", response_model=list[dict])
def list_outcomes() -> list[dict]:
    return [o.model_dump() for o in outcome_store.all()]


@router.get("/outcomes/agent/{agent_id}", response_model=list[dict])
def outcomes_by_agent(agent_id: str) -> list[dict]:
    return [o.model_dump() for o in outcome_store.by_agent(agent_id)]


@router.get("/outcomes/run/{run_id}", response_model=list[dict])
def outcomes_by_run(run_id: str) -> list[dict]:
    return [o.model_dump() for o in outcome_store.by_run(run_id)]


@router.get("/outcomes/regime/{regime}", response_model=list[dict])
def outcomes_by_regime(regime: str) -> list[dict]:
    try:
        m_regime = MarketRegime(regime)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid regime: {regime}")
    return [o.model_dump() for o in outcome_store.by_regime(m_regime)]


@router.post("/evaluation/agent/{agent_id}")
def evaluate_agent(agent_id: str, run_id: str | None = None) -> dict:
    result = evaluation_service.evaluate_agent(agent_id, run_id=run_id)
    return result


@router.post("/evaluation/regime/{agent_id}/{regime}")
def evaluate_regime(agent_id: str, regime: str) -> dict:
    try:
        m_regime = MarketRegime(regime)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid regime: {regime}")
    result = evaluation_service.evaluate_regime_performance(agent_id, m_regime)
    return result


@router.post("/evaluation/drawdown/{agent_id}")
def evaluate_drawdown(agent_id: str, run_id: str | None = None) -> dict:
    result = evaluation_service.evaluate_drawdown(agent_id, run_id=run_id)
    return result


@router.post("/evaluation/out-of-sample")
def evaluate_oos(payload: dict) -> dict:

    def _records_from_dict(items: list[dict]) -> list[OutcomeRecord]:
        return [OutcomeRecord(**item) for item in items]

    train = _records_from_dict(payload.get("train_outcomes", []))
    test = _records_from_dict(payload.get("test_outcomes", []))
    agent_id = payload.get("agent_id", "")
    result = evaluation_service.evaluate_out_of_sample(agent_id, train, test)
    return result


@router.post("/evaluation/walk-forward/{agent_id}")
def evaluate_walk_forward(agent_id: str, payload: dict) -> dict:
    window_size = payload.get("window_size", 50)
    step_size = payload.get("step_size", 10)
    outcomes = outcome_store.by_agent(agent_id)
    result = evaluation_service.evaluate_walk_forward(agent_id, outcomes, window_size, step_size)
    return result


@router.get("/evaluation/results")
def list_evaluation_results() -> list[dict]:
    return [r.model_dump() for r in evaluation_service.get_results()]


@router.get("/evaluation/results/agent/{agent_id}")
def list_evaluation_results_for_agent(agent_id: str) -> list[dict]:
    return [r.model_dump() for r in evaluation_service.get_results(agent_id)]


# ---------------------------------------------------------------------------
# Agent analysis endpoint
# ---------------------------------------------------------------------------

@router.post("/agent/analyze", response_model=AgentAnalysisResult)
async def analyze_agent(
    agent_id: str,
    snapshot_id: str,
    run_id: str | None = None,
) -> AgentAnalysisResult:
    """Run agent analysis on a snapshot and return structured signal."""
    agent = agent_store.get(agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    snapshot = snapshot_store_inst.get(snapshot_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"Snapshot {snapshot_id} not found")
    result = await agent_runtime.analyze(agent, snapshot, run_id=run_id)
    return result


# ---------------------------------------------------------------------------
# Agent analysis query endpoints
# ---------------------------------------------------------------------------

@router.get("/agent/analyses", response_model=list[dict])
def list_agent_analyses() -> list[dict]:
    return [r.model_dump() for r in analysis_store.all()]


@router.get("/agent/analyses/run/{run_id}", response_model=list[dict])
def analyses_by_run(run_id: str) -> list[dict]:
    return [r.model_dump() for r in analysis_store.by_run(run_id)]


@router.get("/agent/analyses/agent/{agent_id}", response_model=list[dict])
def analyses_by_agent(agent_id: str) -> list[dict]:
    return [r.model_dump() for r in analysis_store.by_agent(agent_id)]


@router.get("/agent/analyses/snapshot/{snapshot_id}", response_model=list[dict])
def analyses_by_snapshot(snapshot_id: str) -> list[dict]:
    return [r.model_dump() for r in analysis_store.by_snapshot(snapshot_id)]


# ---------------------------------------------------------------------------
# Phase 3 — Evolution endpoints
# ---------------------------------------------------------------------------

# --- Mutation & Recombination ---


@router.post("/evolution/mutate")
def mutate_agent(payload: dict) -> dict:
    parent_id: str = payload["parent_id"]
    mutation_type = payload.get("mutation_type", "INDICATOR_ADD")
    hypothesized_advantage = payload.get("hypothesized_advantage", "")
    expected_failure_modes = payload.get("expected_failure_modes", [])
    parent = evolution_genome_store.get(parent_id)
    if parent is None:
        raise HTTPException(status_code=404, detail=f"Parent agent {parent_id} not found")
    child, record = evolution_service.generate_mutant(
        parent,
        mutation_type=mutation_type,
        hypothesized_advantage=hypothesized_advantage,
        expected_failure_modes=expected_failure_modes,
    )
    return {
        "child": child.model_dump(),
        "mutation": record.model_dump(),
    }


@router.post("/evolution/recombine")
def recombine_agents(payload: dict) -> dict:
    parent_a_id: str = payload["parent_a_id"]
    parent_b_id: str = payload["parent_b_id"]
    hypothesized_advantage = payload.get("hypothesized_advantage", "")
    expected_failure_modes = payload.get("expected_failure_modes", [])
    parent_a = evolution_genome_store.get(parent_a_id)
    parent_b = evolution_genome_store.get(parent_b_id)
    if parent_a is None or parent_b is None:
        raise HTTPException(
            status_code=404,
            detail=f"Agents {parent_a_id} / {parent_b_id} not found",
        )
    child, record = evolution_service.recombine(
        parent_a,
        parent_b,
        hypothesized_advantage=hypothesized_advantage,
        expected_failure_modes=expected_failure_modes,
    )
    return {
        "child": child.model_dump(),
        "mutation": record.model_dump(),
    }


# --- Challenger Pool ---


@router.post("/evolution/challengers/{agent_id}/add")
def add_challenger(agent_id: str) -> dict:
    agent = evolution_genome_store.get(agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    result = evolution_service.add_challenger(agent)
    return result.model_dump()


@router.get("/evolution/challengers/pairs/{category}", response_model=list[dict])
def get_challenger_pairs(category: str) -> list[dict]:
    return [p.model_dump() for p in evolution_service.get_challenger_pairs(category)]


@router.post("/evolution/challengers/evaluate")
def evaluate_challenger_endpoint(payload: dict) -> dict:
    return evolution_service.evaluate_challenger(
        challenger_id=payload["challenger_id"],
        champion_id=payload["champion_id"],
        category=payload["category"],
        challenger_score=payload["challenger_score"],
        incumbent_score=payload["incumbent_score"],
        observations=payload.get("observations", 0),
        out_of_sample_pass=payload.get("out_of_sample_pass", True),
        walk_forward_pass=payload.get("walk_forward_pass", True),
        shadow_pass=payload.get("shadow_pass", True),
        ensemble_contribution=payload.get("ensemble_contribution", 0.0),
        security_pass=payload.get("security_pass", True),
    ).model_dump()


@router.post("/evolution/challengers/promote")
def promote_challenger_endpoint(payload: dict) -> dict:
    return evolution_service.promote_challenger(
        payload["challenger_id"],
        payload["incumbent_id"],
        payload["category"],
    ).model_dump()


@router.post("/evolution/challengers/demote")
def demote_challenger_endpoint(payload: dict) -> dict:
    agent_id: str = payload["agent_id"]
    reason = payload.get("reason", "PROMOTION_FAILED")
    evolution_service.demote_to_probation(agent_id, reason)
    agent = evolution_genome_store.get(agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    return agent.model_dump()


# --- Hall of Fame ---


@router.post("/evolution/hall-of-fame")
def add_to_hall_of_fame(payload: dict) -> dict:
    hall_of_fame = evolution_service.hall_of_fame
    record = hall_of_fame.add(
        agent_id=payload["agent_id"],
        category=payload["category"],
        score=payload["final_score"],
        observations=payload.get("generation", 0),
    )
    return record.model_dump()


@router.get("/evolution/hall-of-fame", response_model=list[dict])
def list_hall_of_fame() -> list[dict]:
    return [r.model_dump() for r in evolution_service.hall_of_fame.list_all()]


@router.get("/evolution/hall-of-fame/{category}", response_model=list[dict])
def list_hall_of_fame_category(category: str) -> list[dict]:
    return [r.model_dump() for r in evolution_service.hall_of_fame.by_category(category)]


@router.get("/evolution/hall-of-fame/top/{category}", response_model=dict | None)
def get_top_hall_of_fame(category: str) -> dict | None:
    record = evolution_service.hall_of_fame.get_best(category)
    return record.model_dump() if record else None


# --- Graveyard ---


@router.post("/evolution/graveyard")
def add_to_graveyard(payload: dict) -> dict:
    graveyard = evolution_service.graveyard
    record = graveyard.add(
        agent_id=payload["agent_id"],
        category=payload["category"],
        final_score=payload.get("final_score", 0.0),
        reason=payload.get("reason", "UNSPECIFIED"),
    )
    return record.model_dump()


@router.get("/evolution/graveyard", response_model=list[dict])
def list_graveyard() -> list[dict]:
    return [r.model_dump() for r in evolution_service.graveyard.list_all()]


@router.get("/evolution/graveyard/{category}", response_model=list[dict])
def list_graveyard_category(category: str) -> list[dict]:
    return [r.model_dump() for r in evolution_service.graveyard.by_category(category)]


# --- Promotion History & Rollbacks ---


@router.get("/evolution/promotion-history/{category}", response_model=list[dict])
def get_promotion_history(category: str) -> list[dict]:
    return [r.model_dump() for r in evolution_service.get_promotion_history(category)]


@router.get("/evolution/rollbacks", response_model=list[dict])
def list_rollbacks() -> list[dict]:
    return [r.model_dump() for r in evolution_service.get_rollbacks()]


@router.get("/evolution/rollbacks/{agent_id}", response_model=list[dict])
def list_rollbacks_for_agent(agent_id: str) -> list[dict]:
    return [r.model_dump() for r in evolution_service.get_rollbacks_for_agent(agent_id)]


@router.post("/evolution/rollback")
def rollback_agent_status(payload: dict) -> dict:
    try:
        target = AgentStatus(payload["target_status"])
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"Invalid status: {payload.get('target_status')}"
        ) from exc
    entry = evolution_service.rollback_agent_status(
        payload["agent_id"],
        target,
        payload.get("reason", "MANUAL_ROLLBACK"),
    )
    return entry.model_dump()


# --- Population Stats ---


@router.get("/evolution/population-stats/{category}", response_model=dict)
def get_population_stats(category: str) -> dict:
    return evolution_service.get_population_stats(category)


@router.get("/evolution/population-stats", response_model=list[dict])
def list_all_population_stats() -> list[dict]:
    categories = set()
    for agent in evolution_genome_store.list_all():
        categories.add(agent.category)
    return [evolution_service.get_population_stats(cat) for cat in sorted(categories)]


# ---------------------------------------------------------------------------
# Phase 5 — Live Execution endpoints
# ---------------------------------------------------------------------------

from trading_harness.services.credential_manager import CredentialManager
from trading_harness.services.crypto_exchange_adapter import CryptoExecutionRouter
from trading_harness.services.execution_store import ExecutionLogStore
from trading_harness.services.kill_switch import KillSwitch
from trading_harness.services.live_execution_service import (
    ExecutionConfig,
    LiveExecutionService,
)
from trading_harness.services.network_policy import NetworkPolicy
from trading_harness.services.paper_exchange import PaperExchange
from trading_harness.services.paper_exchange_adapter import PaperExchangeAdapter
from trading_harness.services.policy_loader import load_yaml
from trading_harness.services.risk_engine import RiskEngine
from trading_harness.services.shadow_mode_logger import (
    ShadowModeAdapter,
    ShadowModeLogger,
)

execution_log_store = ExecutionLogStore()
# Persistenter Kill Switch: State (inkl. Auto-Trigger) überlebt Prozess-Neustarts
execution_kill_switch = KillSwitch(db_path=settings.kill_switch_state_path)
execution_config = ExecutionConfig(
    live_execution_enabled=settings.live_execution_enabled,
    allowed_endpoints=settings.network_allowed_patterns,
)
risk_engine = RiskEngine(load_yaml(settings.risk_policy_path))

# Network-Policy für Endpoint-Whitelist
network_policy = NetworkPolicy(allowed_patterns=settings.network_allowed_patterns)

# Credential-Manager für API-Schlüssel
credential_manager = CredentialManager()

# PaperExchange-Adapter als erste echte Exchange-Integration.
# Live Execution bleibt standardmäßig deaktiviert — muss explizit aktiviert werden.
_paper_exchange = PaperExchange()
_paper_adapter = PaperExchangeAdapter(paper_exchange=_paper_exchange)

live_execution_service = LiveExecutionService(
    kill_switch=execution_kill_switch,
    rate_limiter=None,
    deduplicator=None,
    exchange_adapter=_paper_adapter,
    risk_engine=risk_engine,
    network_policy=network_policy,
    credential_manager=credential_manager,
    config=execution_config,
)

# Shadow-Mode-Logger für Backtesting ohne echte Order-Ausführung.
_shadow_logger = ShadowModeLogger()
_shadow_adapter = ShadowModeAdapter(delegate=_paper_adapter, shadow=_shadow_logger)

# Crypto-Execution-Router — routet an Bybit, Bitget, Binance oder Coinbase.
# Alle Adapter durchlaufen dieselbe Pipeline (KillSwitch, RateLimiter, …).
_crypto_router = CryptoExecutionRouter(credential_manager=credential_manager)

# Crypto-Execution-Service — nutzt denselben KillSwitch/Lock als Paper-Service
crypto_execution_service = LiveExecutionService(
    kill_switch=execution_kill_switch,
    rate_limiter=None,
    deduplicator=None,
    exchange_adapter=_crypto_router,
    risk_engine=risk_engine,
    network_policy=network_policy,
    credential_manager=credential_manager,
    config=execution_config,
)


@router.post("/execution/orders", dependencies=[Depends(require_trade_key)])
def submit_execution_order(payload: dict) -> dict:
    """Submit execution order (trade API key required)."""
    try:
        return live_execution_service.submit_order(
            decision_id=payload["decision_id"],
            run_id=payload.get("run_id", ""),
            symbol=payload["symbol"],
            side=payload["side"],
            quantity=float(payload["quantity"]),
            price=float(payload["price"]),
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid payload: {exc}") from exc


@router.post(
    "/execution/kill-switch/{enabled}",
    dependencies=[Depends(require_trade_key)],
)
def toggle_execution_kill_switch(enabled: bool) -> dict:
    """Toggle kill switch (trade API key required)."""
    if enabled:
        execution_kill_switch.activate()
    else:
        execution_kill_switch.deactivate()
    return {"kill_switch": execution_kill_switch.is_active()}


@router.get("/execution/status", dependencies=[Depends(require_read_key)])
def get_execution_status() -> dict:
    """Execution status (read API key required)."""
    return {
        "live_execution_enabled": live_execution_service.is_live_enabled,
        "kill_switch": execution_kill_switch.is_active(),
        "execution_logs_count": execution_log_store.count,
    }


@router.get("/execution/logs", dependencies=[Depends(require_read_key)])
def get_execution_logs(decision_id: str | None = None) -> list[dict]:
    """Execution logs (read API key required)."""
    return live_execution_service.get_logs(decision_id=decision_id)


# ---------------------------------------------------------------------------
# Shadow Mode endpoints — Backtesting ohne echte Order-Ausführung
# ---------------------------------------------------------------------------


@router.post(
    "/execution/shadow/submit",
    dependencies=[Depends(require_trade_key)],
)
def shadow_submit_order(payload: dict) -> dict:
    """Shadow-Mode order submit — loggt Order mit simulierte Fill, keine Ausführung."""
    try:
        result = _shadow_logger.submit_order(
            symbol=payload["symbol"],
            side=payload["side"],
            quantity=float(payload["quantity"]),
            price=float(payload["price"]),
            order_type=payload.get("order_type", "MARKET"),
        )
        return {
            "shadow_order_id": result["order_id"],
            "status": result["status"],
            "filled_price": result["filled_price"],
            "slippage": result["slippage"],
            "commission": result["commission"],
        }
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid payload: {exc}") from exc


@router.get(
    "/execution/shadow/summary",
    dependencies=[Depends(require_read_key)],
)
def shadow_summary() -> dict:
    """Shadow-Mode Zusammenfassung aller geloggten Shadow-Orders."""
    return _shadow_logger.summary()


@router.get(
    "/execution/shadow/records",
    dependencies=[Depends(require_read_key)],
)
def shadow_records(
    decision_id: str | None = None,
    symbol: str | None = None,
    run_id: str | None = None,
) -> list[dict]:
    """Shadow-Mode Records abrufen (optional gefiltert)."""
    records = _shadow_logger.get_records(
        decision_id=decision_id,
        symbol=symbol,
        run_id=run_id,
    )
    return [r.model_dump() for r in records]


# ---------------------------------------------------------------------------
# Crypto Submit — unified endpoint durch LiveExecutionService-Pipeline
# ---------------------------------------------------------------------------


@router.post(
    "/execution/crypto/submit",
    dependencies=[Depends(require_trade_key)],
)
def crypto_submit(payload: dict) -> dict:
    """Order durch die volle Pipeline (KillSwitch, RateLimiter, RiskEngine, …)."""
    try:
        return crypto_execution_service.submit_order(
            decision_id=payload["decision_id"],
            run_id=payload.get("run_id", ""),
            symbol=payload["symbol"],
            side=payload["side"],
            quantity=float(payload["quantity"]),
            price=float(payload["price"]),
            exchange_name=payload.get("exchange_name"),
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid payload: {exc}") from exc


@router.get(
    "/execution/crypto/price/{symbol}",
    dependencies=[Depends(require_read_key)],
)
def crypto_get_price(symbol: str) -> dict:
    """Live-Preis vom Crypto-Router abrufen."""
    result = _crypto_router.get_ticker(symbol)
    return {"symbol": symbol, **result}


@router.get(
    "/execution/crypto/status/{order_id}",
    dependencies=[Depends(require_read_key)],
)
def crypto_get_order_status(order_id: str, exchange_name: str | None = None) -> dict:
    """Order-Status über die Pipeline abrufen."""
    return crypto_execution_service.get_order_status(
        order_id=order_id,
        exchange_name=exchange_name,
    )


@router.delete(
    "/execution/crypto/cancel/{order_id}",
    dependencies=[Depends(require_trade_key)],
)
def crypto_cancel_order(order_id: str, exchange_name: str | None = None) -> dict:
    """Order stornieren über die Pipeline."""
    return crypto_execution_service.cancel_order(
        order_id=order_id,
        exchange_name=exchange_name,
    )


# ---------------------------------------------------------------------------
# Crypto Status — zeigt welche Adapter simuliert
# ---------------------------------------------------------------------------


@router.get(
    "/execution/crypto/status",
    dependencies=[Depends(require_read_key)],
)
def crypto_status() -> dict:
    """Crypto-Router Status — zeigt Credentials für alle Exchanges."""
    credential_states: dict[str, str] = {}
    for exchange in CryptoExecutionRouter.SUPPORTED:
        prefixes = CryptoExecutionRouter.CREDENTIAL_PREFIXES.get(exchange, ("", ""))
        has_key = bool(credential_manager.get(prefixes[0])) if credential_manager else False
        has_secret = bool(credential_manager.get(prefixes[1])) if credential_manager else False
        credential_states[exchange] = "LIVE" if (has_key and has_secret) else "SIMULATED"
    return {
        "router_active": True,
        "supported_exchanges": list(CryptoExecutionRouter.SUPPORTED),
        "credential_states": credential_states,
        "shadow_mode_active": True,
    }
