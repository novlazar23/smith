from __future__ import annotations

from fastapi import APIRouter, HTTPException

from trading_harness.config import get_settings
from trading_harness.models import (
    AgentAnalysisResult,
    AgentGenome,
    ChallengerEvaluation,
    MarketRegime,
    MarketSnapshot,
    OutcomeRecord,
    PerformanceRecord,
    RunOutcome,
    TradeProposal,
)
from trading_harness.services.agent_registry import AgentRegistry
from trading_harness.services.agent_runtime import AgentRuntime
from trading_harness.services.db import Database
from trading_harness.services.evaluation import EvaluationService, OutcomeGenerator
from trading_harness.services.evolution import PromotionPolicy
from trading_harness.services.execution_gateway import ExecutionGateway
from trading_harness.services.orchestrator import TradingRunService
from trading_harness.services.performance import PerformanceStore
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
performance_store = PerformanceStore()
outcome_generator = OutcomeGenerator()
evaluation_service = EvaluationService(outcome_generator, performance_store)
agent_runtime = AgentRuntime()


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


@router.get("/audit")
def get_audit_log() -> list[dict]:
    return [e.model_dump() for e in trading_run_service.get_audit_log()]


@router.get("/audit/{entity_id}")
def get_audit_log_for_entity(entity_id: str) -> list[dict]:
    return [e.model_dump() for e in trading_run_service.get_audit_log(entity_id)]


@router.post("/outcomes", response_model=dict)
def create_outcome(record: OutcomeRecord) -> dict:
    return outcome_generator.add(record).model_dump()


@router.get("/outcomes", response_model=list[dict])
def list_outcomes() -> list[dict]:
    return [o.model_dump() for o in outcome_generator.all()]


@router.get("/outcomes/agent/{agent_id}", response_model=list[dict])
def outcomes_by_agent(agent_id: str) -> list[dict]:
    return [o.model_dump() for o in outcome_generator.by_agent(agent_id)]


@router.get("/outcomes/run/{run_id}", response_model=list[dict])
def outcomes_by_run(run_id: str) -> list[dict]:
    return [o.model_dump() for o in outcome_generator.by_run(run_id)]


@router.get("/outcomes/regime/{regime}", response_model=list[dict])
def outcomes_by_regime(regime: str) -> list[dict]:
    try:
        m_regime = MarketRegime(regime)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid regime: {regime}")
    return [o.model_dump() for o in outcome_generator.by_regime(m_regime)]


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
    outcomes = outcome_generator.by_agent(agent_id)
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
