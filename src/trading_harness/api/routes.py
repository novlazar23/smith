from __future__ import annotations

from fastapi import APIRouter, HTTPException

from trading_harness.config import get_settings
from trading_harness.models import (
    AgentGenome,
    ChallengerEvaluation,
    MarketSnapshot,
    TradeProposal,
)
from trading_harness.services.agent_registry import AgentRegistry
from trading_harness.services.evolution import PromotionPolicy
from trading_harness.services.execution_gateway import ExecutionGateway
from trading_harness.services.policy_loader import load_yaml
from trading_harness.services.risk_engine import RiskEngine
from trading_harness.services.snapshot_store import SnapshotStore

router = APIRouter()
settings = get_settings()

agent_registry = AgentRegistry()
snapshot_store = SnapshotStore()
risk_engine = RiskEngine(load_yaml(settings.risk_policy_path))
promotion_policy = PromotionPolicy(load_yaml(settings.population_policy_path))
execution_gateway = ExecutionGateway(settings.live_execution_enabled)
kill_switch = settings.kill_switch_default


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "live_execution_enabled": settings.live_execution_enabled,
        "kill_switch": kill_switch,
    }


@router.get("/agents")
def list_agents() -> list[AgentGenome]:
    return agent_registry.list()


@router.post("/agents", response_model=AgentGenome)
def create_agent(agent: AgentGenome) -> AgentGenome:
    try:
        return agent_registry.add(agent)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/agents/{agent_id}", response_model=AgentGenome)
def get_agent(agent_id: str) -> AgentGenome:
    agent = agent_registry.get(agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


@router.post("/snapshots", response_model=MarketSnapshot)
def create_snapshot(snapshot: MarketSnapshot) -> MarketSnapshot:
    return snapshot_store.add(snapshot)


@router.get("/snapshots/{snapshot_id}", response_model=MarketSnapshot)
def get_snapshot(snapshot_id: str) -> MarketSnapshot:
    snapshot = snapshot_store.get(snapshot_id)
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
