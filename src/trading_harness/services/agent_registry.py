from __future__ import annotations

from threading import RLock

from trading_harness.models import AgentGenome


class AgentRegistry:
    """Thread-safe in-memory registry for the MVP.

    Replace with PostgreSQL persistence before any production deployment.
    """

    def __init__(self) -> None:
        self._agents: dict[str, AgentGenome] = {}
        self._lock = RLock()

    def list(self) -> list[AgentGenome]:
        with self._lock:
            return list(self._agents.values())

    def get(self, agent_id: str) -> AgentGenome | None:
        with self._lock:
            return self._agents.get(agent_id)

    def add(self, agent: AgentGenome) -> AgentGenome:
        with self._lock:
            if agent.id in self._agents:
                raise ValueError(f"Agent {agent.id} already exists")
            self._agents[agent.id] = agent
            return agent
