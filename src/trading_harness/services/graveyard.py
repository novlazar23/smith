from __future__ import annotations

from trading_harness.models import GraveyardRecord


class Graveyard:
    """Tracks retired or failed agents with their final metrics."""

    def __init__(self) -> None:
        self._entries: list[GraveyardRecord] = []
        self._by_agent: dict[str, GraveyardRecord] = {}

    def add(
        self,
        agent_id: str,
        category: str,
        final_score: float,
        reason: str,
    ) -> GraveyardRecord:
        record = GraveyardRecord(
            agent_id=agent_id,
            category=category,
            final_score=final_score,
            reason=reason,
        )
        self._entries.append(record)
        self._by_agent[agent_id] = record
        return record

    def get_entries(self) -> list[GraveyardRecord]:
        return list(self._entries)

    def get_by_agent(self, agent_id: str) -> GraveyardRecord | None:
        return self._by_agent.get(agent_id)

    def get_by_category(self, category: str) -> list[GraveyardRecord]:
        return [r for r in self._entries if r.category == category]

    def list_all(self) -> list[GraveyardRecord]:
        return self.get_entries()

    def by_category(self, category: str) -> list[GraveyardRecord]:
        return self.get_by_category(category)