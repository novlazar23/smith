from __future__ import annotations

import logging

from trading_harness.models import HallOfFameRecord

logger = logging.getLogger(__name__)


class HallOfFame:
    """Tracks top-performing agents across all categories."""

    def __init__(self, max_entries: int = 100) -> None:
        self._entries: list[HallOfFameRecord] = []
        self._max_entries = max_entries
        self._by_agent: dict[str, HallOfFameRecord] = {}

    def add(
        self,
        agent_id: str,
        category: str,
        score: float,
        observations: int,
        reason: str = "",
    ) -> HallOfFameRecord:
        if agent_id in self._by_agent:
            existing = self._by_agent[agent_id]
            if score > existing.score:
                existing.score = score
                existing.observations = observations
                existing.reason = reason
                return existing
            return existing
        record = HallOfFameRecord(
            agent_id=agent_id,
            category=category,
            score=score,
            observations=observations,
            reason=reason,
        )
        self._entries.append(record)
        self._by_agent[agent_id] = record
        self._entries.sort(key=lambda r: r.score, reverse=True)
        self._entries = self._entries[: self._max_entries]
        return record

    def get_entries(self) -> list[HallOfFameRecord]:
        return list(self._entries)

    def get_by_agent(self, agent_id: str) -> HallOfFameRecord | None:
        return self._by_agent.get(agent_id)

    def get_top_n(self, n: int) -> list[HallOfFameRecord]:
        return self._entries[:n]

    def get_by_category(self, category: str) -> list[HallOfFameRecord]:
        return [r for r in self._entries if r.category == category]

    # Aliases for routes.py compatibility
    def list_all(self) -> list[HallOfFameRecord]:
        return self.get_entries()

    def get_best(self, category: str | None = None) -> HallOfFameRecord | None:
        if category:
            cat_entries = self.get_by_category(category)
            return max(cat_entries, key=lambda r: r.score) if cat_entries else None
        return self.get_top_n(1)[0] if self._entries else None

    def by_category(self, category: str) -> list[HallOfFameRecord]:
        return self.get_by_category(category)