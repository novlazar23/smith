from __future__ import annotations


class ExecutionGateway:
    """Fail-closed execution boundary.

    Real exchange adapters must be implemented in a separate, isolated service.
    """

    def __init__(self, live_enabled: bool = False) -> None:
        self.live_enabled = live_enabled
        self._seen_decisions: set[str] = set()

    def submit(self, decision_id: str) -> dict:
        if not self.live_enabled:
            return {"status": "REJECTED", "reason": "LIVE_EXECUTION_DISABLED"}

        if decision_id in self._seen_decisions:
            return {"status": "REJECTED", "reason": "DUPLICATE_DECISION_ID"}

        self._seen_decisions.add(decision_id)
        return {
            "status": "REJECTED",
            "reason": "NO_EXCHANGE_ADAPTER_IMPLEMENTED",
        }
