from __future__ import annotations

from threading import RLock
from typing import Protocol

from trading_harness.models import MarketRegime, OutcomeRecord


class _OutcomeStoreProto(Protocol):
    """Protocol for outcome stores used by OutcomeGenerator."""

    def add(self, record: OutcomeRecord) -> OutcomeRecord: ...
    def get(self, outcome_id: str) -> OutcomeRecord | None: ...
    def by_agent(self, agent_id: str) -> list[OutcomeRecord]: ...
    def by_run(self, run_id: str) -> list[OutcomeRecord]: ...
    def by_regime(self, regime: MarketRegime) -> list[OutcomeRecord]: ...
    def all(self) -> list[OutcomeRecord]: ...


class InMemoryOutcomeStore:
    """Thread-safe in-memory outcome store for testing and fallback."""

    def __init__(self) -> None:
        self._outcomes: dict[str, OutcomeRecord] = {}
        self._lock = RLock()

    def add(self, record: OutcomeRecord) -> OutcomeRecord:
        with self._lock:
            self._outcomes[record.id] = record
        return record

    def get(self, outcome_id: str) -> OutcomeRecord | None:
        with self._lock:
            return self._outcomes.get(outcome_id)

    def by_agent(self, agent_id: str) -> list[OutcomeRecord]:
        with self._lock:
            return [o for o in self._outcomes.values() if o.agent_id == agent_id]

    def by_run(self, run_id: str) -> list[OutcomeRecord]:
        with self._lock:
            return [o for o in self._outcomes.values() if o.run_id == run_id]

    def by_regime(self, regime: MarketRegime) -> list[OutcomeRecord]:
        with self._lock:
            return [o for o in self._outcomes.values() if o.regime == regime]

    def all(self) -> list[OutcomeRecord]:
        with self._lock:
            return list(self._outcomes.values())


class OutcomeGenerator:
    """Generates outcome records from predictions and actual market data.

    Computes MFE/MAE from prediction and actual market movements.
    Stores outcomes via the provided store (in-memory or persisted).
    """

    def __init__(self, store: _OutcomeStoreProto | None = None) -> None:
        self._store = store or InMemoryOutcomeStore()

    @property
    def store(self) -> _OutcomeStoreProto:
        return self._store

    def generate(
        self,
        *,
        prediction_id: str,
        agent_id: str,
        run_id: str,
        snapshot_id: str,
        symbol: str,
        direction_predicted: str,
        direction_actual: str,
        confidence_predicted: float,
        entry_price: float,
        exit_price: float,
        holding_period_bars: int = 0,
        realized_pnl: float = 0.0,
        regime: MarketRegime = MarketRegime.UNKNOWN,
    ) -> OutcomeRecord:
        if entry_price <= 0 or exit_price <= 0:
            raise ValueError("entry_price and exit_price must be positive")

        direction_actual_normalized = direction_actual.upper()
        direction_predicted_normalized = direction_predicted.upper()

        price_return = self._compute_return(
            direction_predicted_normalized,
            entry_price,
            exit_price,
        )

        mfe = price_return if price_return > 0 else 0.0
        mae = abs(price_return) if price_return < 0 else 0.0

        outcome = OutcomeRecord(
            prediction_id=prediction_id,
            agent_id=agent_id,
            run_id=run_id,
            snapshot_id=snapshot_id,
            symbol=symbol,
            direction_predicted=direction_predicted_normalized,
            direction_actual=direction_actual_normalized,
            confidence_predicted=confidence_predicted,
            entry_price=entry_price,
            exit_price=exit_price,
            mfe=mfe,
            mae=mae,
            holding_period_bars=holding_period_bars,
            realized_pnl=realized_pnl,
            regime=regime,
        )

        self._store.add(outcome)
        return outcome

    def add(self, outcome: OutcomeRecord) -> OutcomeRecord:
        return self._store.add(outcome)

    def get(self, outcome_id: str) -> OutcomeRecord | None:
        return self._store.get(outcome_id)

    def by_agent(self, agent_id: str) -> list[OutcomeRecord]:
        return self._store.by_agent(agent_id)

    def by_run(self, run_id: str) -> list[OutcomeRecord]:
        return self._store.by_run(run_id)

    def by_regime(self, regime: MarketRegime) -> list[OutcomeRecord]:
        return self._store.by_regime(regime)

    def all(self) -> list[OutcomeRecord]:
        return self._store.all()

    def get_by_prediction_id(self, prediction_id: str) -> OutcomeRecord | None:
        for o in self._store.all():
            if o.prediction_id == prediction_id:
                return o
        return None

    @staticmethod
    def _compute_return(
        direction_predicted: str,
        entry_price: float,
        exit_price: float,
    ) -> float:
        if direction_predicted in ("LONG", "BUY"):
            return (exit_price - entry_price) / entry_price
        if direction_predicted in ("SHORT", "SELL"):
            return (entry_price - exit_price) / entry_price
        return 0.0