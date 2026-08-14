from __future__ import annotations

from threading import RLock

from trading_harness.models import MarketRegime, OutcomeRecord


class OutcomeGenerator:
    """Generates outcome records from predictions and actual market data.

    Given a set of performance records (predictions) and actual market data
    (exit prices, realized PnL, etc.), produces OutcomeRecord instances that
    can be used for downstream evaluation metrics.

    NOTE: This class is for test use only. Production code should use
    PersistedOutcomeStore in outcome_store.py.
    """

    def __init__(self) -> None:
        self._outcomes: dict[str, OutcomeRecord] = {}
        self._lock = RLock()

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
        """Create an outcome record and compute MFE/MAE."""
        if entry_price <= 0 or exit_price <= 0:
            raise ValueError("entry_price and exit_price must be positive")

        direction_actual_normalized = direction_actual.upper()
        direction_predicted_normalized = direction_predicted.upper()

        # Compute return
        if direction_predicted_normalized in ("LONG", "BUY"):
            price_return = (exit_price - entry_price) / entry_price
        elif direction_predicted_normalized in ("SHORT", "SELL"):
            price_return = (entry_price - exit_price) / entry_price
        else:
            price_return = 0.0

        # Compute MFE (Maximum Favorable Excursion)
        # Simplified: using the return as a proxy
        if price_return > 0:
            mfe = price_return
        else:
            mfe = 0.0

        # Compute MAE (Maximum Adverse Excursion)
        if price_return < 0:
            mae = abs(price_return)
        else:
            mae = 0.0

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

        with self._lock:
            self._outcomes[outcome.id] = outcome

        return outcome

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

    def add(self, outcome: OutcomeRecord) -> OutcomeRecord:
        with self._lock:
            self._outcomes[outcome.id] = outcome
        return outcome

    def get_by_prediction_id(self, prediction_id: str) -> OutcomeRecord | None:
        with self._lock:
            for o in self._outcomes.values():
                if o.prediction_id == prediction_id:
                    return o
            return None