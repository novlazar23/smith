from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from trading_harness.models import MarketRegime, OutcomeRecord
from trading_harness.services.db import Database


def _compute_single_mfe_mae(
    direction_predicted: str,
    direction_actual: str,
    entry_price: float,
    exit_price: float,
) -> tuple[float, float]:
    """Compute MFE/MAE for a single outcome."""
    if direction_predicted.upper() in ("LONG", "BUY"):
        price_return = (exit_price - entry_price) / entry_price
    elif direction_predicted.upper() in ("SHORT", "SELL"):
        price_return = (entry_price - exit_price) / entry_price
    else:
        price_return = 0.0

    mfe = price_return if price_return > 0 else 0.0
    mae = abs(price_return) if price_return < 0 else 0.0
    return mfe, mae


def _record_to_row(record: OutcomeRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "prediction_id": record.prediction_id,
        "agent_id": record.agent_id,
        "run_id": record.run_id,
        "snapshot_id": record.snapshot_id,
        "symbol": record.symbol,
        "direction_predicted": record.direction_predicted,
        "direction_actual": record.direction_actual,
        "confidence_predicted": record.confidence_predicted,
        "entry_price": record.entry_price,
        "exit_price": record.exit_price,
        "mfe": record.mfe,
        "mae": record.mae,
        "holding_period_bars": record.holding_period_bars,
        "realized_pnl": record.realized_pnl,
        "regime": record.regime.value,
        "timestamp": record.timestamp.isoformat(),
    }


def _row_to_record(row: dict[str, Any]) -> OutcomeRecord:
    regime_raw = row.get("regime", "unknown")
    try:
        regime = MarketRegime(regime_raw)
    except ValueError:
        regime = MarketRegime.UNKNOWN
    return OutcomeRecord(
        id=row["id"],
        prediction_id=row["prediction_id"],
        agent_id=row["agent_id"],
        run_id=row["run_id"],
        snapshot_id=row["snapshot_id"],
        symbol=row["symbol"],
        direction_predicted=row["direction_predicted"],
        direction_actual=row["direction_actual"],
        confidence_predicted=row["confidence_predicted"],
        entry_price=row["entry_price"],
        exit_price=row["exit_price"],
        mfe=row.get("mfe", 0.0),
        mae=row.get("mae", 0.0),
        holding_period_bars=row.get("holding_period_bars", 0),
        realized_pnl=row.get("realized_pnl", 0.0),
        regime=regime,
        timestamp=datetime.fromisoformat(str(row["timestamp"])).replace(tzinfo=UTC),
    )


class PersistedOutcomeStore:
    """PostgreSQL-backed store for outcome records.

    Falls back to in-memory store when PostgreSQL is unavailable.
    """

    def __init__(self, db: Database | None = None) -> None:
        self._db = db
        self._fallback: dict[str, OutcomeRecord] = {}

    def add(self, record: OutcomeRecord) -> OutcomeRecord:
        if self._db and self._db.is_available:
            row = _record_to_row(record)
            cols = list(row.keys())
            placeholders = ",".join(["%s"] * len(cols))
            self._db.execute_write(
                f"INSERT INTO outcomes ({','.join(cols)}) "
                f"VALUES ({placeholders}) "
                "ON CONFLICT (id) DO UPDATE SET "
                + ",".join(f"{c} = EXCLUDED.{c}" for c in cols if c != "id"),
                *[row[c] for c in cols],
            )
        else:
            self._fallback[record.id] = record
        return record

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
        """Create an outcome record and persist it (with MFE/MAE computation)."""
        from trading_harness.models import OutcomeRecord as _OR

        if entry_price <= 0 or exit_price <= 0:
            raise ValueError("entry_price and exit_price must be positive")

        direction_actual_normalized = direction_actual.upper()
        direction_predicted_normalized = direction_predicted.upper()

        mfe, mae = _compute_single_mfe_mae(
            direction_predicted_normalized,
            direction_actual_normalized,
            entry_price,
            exit_price,
        )

        record = _OR(
            id=f"outcome-{prediction_id}",
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
        return self.add(record)

    def get(self, outcome_id: str) -> OutcomeRecord | None:
        if self._db and self._db.is_available:
            rows = self._db.execute(
                "SELECT * FROM outcomes WHERE id = %s", (outcome_id,)
            )
            if rows:
                return _row_to_record(rows[0])
            return None
        return self._fallback.get(outcome_id)

    def by_agent(self, agent_id: str) -> list[OutcomeRecord]:
        if self._db and self._db.is_available:
            rows = self._db.execute(
                "SELECT * FROM outcomes WHERE agent_id = %s ORDER BY timestamp",
                (agent_id,),
            )
            return [_row_to_record(r) for r in rows]
        return [r for r in self._fallback.values() if r.agent_id == agent_id]

    def by_run(self, run_id: str) -> list[OutcomeRecord]:
        if self._db and self._db.is_available:
            rows = self._db.execute(
                "SELECT * FROM outcomes WHERE run_id = %s ORDER BY timestamp",
                (run_id,),
            )
            return [_row_to_record(r) for r in rows]
        return [r for r in self._fallback.values() if r.run_id == run_id]

    def by_regime(self, regime: MarketRegime) -> list[OutcomeRecord]:
        if self._db and self._db.is_available:
            rows = self._db.execute(
                "SELECT * FROM outcomes WHERE regime = %s ORDER BY timestamp",
                (regime.value,),
            )
            return [_row_to_record(r) for r in rows]
        return [r for r in self._fallback.values() if r.regime == regime]

    def all(self) -> list[OutcomeRecord]:
        if self._db and self._db.is_available:
            rows = self._db.execute(
                "SELECT * FROM outcomes ORDER BY timestamp"
            )
            return [_row_to_record(r) for r in rows]
        return list(self._fallback.values())

    def get_by_prediction_id(self, prediction_id: str) -> OutcomeRecord | None:
        """Look up by prediction_id (not stored as PK)."""
        if self._db and self._db.is_available:
            rows = self._db.execute(
                "SELECT * FROM outcomes WHERE prediction_id = %s", (prediction_id,)
            )
            if rows:
                return _row_to_record(rows[0])
            return None
        for o in self._fallback.values():
            if o.prediction_id == prediction_id:
                return o
        return None