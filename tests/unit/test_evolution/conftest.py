"""Helfer für die Evolutions-Pipeline-Tests (deterministisch, ohne Netzwerk)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest
from apps.evolution.evaluate import FeedFactory
from apps.evolution.models import DecisionRule, Hypothesis, TestPlan, Variant
from apps.evolution.state import EvolutionStore
from packages.backtesting.core import Candle

BASE_TIME = datetime(2024, 1, 1, tzinfo=UTC)

#: Valider Mechanismus-Code (besteht Jail + Smoke-Test; kein ``__future__``-Import).
VALID_CODE = '''"""Test-Mechanismus: BUY auf Aufwaerts-Kerze (synthetisch)."""

from typing import ClassVar

from packages.backtesting.core import Candle
from packages.backtesting.strategies import SignalAction, StrategySignal

from ._common import RuleStrategy


class TestStrategyStrategy(RuleStrategy):
    strategy_name = "test_strategy"
    description = "Synthetischer Test-Mechanismus fuer die Evolution-Unit-Tests."
    param_specs: ClassVar[dict[str, tuple[float, float, float]]] = {"threshold": (0.0, -10.0, 10.0)}
    min_bars = 60

    def _evaluate(self, candle: Candle) -> StrategySignal | None:
        closes = self._arrays()[3]
        if len(closes) < 2:
            return None
        prev = float(closes[-2])
        curr = float(closes[-1])
        if prev <= 0.0:
            return None
        change_pct = (curr - prev) / prev * 100.0
        if change_pct > float(self.params["threshold"]):
            return self._signal(candle, SignalAction.BUY, 0.5, "Aufwaerts-Kerze")
        return None
'''


def make_candles(
    n: int,
    start: datetime = BASE_TIME,
    symbol: str = "BTC/USDT",
    price0: float = 100.0,
    seed: int = 7,
    step_minutes: int = 5,
) -> list[Candle]:
    """Erzeugt n deterministische 5m-Kerzen (Random-Walk mit festem Seed)."""
    rng = np.random.default_rng(seed)
    candles: list[Candle] = []
    price = price0
    for i in range(n):
        shock = float(rng.normal(0.0, 0.004))
        close = max(1.0, price * (1.0 + shock))
        candles.append(
            Candle(
                timestamp=start + timedelta(minutes=step_minutes * i),
                symbol=symbol,
                open=price,
                high=max(price, close) * 1.0005,
                low=min(price, close) * 0.9995,
                close=close,
                volume=1000.0,
            )
        )
        price = close
    return candles


def make_feed_factory() -> FeedFactory:
    """FeedFactory, der 5m-Kerzen für das angeforderte [start, end)-Fenster generiert."""

    def feed_factory(
        instrument: str,
        start_iso: str | None,
        end_iso: str | None,
        resample: str | None,
    ) -> list[Candle]:
        start = datetime.fromisoformat(start_iso) if start_iso else BASE_TIME
        if start.tzinfo is None:
            start = start.replace(tzinfo=UTC)
        end = datetime.fromisoformat(end_iso) if end_iso else start + timedelta(days=3)
        if end.tzinfo is None:
            end = end.replace(tzinfo=UTC)
        count = max(0, int((end - start).total_seconds() // 300))
        return make_candles(count, start=start, symbol=instrument)

    return feed_factory


def make_hypothesis(
    family: str = "rsi_mean_reversion",
    *,
    kind: Literal["config", "mechanism"] = "config",
    status: str = "proposed",
    params: dict[str, float] | None = None,
    code: str | None = None,
    test_plan: TestPlan | None = None,
    decision_rule: DecisionRule | None = None,
    claim: str = "Test-Claim mit mindestens zehn Zeichen Laenge.",
    hid: str | None = None,
) -> Hypothesis:
    """Baut eine Hypothese für Tests (ID und Zeitstempel sind fix, nicht „now")."""
    variant: dict[str, Any] = {"strategy": family}
    if params is not None:
        variant["params"] = params
    if code is not None:
        variant["code"] = code
        variant["code_file"] = f"packages/strategies/{family}.py"
    return Hypothesis(
        id=hid or f"{family}-20240101-1",
        created_at="2024-01-01T00:00:00+00:00",
        family=family,
        kind=kind,
        claim=claim,
        variant=Variant(**variant),
        test_plan=test_plan if test_plan is not None else TestPlan(),
        decision_rule=decision_rule if decision_rule is not None else DecisionRule(),
        status=status,
    )


@pytest.fixture
def store(tmp_path: Path) -> EvolutionStore:
    """Isolierte Evolutions-State-Instanz unter tmp_path."""
    return EvolutionStore(tmp_path / "evolution")


@pytest.fixture
def small_plan() -> TestPlan:
    """Kurze 3-Tage-Fenster (864 Kerzen je Fenster) mit niedriger Mindestkerzenzahl."""
    return TestPlan(
        instruments=("BTC/USDT",),
        timeframe="5m",
        calibration_start="2024-01-01",
        calibration_end="2024-01-04",
        oos_start="2024-01-05",
        oos_end="2024-01-08",
        min_candles=100,
    )
