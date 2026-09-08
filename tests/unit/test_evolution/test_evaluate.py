"""Tests für Backtest-Matrix, deterministischen Judge und Bootstrap-CI."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from apps.evolution.evaluate import (
    DEFLATION_FAMILY_THRESHOLD,
    block_bootstrap_ci,
    daily_returns,
    judge,
    run_variant_matrix,
    run_window,
)
from apps.evolution.models import Hypothesis, TestPlan, Variant
from packages.backtesting.core import Candle
from tests.unit.test_evolution.conftest import (
    BASE_TIME,
    make_candles,
    make_feed_factory,
    make_hypothesis,
)


def test_run_window_rejects_empty_candles() -> None:
    with pytest.raises(ValueError, match="keine Kerzen"):
        run_window([], "rsi_mean_reversion", {})


def test_run_window_returns_full_metric_set() -> None:
    result = run_window(make_candles(500), "rsi_mean_reversion", {})
    assert set(result) == {
        "return_pct",
        "max_dd_pct",
        "legs",
        "win_rate",
        "n_candles",
        "sharpe_ratio",
        "profit_factor",
    }
    assert result["n_candles"] == 500
    assert result["legs"] >= 0
    if result["win_rate"] is not None:
        assert 0.0 <= result["win_rate"] <= 1.0


def _candles_for_closes(closes: list[tuple[int, float]]) -> list[Candle]:
    """Eine Kerze pro (Tageversatz, Close) — der Close ist der Tages-Endwert."""
    return [
        Candle(
            timestamp=BASE_TIME + timedelta(days=day),
            symbol="BTC/USDT",
            open=price,
            high=price,
            low=price,
            close=price,
        )
        for day, price in closes
    ]


def test_daily_returns_computes_close_to_close_per_utc_day() -> None:
    candles = _candles_for_closes([(0, 100.0), (1, 110.0), (2, 105.0)])
    assert daily_returns(candles) == [pytest.approx(0.1), pytest.approx(105.0 / 110.0 - 1.0)]


def test_daily_returns_single_day_is_empty() -> None:
    candles = _candles_for_closes([(0, 100.0), (0, 101.0)])
    assert daily_returns(candles) == []


def test_block_bootstrap_ci_none_when_not_enough_daily_returns() -> None:
    assert block_bootstrap_ci([[0.01] * 3]) is None
    assert block_bootstrap_ci([[], [0.01] * 4]) is None


def test_block_bootstrap_ci_is_deterministic_and_ordered() -> None:
    daily = [0.01, -0.005, 0.002, 0.01, -0.001, 0.003, 0.002, -0.002]
    first = block_bootstrap_ci([daily], n_iterations=200)
    second = block_bootstrap_ci([daily], n_iterations=200)
    assert first == second
    assert first is not None
    assert first["ci_low_pct"] <= first["mean_pct"] <= first["ci_high_pct"]
    assert first["n_iterations"] == 200
    assert first["seed"] == 42


def test_block_bootstrap_ci_drops_short_assets() -> None:
    daily = [0.01, -0.005, 0.002, 0.01, -0.001, 0.003, 0.002, -0.002]
    result = block_bootstrap_ci([daily, [0.01]], n_iterations=100)
    assert result is not None


def _oos_matrix(
    variant: float = 10.0,
    baseline: float = 0.0,
    legs: int = 20,
    dd: float = 2.0,
    positive: int = 2,
    n: int = 2,
) -> dict[str, Any]:
    return {
        "windows": {},
        "portfolio": {
            "oos": {
                "variant_mean_pct": variant,
                "baseline_mean_pct": baseline,
                "legs": legs,
                "positive_windows": positive,
                "max_dd_max_pct": dd,
                "n": n,
            }
        },
        "bootstrap": None,
        "data_gaps": [],
    }


def _hypothesis() -> Hypothesis:
    return make_hypothesis()


def test_judge_promotes_when_all_rules_hold() -> None:
    verdict = judge(_hypothesis(), _oos_matrix(), n_family=0)
    assert verdict["decision"] == "promoted"
    assert verdict["reasons"] == []
    assert verdict["delta_pp"] == pytest.approx(10.0)
    assert verdict["deflated"] is False


def test_judge_rejects_when_margin_missing() -> None:
    verdict = judge(_hypothesis(), _oos_matrix(variant=0.5), n_family=0)
    assert verdict["decision"] == "rejected"
    assert verdict["reasons"][0].startswith("OOS-Marge")
    assert "deflatiert" not in verdict["reasons"][0]


def test_judge_deflates_margin_at_family_threshold() -> None:
    weak = _oos_matrix(variant=1.5)
    assert judge(_hypothesis(), weak, n_family=DEFLATION_FAMILY_THRESHOLD - 1)["decision"] == "promoted"
    verdict = judge(_hypothesis(), weak, n_family=DEFLATION_FAMILY_THRESHOLD)
    assert verdict["decision"] == "rejected"
    assert "deflatiert" in verdict["reasons"][0]
    assert verdict["deflated"] is True


def test_judge_rejects_too_few_legs() -> None:
    verdict = judge(_hypothesis(), _oos_matrix(legs=5), n_family=0)
    assert verdict["decision"] == "rejected"
    assert any("Legs 5 < 10" in r for r in verdict["reasons"])


def test_judge_rejects_high_drawdown() -> None:
    verdict = judge(_hypothesis(), _oos_matrix(dd=9.0), n_family=0)
    assert verdict["decision"] == "rejected"
    assert any("Max-DD" in r for r in verdict["reasons"])


def test_judge_rejects_missing_majority() -> None:
    verdict = judge(_hypothesis(), _oos_matrix(positive=0, n=2), n_family=0)
    assert verdict["decision"] == "rejected"
    assert any("Mehrheit" in r for r in verdict["reasons"])


def test_judge_accepts_exact_boundaries() -> None:
    # delta == Marge, Legs == Minimum, Max-DD == Cap, Mehrheit genau 50 %
    matrix = _oos_matrix(variant=2.0, baseline=1.0, legs=10, dd=8.0, positive=1, n=2)
    verdict = judge(_hypothesis(), matrix, n_family=0)
    assert verdict["decision"] == "promoted"


def test_run_variant_matrix_fails_without_oos_data(small_plan: TestPlan) -> None:
    hypothesis = make_hypothesis(test_plan=small_plan)
    matrix, error = run_variant_matrix(hypothesis, lambda *args: [], Variant(strategy="rsi_mean_reversion"))
    assert matrix is None
    assert error == "kein OOS-Asset-Fenster mit ausreichenden Daten"


def test_run_variant_matrix_full_run_structure(small_plan: TestPlan) -> None:
    hypothesis = make_hypothesis(params={"buy_below": 25.0}, test_plan=small_plan)
    baseline = Variant(strategy="rsi_mean_reversion", params={"period": 30.0, "buy_below": 20.0, "sell_above": 80.0})
    matrix, error = run_variant_matrix(hypothesis, make_feed_factory(), baseline)
    assert error is None
    assert matrix is not None
    assert set(matrix) == {"windows", "portfolio", "bootstrap", "data_gaps"}
    assert matrix["data_gaps"] == []
    for window in ("calibration", "oos"):
        entry = matrix["windows"][window]["BTC/USDT"]
        assert {"variant", "baseline", "start", "end"} <= set(entry)
        assert entry["variant"]["n_candles"] == 864
    oos = matrix["portfolio"]["oos"]
    assert oos["n"] == 1
    assert oos["legs"] >= 0
    assert matrix["bootstrap"] is None  # 3 Tage liefern < 5 Tagesrenditen
