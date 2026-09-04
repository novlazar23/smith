"""Tests für die Strategie-Registry (Manifest, Validierung, Factory)."""

from __future__ import annotations

import pytest
from packages.strategies import (
    STRATEGIES,
    StrategyParamError,
    create_strategy,
    describe,
    list_strategies,
)
from packages.strategies._common import RuleStrategy

EXPECTED = [
    "bollinger_reversion",
    "donchian_breakout",
    "ema_cross",
    "keltner_breakout",
    "macd_cross",
    "momentum_roc",
    "rsi_mean_reversion",
    "stochastics",
    "supertrend",
    "vwap_reversion",
]


def test_lists_all_ten_strategies() -> None:
    assert list_strategies() == EXPECTED
    assert set(STRATEGIES) == set(EXPECTED)


def test_describe_returns_full_manifest() -> None:
    info = describe("ema_cross")
    assert info["name"] == "ema_cross"
    assert "EMA" in info["description"]
    assert set(info["params"]) == {"fast", "slow"}
    assert info["params"]["fast"] == {"default": 12.0, "min": 5.0, "max": 30.0}
    assert info["min_bars"] == 110


def test_every_strategy_has_consistent_manifest() -> None:
    for name in list_strategies():
        info = describe(name)
        assert info["description"]
        assert info["min_bars"] >= 2
        for spec in info["params"].values():
            assert spec["min"] <= spec["default"] <= spec["max"]


def test_create_strategy_unknown_name_raises() -> None:
    with pytest.raises(StrategyParamError, match="Unbekannte Strategie"):
        create_strategy("does_not_exist", "BTC/USDT")


def test_create_strategy_applies_defaults() -> None:
    strat = create_strategy("ema_cross", "BTC/USDT")
    assert isinstance(strat, RuleStrategy)
    assert strat.params == {"fast": 12.0, "slow": 26.0}
    assert strat.instrument == "BTC/USDT"
    assert strat.initial_capital == 100_000.0
    assert strat.trade_notional == 2_000.0
    assert strat.candle_limit == 300


def test_create_strategy_overrides_params() -> None:
    strat = create_strategy("ema_cross", "BTC/USDT", {"fast": 8.0, "slow": 30.0})
    assert strat.params == {"fast": 8.0, "slow": 30.0}


def test_create_strategy_unknown_param_raises() -> None:
    with pytest.raises(StrategyParamError, match="Unbekannte Parameter"):
        create_strategy("ema_cross", "BTC/USDT", {"bogus": 1.0})


def test_create_strategy_out_of_range_param_raises() -> None:
    with pytest.raises(StrategyParamError, match="außerhalb"):
        create_strategy("ema_cross", "BTC/USDT", {"fast": 1.0})


def test_relational_param_constraint_enforced() -> None:
    with pytest.raises(StrategyParamError, match="fast muss kleiner"):
        create_strategy("ema_cross", "BTC/USDT", {"fast": 30.0, "slow": 30.0})


def test_capital_validation() -> None:
    with pytest.raises(StrategyParamError, match="initial_capital"):
        create_strategy("ema_cross", "BTC/USDT", initial_capital=0.0)
    with pytest.raises(StrategyParamError, match="trade_notional"):
        create_strategy("ema_cross", "BTC/USDT", trade_notional=100_000.0)


def test_to_dict_roundtrip_metadata() -> None:
    strat = create_strategy("macd_cross", "ETH/USDT", {"signal": 12.0})
    state = strat.to_dict()
    assert state["strategy"] == "macd_cross"
    assert state["instrument"] == "ETH/USDT"
    assert state["params"]["signal"] == 12.0
    assert state["n_bars_seen"] == 0
