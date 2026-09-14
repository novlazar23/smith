"""Regression: Default-Parameter der Agenten = Pre-Refactor-Verhalten.

Die Referenzwerte wurden vor der Parameterisierung auf festen synthetischen
Datensätzen gefangen; ein Drift der Defaults oder der Parameter-Verdrahtung
schiebt diese Tests. Zusätzlich: to_dict/from_dict-Serialisierung der
Parameter-Datenklassen (Basis für champion_configs.json).
"""

from __future__ import annotations

import numpy as np
import pytest
from packages.agents.mean_reversion_agent import MeanReversionAgent, MeanReversionParams
from packages.agents.trend_agent import TrendAgent, TrendParams
from packages.agents.volatility_regime_agent import VolatilityRegimeAgent, VolatilityRegimeParams
from packages.agents.volume_conviction_agent import VolumeConvictionAgent, VolumeConvictionParams


def _ramp(n: int, step: float, start: float = 100.0) -> dict[str, np.ndarray]:
    close = start + step * np.arange(n, dtype=np.float64)
    return {
        "close": close,
        "high": close + 0.2,
        "low": close - 0.2,
        "open": close - step / 2.0,
        "volume": 1000.0 * (1.0 + 0.001 * np.arange(n, dtype=np.float64)),
    }


def _flat(n: int) -> dict[str, np.ndarray]:
    return {
        "close": np.full(n, 100.0),
        "high": np.full(n, 101.0),
        "low": np.full(n, 99.0),
        "open": np.full(n, 100.0),
        "volume": np.full(n, 1000.0),
    }


def _mr_flat(n: int = 150) -> dict[str, np.ndarray]:
    close = 100.0 + 0.1 * np.sin(np.arange(n, dtype=np.float64))
    return {"close": close, "high": close + 0.2, "low": close - 0.2, "open": close, "volume": np.full(n, 1000.0)}


def _mr_crash() -> dict[str, np.ndarray]:
    n_flat, n_crash = 140, 10
    close = np.concatenate([100.0 + 0.1 * np.sin(np.arange(n_flat, dtype=np.float64)), np.linspace(100.0, 55.0, n_crash)])
    return {"close": close, "high": close + 0.2, "low": close - 0.2, "open": close, "volume": np.full(len(close), 1000.0)}


def _mr_pump() -> dict[str, np.ndarray]:
    n_flat, n_pump = 140, 10
    close = np.concatenate([100.0 + 0.1 * np.sin(np.arange(n_flat, dtype=np.float64)), np.linspace(100.0, 145.0, n_pump)])
    return {"close": close, "high": close + 0.2, "low": close - 0.2, "open": close, "volume": np.full(len(close), 1000.0)}


def _vr_normal(n: int = 150) -> dict[str, np.ndarray]:
    close = 100.0 + 1.0 * np.sin(np.arange(n, dtype=np.float64))
    return {"close": close, "high": close + 0.2, "low": close - 0.2, "open": close, "volume": np.full(n, 1000.0)}


def _vr_breakout_up() -> dict[str, np.ndarray]:
    normal = 100.0 + 1.0 * np.sin(np.arange(100, dtype=np.float64))
    squeeze = 100.0 + 0.05 * np.sin(np.arange(40, dtype=np.float64))
    break_ = 100.5 + 0.5 * np.arange(10, dtype=np.float64)
    close = np.concatenate([normal, squeeze, break_])
    return {"close": close, "high": close + 0.2, "low": close - 0.2, "open": close, "volume": np.full(len(close), 1000.0)}


def _vr_breakout_down() -> dict[str, np.ndarray]:
    normal = 100.0 + 1.0 * np.sin(np.arange(100, dtype=np.float64))
    squeeze = 100.0 + 0.05 * np.sin(np.arange(40, dtype=np.float64))
    break_ = 99.5 - 0.5 * np.arange(10, dtype=np.float64)
    close = np.concatenate([normal, squeeze, break_])
    return {"close": close, "high": close + 0.2, "low": close - 0.2, "open": close, "volume": np.full(len(close), 1000.0)}


def _vr_expansion() -> dict[str, np.ndarray]:
    normal = 100.0 + 1.0 * np.sin(np.arange(100, dtype=np.float64))
    exp = 100.0 + 3.0 * np.sin(2.0 * np.arange(50, dtype=np.float64))
    close = np.concatenate([normal, exp])
    return {"close": close, "high": close + 0.2, "low": close - 0.2, "open": close, "volume": np.full(len(close), 1000.0)}


def _finish(close: np.ndarray, volume: np.ndarray) -> dict[str, np.ndarray]:
    n = len(close)
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) + 0.05
    low = np.minimum(open_, close) - 0.05
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}


def _vc_accumulation(n: int = 60) -> dict[str, np.ndarray]:
    close = np.empty(n)
    volume = np.empty(n)
    close[0] = 100.0
    volume[0] = 1000.0
    for i in range(1, n):
        if i % 5 == 4:
            close[i] = close[i - 1] - 0.1
            volume[i] = 200.0
        else:
            close[i] = close[i - 1] + 0.2
            volume[i] = 1000.0
    return _finish(close, volume)


def _vc_distribution(n: int = 60) -> dict[str, np.ndarray]:
    close = np.empty(n)
    volume = np.empty(n)
    close[0] = 100.0
    volume[0] = 1000.0
    for i in range(1, n):
        if i % 5 == 4:
            close[i] = close[i - 1] + 0.1
            volume[i] = 200.0
        else:
            close[i] = close[i - 1] - 0.2
            volume[i] = 1000.0
    return _finish(close, volume)


def _vc_flat(n: int = 60) -> dict[str, np.ndarray]:
    close = np.array([100.05 if i % 2 == 1 else 99.95 for i in range(n)])
    return _finish(close, np.full(n, 1000.0))


def _vc_divergence(n: int = 60) -> dict[str, np.ndarray]:
    close = np.empty(n)
    volume = np.empty(n)
    close[0] = 100.0
    volume[0] = 200.0
    for i in range(1, n):
        if i % 2 == 1:
            close[i] = close[i - 1] + 0.5
            volume[i] = 200.0
        else:
            close[i] = close[i - 1] - 0.2
            volume[i] = 1000.0
    return _finish(close, volume)


def _cases() -> list[tuple[str, object, dict[str, np.ndarray], dict[str, float]]]:
    return [
        ("trend_ramp_up", TrendAgent(), _ramp(100, 0.5), {"up": 0.85, "down": 0.05, "range": 0.1}),
        ("trend_ramp_down", TrendAgent(), _ramp(100, -0.5), {"down": 0.85, "up": 0.05, "range": 0.1}),
        ("trend_flat", TrendAgent(), _flat(100), {"up": 0.1, "down": 0.1, "range": 0.8}),
        ("mr_crash", MeanReversionAgent(), _mr_crash(), {"up": 0.85, "down": 0.05, "range": 0.1}),
        ("mr_pump", MeanReversionAgent(), _mr_pump(), {"down": 0.85, "up": 0.05, "range": 0.1}),
        ("mr_flat", MeanReversionAgent(), _mr_flat(), {"up": 0.2694, "down": 0.0, "range": 0.7306}),
        ("vr_breakout_up", VolatilityRegimeAgent(), _vr_breakout_up(), {"up": 0.8226, "down": 0.05, "range": 0.1274}),
        ("vr_breakout_down", VolatilityRegimeAgent(), _vr_breakout_down(), {"down": 0.8226, "up": 0.05, "range": 0.1274}),
        ("vr_expansion", VolatilityRegimeAgent(), _vr_expansion(), {"up": 0.0923, "down": 0.3077, "range": 0.6}),
        ("vr_normal", VolatilityRegimeAgent(), _vr_normal(), {"up": 0.0282, "down": 0.2739, "range": 0.6979}),
        ("vc_accumulation", VolumeConvictionAgent(), _vc_accumulation(), {"up": 0.85, "down": 0.0375, "range": 0.1125}),
        ("vc_distribution", VolumeConvictionAgent(), _vc_distribution(), {"up": 0.0383, "down": 0.8469, "range": 0.1148}),
        ("vc_flat", VolumeConvictionAgent(), _vc_flat(), {"up": 0.33, "down": 0.33, "range": 0.34}),
        ("vc_divergence", VolumeConvictionAgent(), _vc_divergence(), {"up": 0.25, "down": 0.25, "range": 0.5}),
    ]


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c[0])
def test_defaults_reproduce_pre_refactor_output(
    case: tuple[str, object, dict[str, np.ndarray], dict[str, float]],
) -> None:
    _name, agent, data, expected = case
    assert agent.analyze(data).probabilities == pytest.approx(expected)


class TestParamsSerialization:
    @pytest.mark.parametrize(
        "params_cls",
        [TrendParams, MeanReversionParams, VolatilityRegimeParams, VolumeConvictionParams],
    )
    def test_roundtrip(self, params_cls: type) -> None:
        params = params_cls()
        assert type(params).from_dict(params.to_dict()) == params

    @pytest.mark.parametrize(
        "params_cls",
        [TrendParams, MeanReversionParams, VolatilityRegimeParams, VolumeConvictionParams],
    )
    def test_space_covers_all_fields(self, params_cls: type) -> None:
        assert set(params_cls.PARAM_SPACE) == set(params_cls().to_dict())

    def test_from_dict_ignores_unknown_and_missing_keys(self) -> None:
        assert TrendParams.from_dict({**TrendParams().to_dict(), "bogus": 3}) == TrendParams()
        assert TrendParams.from_dict({}) == TrendParams()
