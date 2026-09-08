"""Tests für den TimesFM-Research-Spike (Provider-Abstraktion, Features, CLI)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from apps.evolution.__main__ import build_parser
from apps.evolution.timesfm_spike import FEATURE_COLUMNS, run_timesfm_spike, safe_instrument
from packages.forecasting.timesfm import (
    FakeProvider,
    QuantileForecast,
    TimesFMProvider,
    TimesFMUnavailableError,
    load_timesfm_provider,
    provider_label,
)
from tests.unit.test_evolution.conftest import make_candles


class RecordingProvider:
    """Provider, der den übergebenen Context aufzeichnet (Leakage-Check)."""

    label: str = "recording"

    def __init__(self) -> None:
        self.contexts: list[np.ndarray] = []
        self.horizons: list[int] = []

    def predict(self, context: np.ndarray, horizon: int) -> QuantileForecast:
        self.contexts.append(context.copy())
        self.horizons.append(horizon)
        base = float(context[-1])
        steps = np.arange(1, horizon + 1, dtype=np.float64)
        return QuantileForecast(
            median=base * (1.0 + 0.001 * steps),
            q10=base * (1.0 + 0.0005 * steps),
            q90=base * (1.0 + 0.0015 * steps),
        )


class ShortProvider:
    label: str = "short"

    def predict(self, context: np.ndarray, horizon: int) -> QuantileForecast:
        base = float(context[-1])
        size = max(horizon - 1, 0)
        return QuantileForecast(
            median=np.full(size, base, dtype=np.float64),
            q10=np.full(size, base, dtype=np.float64),
            q90=np.full(size, base, dtype=np.float64),
        )


def test_safe_instrument_replaces_slash() -> None:
    assert safe_instrument("BTC/USDT") == "BTC-USDT"


def test_fake_provider_is_deterministic_and_ordered() -> None:
    provider = FakeProvider()
    context = np.linspace(100.0, 110.0, 50)
    first = provider.predict(context, 12)
    second = provider.predict(context, 12)
    assert np.array_equal(first.median, second.median)
    assert np.array_equal(first.q10, second.q10)
    assert np.array_equal(first.q90, second.q90)
    assert first.median.shape == (12,)
    assert np.all(first.q10 <= first.median)
    assert np.all(first.median <= first.q90)


def test_fake_provider_rejects_invalid_context() -> None:
    provider = FakeProvider()
    with pytest.raises(ValueError, match="mindestens zwei"):
        provider.predict(np.array([100.0]), 5)
    with pytest.raises(ValueError, match="positiv"):
        provider.predict(np.array([100.0, 0.0]), 5)
    with pytest.raises(ValueError, match="horizon"):
        provider.predict(np.array([100.0, 101.0]), 0)


class _FakeTimesFMModel:
    def __init__(self) -> None:
        self.model_id = ""
        self.cache_dir: str | None = None
        self.config: Any = None
        self.last_inputs: list[np.ndarray] = []
        self.last_horizon = 0

    def compile(self, config: Any) -> None:
        self.config = config

    def forecast(
        self, horizon: int, inputs: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        self.last_horizon = horizon
        self.last_inputs = [np.asarray(values, dtype=np.float32) for values in inputs]
        base = float(self.last_inputs[0][-1])
        point = np.full((1, horizon), base * 1.01, dtype=np.float64)
        quantiles = np.empty((1, horizon, 10), dtype=np.float64)
        quantiles[..., 0] = point
        quantiles[..., 1] = point * 0.99
        quantiles[..., 5] = point
        quantiles[..., 9] = point * 1.02
        return point, quantiles


class _FakeForecastConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _install_fake_timesfm(monkeypatch: pytest.MonkeyPatch) -> _FakeTimesFMModel:
    model = _FakeTimesFMModel()
    fake_module = types.ModuleType("timesfm")

    class _ModelClass:
        @staticmethod
        def from_pretrained(model_id: str, cache_dir: str | None = None) -> _FakeTimesFMModel:
            model.model_id = model_id
            model.cache_dir = cache_dir
            return model

    fake_module.TimesFM_2p5_200M_torch = _ModelClass
    fake_module.ForecastConfig = _FakeForecastConfig
    monkeypatch.setitem(sys.modules, "timesfm", fake_module)
    return model


def test_timesfm_provider_unavailable_without_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "timesfm", None)
    provider = load_timesfm_provider("test-model")
    assert provider_label(provider) == "timesfm:test-model"
    with pytest.raises(TimesFMUnavailableError, match="nicht installiert"):
        provider.predict(np.linspace(100.0, 110.0, 40), 3)


def test_timesfm_provider_maps_quantiles(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_model = _install_fake_timesfm(monkeypatch)
    provider = TimesFMProvider(model="test-model", max_context=64, max_horizon=8)
    context = np.linspace(100.0, 110.0, 40)
    forecast = provider.predict(context, 6)

    assert fake_model.model_id == "test-model"
    assert fake_model.last_horizon == 6
    assert fake_model.last_inputs[0].dtype == np.float32
    assert fake_model.last_inputs[0].size == 40
    assert fake_model.config.kwargs["max_context"] == 64
    assert fake_model.config.kwargs["max_horizon"] == 8
    assert fake_model.config.kwargs["infer_is_positive"] is True
    assert forecast.median.shape == (6,)
    assert np.all(forecast.q10 < forecast.median)
    assert np.all(forecast.median < forecast.q90)


def test_timesfm_provider_truncates_context_to_max(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_model = _install_fake_timesfm(monkeypatch)
    provider = TimesFMProvider(model="test-model", max_context=32, max_horizon=8)
    context = np.linspace(100.0, 110.0, 50)
    provider.predict(context, 3)
    assert fake_model.last_inputs[0].size == 32
    assert fake_model.last_inputs[0][0] == pytest.approx(context[18])


def test_timesfm_provider_rejects_short_context_and_horizon() -> None:
    provider = TimesFMProvider(model="test-model", max_context=64, max_horizon=4)
    with pytest.raises(ValueError, match="32"):
        provider.predict(np.linspace(100.0, 110.0, 31), 2)
    with pytest.raises(ValueError, match="horizon"):
        provider.predict(np.linspace(100.0, 110.0, 40), 5)


def test_run_timesfm_spike_writes_features_and_report(tmp_path: Path) -> None:
    candles = make_candles(600)
    provider = RecordingProvider()
    out_dir = tmp_path / "timesfm"
    report = run_timesfm_spike(
        candles,
        instrument="BTC/USDT",
        provider=provider,
        horizon=10,
        context=100,
        step=50,
        out_dir=out_dir,
    )

    assert report["n_candles"] == 600
    assert report["n_features"] == 11
    assert report["provider"] == "recording"
    assert report["cache_key"]

    features = pd.read_parquet(out_dir / "features.parquet")
    assert list(features.columns) == list(FEATURE_COLUMNS)
    assert len(features) == 11
    widths = features["tfm_interval_width_pct"].to_numpy(dtype=float)
    assert widths.size == 11
    assert np.isfinite(widths).all()
    assert np.all(widths >= 0.0)

    saved = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert saved["cache_key"] == report["cache_key"]
    assert saved["n_features"] == 11

    # Leakage-Check: der Context endet exakt beim Feature-Zeitstempel.
    assert len(provider.contexts) == 11
    for i, context_array in enumerate(provider.contexts):
        end_index = 99 + i * 50
        assert context_array.size == 100
        assert context_array[-1] == pytest.approx(candles[end_index].close)
        assert provider.horizons[i] == 10
        row_timestamp = features.loc[i, "timestamp"].to_pydatetime()
        assert row_timestamp == candles[end_index].timestamp


def test_run_timesfm_spike_cache_key_is_deterministic(tmp_path: Path) -> None:
    candles = make_candles(400)
    first = run_timesfm_spike(
        candles,
        instrument="BTC/USDT",
        provider=FakeProvider(),
        horizon=10,
        context=100,
        step=50,
        out_dir=tmp_path / "a",
    )
    second = run_timesfm_spike(
        candles,
        instrument="BTC/USDT",
        provider=FakeProvider(),
        horizon=10,
        context=100,
        step=50,
        out_dir=tmp_path / "b",
    )
    assert first["cache_key"] == second["cache_key"]

    different = run_timesfm_spike(
        candles,
        instrument="ETH/USDT",
        provider=FakeProvider(),
        horizon=10,
        context=100,
        step=50,
        out_dir=tmp_path / "c",
    )
    assert different["cache_key"] != first["cache_key"]


def test_run_timesfm_spike_rejects_too_few_candles() -> None:
    with pytest.raises(ValueError, match="Zu wenige Kerzen"):
        run_timesfm_spike(make_candles(50), instrument="BTC/USDT", provider=FakeProvider(), context=100)


def test_run_timesfm_spike_rejects_invalid_params() -> None:
    candles = make_candles(200)
    with pytest.raises(ValueError, match="horizon"):
        run_timesfm_spike(candles, instrument="BTC/USDT", provider=FakeProvider(), horizon=0)
    with pytest.raises(ValueError, match="context"):
        run_timesfm_spike(candles, instrument="BTC/USDT", provider=FakeProvider(), context=1)
    with pytest.raises(ValueError, match="step"):
        run_timesfm_spike(candles, instrument="BTC/USDT", provider=FakeProvider(), step=0)


def test_run_timesfm_spike_rejects_short_forecast() -> None:
    candles = make_candles(200)
    with pytest.raises(ValueError, match="kürzer"):
        run_timesfm_spike(candles, instrument="BTC/USDT", provider=ShortProvider(), horizon=10, context=50)


def test_cli_parser_exposes_timesfm_spike_options() -> None:
    args = build_parser().parse_args(
        [
            "--timesfm-spike",
            "--timesfm-instrument",
            "ETH/USDT",
            "--timesfm-start",
            "2024-01-01",
            "--timesfm-end",
            "2024-02-01",
            "--timesfm-resample",
            "1h",
            "--timesfm-context",
            "128",
            "--timesfm-horizon",
            "24",
            "--timesfm-step",
            "12",
            "--timesfm-out",
            "/tmp/tfm",
            "--timesfm-fake",
        ]
    )
    assert args.timesfm_spike is True
    assert args.timesfm_instrument == "ETH/USDT"
    assert args.timesfm_start == "2024-01-01"
    assert args.timesfm_end == "2024-02-01"
    assert args.timesfm_resample == "1h"
    assert args.timesfm_context == 128
    assert args.timesfm_horizon == 24
    assert args.timesfm_step == 12
    assert args.timesfm_out == "/tmp/tfm"
    assert args.timesfm_fake is True
