"""Tests for FeatureExtractor — apps/feature_worker/worker.py."""

from __future__ import annotations

import numpy as np
import pytest
from apps.feature_worker.worker import FeatureExtractor

# ── helpers ───────────────────────────────────────────────────────────

def _make_partial(
    open_: np.ndarray | None = None,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    close: np.ndarray | None = None,
    volume: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build a minimal OHLCV dict (only specified fields filled)."""
    return {
        "open": np.asarray(open_ if open_ is not None else [100.0]),
        "high": np.asarray(high if high is not None else [101.0]),
        "low": np.asarray(low if low is not None else [99.0]),
        "close": np.asarray(close if close is not None else [100.0]),
        "volume": np.asarray(volume if volume is not None else [1000.0]),
    }


def _ohlcv(
    n: int = 100,
    open_price: float = 100.0,
    close_price: float | None = None,
    high_pct: float = 1.01,
    low_pct: float = 0.99,
    volume: float = 1000.0,
) -> dict[str, np.ndarray]:
    """Build a realistic OHLCV dict with *n* candles.

    Close follows a random-walk-like path starting from *open_price*.
    """
    closes = np.linspace(open_price, close_price or open_price * 1.05, n)
    opens = np.roll(closes, 1)
    opens[0] = open_price
    highs = closes * high_pct
    lows = closes * low_pct
    volumes = np.full(n, volume, dtype=np.float64)
    return {
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }


# ═══════════════════════════════════════════════════════════════════════
# extract() — basic behaviour
# ═══════════════════════════════════════════════════════════════════════

class TestExtractBasic:
    """extract() returns a flat dict of floats with expected feature names."""

    def test_returns_dict_of_floats(self) -> None:
        fx = FeatureExtractor()
        ohlcv = _ohlcv(n=50)
        result = fx.extract(ohlcv)
        assert isinstance(result, dict)
        for v in result.values():
            assert isinstance(v, float)

    def test_feature_names(self) -> None:
        fx = FeatureExtractor()
        ohlcv = _ohlcv(n=100)
        result = fx.extract(ohlcv)
        expected_keys = {
            "return",
            "volatility_20",
            "momentum_10",
            "volume_ratio",
            "price_change",
            "price_range",
            "volume_change",
        }
        assert set(result.keys()) == expected_keys

    def test_return_is_non_zero_for_price_movement(self) -> None:
        fx = FeatureExtractor()
        close = np.array([100.0, 102.0])
        ohlcv = _make_partial(close=close)
        result = fx.extract(ohlcv)
        assert result["return"] == pytest.approx(np.log(102.0 / 100.0), rel=1e-9)

    def test_price_change_matches_close_minus_open(self) -> None:
        fx = FeatureExtractor()
        ohlcv = _ohlcv(n=10)
        result = fx.extract(ohlcv)
        expected = float(ohlcv["close"][-1] - ohlcv["open"][-1])
        assert result["price_change"] == pytest.approx(expected)

    def test_price_range_matches_high_minus_low(self) -> None:
        fx = FeatureExtractor()
        ohlcv = _ohlcv(n=10)
        result = fx.extract(ohlcv)
        expected = float(ohlcv["high"][-1] - ohlcv["low"][-1])
        assert result["price_range"] == pytest.approx(expected)


# ═══════════════════════════════════════════════════════════════════════
# extract() — edge cases
# ═══════════════════════════════════════════════════════════════════════

class TestExtractEdgeCases:
    """Branch coverage for short data, empty metadata, zeros, NaN."""

    def test_empty_metadata_ignored(self) -> None:
        fx = FeatureExtractor()
        ohlcv = _ohlcv(n=30)
        result = fx.extract(ohlcv, metadata={"symbol": "BTC/USDT"})
        assert "return" in result  # metadata is unused but no crash

    def test_zero_length_array(self) -> None:
        fx = FeatureExtractor()
        empty = {
            "open": np.array([]),
            "high": np.array([]),
            "low": np.array([]),
            "close": np.array([]),
            "volume": np.array([]),
        }
        result = fx.extract(empty)
        assert all(v == 0.0 for v in result.values())

    def test_single_candle(self) -> None:
        fx = FeatureExtractor()
        single = {
            "open": np.array([50.0]),
            "high": np.array([52.0]),
            "low": np.array([48.0]),
            "close": np.array([50.0]),
            "volume": np.array([100.0]),
        }
        result = fx.extract(single)
        assert result["return"] == 0.0
        assert result["price_change"] == 0.0
        assert result["price_range"] == pytest.approx(4.0)

    def test_two_candles_minimum_for_return(self) -> None:
        fx = FeatureExtractor()
        single = {
            "open": np.array([10.0, 11.0]),
            "high": np.array([11.0, 12.0]),
            "low": np.array([9.0, 10.0]),
            "close": np.array([10.5, 11.5]),
            "volume": np.array([100.0, 200.0]),
        }
        result = fx.extract(single)
        assert result["return"] == pytest.approx(np.log(11.5 / 10.5))
        assert result["volume_change"] == pytest.approx(200.0 / 100.0)

    def test_volume_ratio_short_data(self) -> None:
        """volume < 21 → falls back to volume[-1]/volume[-2]."""
        fx = FeatureExtractor()
        volumes = np.array([100.0, 200.0, 300.0])
        ohlcv = _make_partial(
            close=np.array([10.0, 11.0, 12.0]),
            volume=volumes,
        )
        result = fx.extract(ohlcv)
        assert result["volume_ratio"] == pytest.approx(300.0 / 200.0)

    def test_volume_ratio_long_data(self) -> None:
        """volume >= 21 → avg of last 20 values."""
        fx = FeatureExtractor()
        volumes = np.full(25, 100.0, dtype=np.float64)
        volumes[-1] = 300.0  # spike
        ohlcv = _make_partial(
            close=np.linspace(10.0, 15.0, 26),
            volume=volumes,
        )
        result = fx.extract(ohlcv)
        expected_avg = float(np.mean(volumes[-21:-1]))
        assert result["volume_ratio"] == pytest.approx(300.0 / expected_avg)

    def test_zero_volume_division(self) -> None:
        """volume[-2] == 0 → returns 0.0 instead of divide-by-zero."""
        fx = FeatureExtractor()
        ohlcv = _make_partial(
            close=np.array([10.0, 11.0]),
            volume=np.array([0.0, 100.0]),
        )
        result = fx.extract(ohlcv)
        assert result["volume_change"] == 0.0
        assert result["volume_ratio"] == 0.0

    def test_float_conversion_round_trip(self) -> None:
        """Ensure numpy float64 inputs produce plain Python floats."""
        fx = FeatureExtractor()
        ohlcv = _ohlcv(n=50)
        result = fx.extract(ohlcv)
        for val in result.values():
            assert isinstance(val, float)
            # No NaN or inf should leak into results
            assert np.isfinite(val) or val == 0.0


# ═══════════════════════════════════════════════════════════════════════
# extract_rolling_features() — basic behaviour
# ═══════════════════════════════════════════════════════════════════════

class TestExtractRollingFeatures:
    """extract_rolling_features() returns expected keys and reasonable values."""

    def test_returns_expected_keys(self) -> None:
        fx = FeatureExtractor()
        ohlcv = _ohlcv(n=100)
        result = fx.extract_rolling_features(ohlcv)
        expected = {"mean_return", "std_return", "max_drawdown", "volume_avg", "volume_std"}
        assert set(result.keys()) == expected

    def test_mean_return_positive_upward_trend(self) -> None:
        fx = FeatureExtractor()
        ohlcv = _ohlcv(n=100, close_price=150.0)
        result = fx.extract_rolling_features(ohlcv)
        assert result["mean_return"] > 0.0

    def test_std_return_positive_for_volatility(self) -> None:
        fx = FeatureExtractor()
        ohlcv = _ohlcv(n=60)
        result = fx.extract_rolling_features(ohlcv)
        assert result["std_return"] >= 0.0

    def test_max_drawdown_non_negative(self) -> None:
        fx = FeatureExtractor()
        ohlcv = _ohlcv(n=100)
        result = fx.extract_rolling_features(ohlcv)
        assert result["max_drawdown"] >= 0.0

    def test_window_custom(self) -> None:
        fx = FeatureExtractor()
        ohlcv = _ohlcv(n=60)
        result = fx.extract_rolling_features(ohlcv, window=10)
        assert "mean_return" in result
        assert result["mean_return"] != 0.0  # 60 elements, window=10 → data available

    def test_volume_avg_matches_mean(self) -> None:
        fx = FeatureExtractor()
        volumes = np.full(40, 500.0, dtype=np.float64)
        ohlcv = _make_partial(
            close=np.linspace(100.0, 120.0, 41),
            volume=volumes,
        )
        result = fx.extract_rolling_features(ohlcv)
        assert result["volume_avg"] == pytest.approx(500.0)


# ═══════════════════════════════════════════════════════════════════════
# extract_rolling_features() — edge cases
# ═══════════════════════════════════════════════════════════════════════

class TestExtractRollingEdgeCases:
    """Short data, zero returns, NaN handling."""

    def test_single_candle_returns_zeros(self) -> None:
        fx = FeatureExtractor()
        ohlcv = _make_partial(
            close=np.array([100.0]),
            volume=np.array([]),
        )
        result = fx.extract_rolling_features(ohlcv)
        assert all(v == 0.0 for v in result.values())

    def test_finite_clips_nan_in_returns(self) -> None:
        """NaN returns should be replaced by 0.0 (np.isfinite)."""
        fx = FeatureExtractor()
        close = np.array([100.0, float("nan"), 120.0])
        ohlcv = _make_partial(
            close=close,
            volume=np.array([100.0, 200.0, 300.0]),
        )
        result = fx.extract_rolling_features(ohlcv)
        # NaN is replaced; result should be finite
        assert np.isfinite(result["mean_return"]) or result["mean_return"] == 0.0

    def test_fewer_than_window_uses_all_data(self) -> None:
        fx = FeatureExtractor()
        close = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        volume = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
        ohlcv = _make_partial(close=close, volume=volume)
        result = fx.extract_rolling_features(ohlcv, window=20)
        assert result["volume_avg"] == pytest.approx(np.mean(volume))


# ═══════════════════════════════════════════════════════════════════════
# normalize_features() — min-max normalisation
# ═══════════════════════════════════════════════════════════════════════

class TestNormalizeFeatures:
    """normalize_features() applies min-max normalisation correctly."""

    def test_empty_dict_returns_empty(self) -> None:
        fx = FeatureExtractor()
        assert fx.normalize_features({}) == {}

    def test_values_normalized_to_01(self) -> None:
        fx = FeatureExtractor()
        features = {"a": 1.0, "b": 5.0, "c": 10.0}
        result = fx.normalize_features(features)
        assert result["a"] == pytest.approx(0.0)
        assert result["b"] == pytest.approx(4.0 / 9.0)
        assert result["c"] == pytest.approx(1.0)

    def test_all_same_value_all_zeros(self) -> None:
        fx = FeatureExtractor()
        features = {"x": 3.0, "y": 3.0, "z": 3.0}
        result = fx.normalize_features(features)
        assert all(v == 0.0 for v in result.values())

    def test_preserves_keys(self) -> None:
        fx = FeatureExtractor()
        features = {"alpha": 0.0, "beta": 100.0, "gamma": -50.0}
        result = fx.normalize_features(features)
        assert set(result.keys()) == set(features.keys())

    def test_range_preserves_order(self) -> None:
        fx = FeatureExtractor()
        features = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0}
        result = fx.normalize_features(features)
        values = list(result.values())
        assert values == sorted(values)  # order preserved


# ═══════════════════════════════════════════════════════════════════════
# _pad_to_max() — static helper
# ═══════════════════════════════════════════════════════════════════════

class TestPadToMax:
    """_pad_to_max truncates or zero-pads as expected."""

    def test_shorter_than_max_pads_with_zeros(self) -> None:
        arr = np.array([1.0, 2.0])
        result = FeatureExtractor._pad_to_max(arr, 5)
        expected = np.array([1.0, 2.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_longer_than_max_truncates(self) -> None:
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = FeatureExtractor._pad_to_max(arr, 4)
        expected = np.array([3.0, 4.0, 5.0, 6.0])
        np.testing.assert_array_equal(result, expected)

    def test_exact_length_returns_last_n(self) -> None:
        arr = np.array([1.0, 2.0, 3.0])
        result = FeatureExtractor._pad_to_max(arr, 3)
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result, expected)

    def test_dtype_preserved(self) -> None:
        arr = np.array([1, 2], dtype=np.int32)
        result = FeatureExtractor._pad_to_max(arr, 4)
        assert result.dtype == np.int32
        assert len(result) == 4
