"""Tests für packages.indicators.volatility — ATR."""

from __future__ import annotations

import numpy as np
import pytest
from packages.indicators import ATR


class TestATR:
    """Testet Average True Range."""

    @pytest.fixture
    def ohlcv_data(self) -> dict[str, np.ndarray]:
        rng = np.random.RandomState(42)
        n = 100
        close = 100.0 + np.cumsum(rng.randn(n) * 0.5)
        high = close + np.abs(rng.randn(n) * 0.3)
        low = close - np.abs(rng.randn(n) * 0.3)
        return {"high": high, "low": low, "close": close}

    def test_atr_returns_result(self, ohlcv_data: dict[str, np.ndarray]) -> None:
        """Ergebnis hat Namen 'ATR' und korrekte Länge."""
        result = ATR().compute(ohlcv_data)
        assert result.name == "ATR"
        assert len(result.values) == 100

    def test_atr_default_period(self) -> None:
        """Standard-Periode ist 14."""
        atr = ATR()
        assert atr.period == 14

    def test_atr_custom_period(self) -> None:
        """Benutzerdefinierte Periode wird angewendet."""
        atr = ATR(period=7)
        assert atr.period == 7
        assert atr.min_periods == 7

    def test_atr_missing_high_raises(self) -> None:
        """ValueError bei fehlendem 'high'."""
        with pytest.raises(ValueError, match="Missing required"):
            ATR().compute({"low": np.array([1.0, 2.0]), "close": np.array([1.0, 2.0])})

    def test_atr_missing_low_raises(self) -> None:
        """ValueError bei fehlendem 'low'."""
        with pytest.raises(ValueError, match="Missing required"):
            ATR().compute({"high": np.array([1.0, 2.0]), "close": np.array([1.0, 2.0])})

    def test_atr_missing_close_raises(self) -> None:
        """ValueError bei fehlendem 'close'."""
        with pytest.raises(ValueError, match="Missing required"):
            ATR().compute({"high": np.array([1.0, 2.0]), "low": np.array([1.0, 2.0])})

    def test_atr_too_few_data_raises(self) -> None:
        """ValueError bei weniger Daten als Periode."""
        high = np.array([1.0, 2.0, 3.0])
        low = np.array([0.5, 1.5, 2.5])
        close = np.array([1.0, 1.5, 2.0])
        with pytest.raises(ValueError, match="Need at least"):
            ATR(period=14).compute({"high": high, "low": low, "close": close})

    def test_atr_positive_values(self, ohlcv_data: dict[str, np.ndarray]) -> None:
        """ATR-Werte sind positiv."""
        result = ATR().compute(ohlcv_data)
        valid = result.values[~np.isnan(result.values)]
        assert len(valid) > 0
        assert np.all(valid > 0)

    def test_atr_nan_before_period(self) -> None:
        """ATR ist NaN vor der ersten Periode."""
        high = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        low = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5])
        close = np.array([1.0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5])
        result = ATR().compute({"high": high, "low": low, "close": close})
        assert len(result.values) == 15
        assert np.isnan(result.values[0])

    def test_atr_constant_high_low(self) -> None:
        """Konstantes High/Low gibt TR = high - low."""
        n = 30
        high = np.full(n, 101.0)
        low = np.full(n, 99.0)
        close = np.full(n, 100.0)
        result = ATR().compute({"high": high, "low": low, "close": close})
        valid = result.values[~np.isnan(result.values)]
        assert len(valid) > 0
        assert np.allclose(valid, 2.0)

    def test_atr_period_5(self) -> None:
        """ATR mit Periode 5."""
        n = 30
        rng = np.random.RandomState(42)
        close = 100.0 + np.cumsum(rng.randn(n) * 0.5)
        high = close + np.abs(rng.randn(n) * 0.3) + 0.5
        low = close - np.abs(rng.randn(n) * 0.3) - 0.5
        result = ATR(period=5).compute({"high": high, "low": low, "close": close})
        assert len(result.values) == 30

    def test_atr_increasing_volatility(self) -> None:
        """ATR steigt mit zunehmender Volatilität."""
        close = np.arange(1, 41, dtype=float)
        high = close + np.arange(1, 41, dtype=float) * 0.1
        low = close - np.arange(1, 41, dtype=float) * 0.1
        result = ATR(period=10).compute({"high": high, "low": low, "close": close})
        valid = result.values[~np.isnan(result.values)]
        if len(valid) >= 10:
            first_half = valid[: len(valid) // 2]
            second_half = valid[len(valid) // 2 :]
            assert np.mean(second_half) > np.mean(first_half)

    def test_atr_metadata_period(self) -> None:
        """Metadata enthält die Periode."""
        n = 30
        close = np.full(n, 100.0)
        result = ATR(period=7).compute(
            {"high": close + 1, "low": close - 1, "close": close}
        )
        assert result.metadata["period"] == 7
