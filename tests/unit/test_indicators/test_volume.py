"""Tests für packages.indicators.volume — OBV, VWAP."""

from __future__ import annotations

import numpy as np
import pytest
from packages.indicators import OBV, VWAP


@pytest.fixture
def ohlcv_data() -> dict[str, np.ndarray]:
    """Standard OHLCV-Daten für Tests."""
    rng = np.random.RandomState(42)
    n = 100
    close = 100.0 + np.cumsum(rng.randn(n) * 0.5)
    high = close + np.abs(rng.randn(n) * 0.3)
    low = close - np.abs(rng.randn(n) * 0.3)
    volume = np.abs(rng.randn(n) * 1000) + 500
    return {
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


class TestOBV:
    """Testet On-Balance Volume."""

    def test_obv_returns_result(self, ohlcv_data: dict[str, np.ndarray]) -> None:
        """Ergebnis ist IndicatorResult mit korrektem Namen."""
        result = OBV().compute(ohlcv_data)
        assert result.name == "OBV"
        assert len(result.values) == 100

    def test_obv_length_matches(self, ohlcv_data: dict[str, np.ndarray]) -> None:
        """OBV-Werte haben gleiche Länge wie Input."""
        n = 50
        close = 100.0 + np.cumsum(np.random.RandomState(42).randn(n) * 0.5)
        high = close + 0.5
        low = close - 0.5
        volume = np.ones(n)
        result = OBV().compute({"close": close, "volume": volume, "high": high, "low": low})
        assert len(result.values) == n

    def test_obv_uptrend_increases(self) -> None:
        """OBV steigt im Aufwärtstrend."""
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        volume = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
        high = close + 1.0
        low = close - 1.0
        result = OBV().compute({"close": close, "volume": volume, "high": high, "low": low})
        # Erster Wert = volume[0] = 100
        assert result.values[0] == 100.0
        # Jeder Schritt steigt weil close[i] > close[i-1]
        for i in range(1, len(result.values)):
            assert result.values[i] > result.values[i - 1]

    def test_obv_downtrend_decreases(self) -> None:
        """OBV fällt im Abwärtstrend."""
        close = np.array([105.0, 104.0, 103.0, 102.0, 101.0, 100.0])
        volume = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
        high = close + 1.0
        low = close - 1.0
        result = OBV().compute({"close": close, "volume": volume, "high": high, "low": low})
        assert result.values[0] == 100.0
        for i in range(1, len(result.values)):
            assert result.values[i] < result.values[i - 1]

    def test_obv_flat_price_no_change(self) -> None:
        """Bei gleichem Preis bleibt OBV unverändert."""
        close = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        volume = np.array([100.0, 200.0, 50.0, 300.0, 100.0])
        high = close + 1.0
        low = close - 1.0
        result = OBV().compute({"close": close, "volume": volume, "high": high, "low": low})
        for i in range(1, len(result.values)):
            assert result.values[i] == result.values[i - 1]
        assert result.values[0] == 100.0  # volume[0]=100 > 0 und close[0]>0 → OBV[0]=volume[0]

    def test_obv_mixed_trend(self) -> None:
        """OBV bei gemischtem Trend steigt und fällt."""
        close = np.array([100.0, 101.0, 100.0, 102.0, 101.0])
        volume = np.array([100.0, 100.0, 50.0, 100.0, 50.0])
        high = close + 1.0
        low = close - 1.0
        result = OBV().compute({"close": close, "volume": volume, "high": high, "low": low})
        # 100, 200, 150, 250, 200
        assert result.values[0] == 100.0
        assert result.values[1] > result.values[0]  # +100
        assert result.values[2] < result.values[1]  # -50
        assert result.values[3] > result.values[2]  # +100
        assert result.values[4] < result.values[3]  # -50

    def test_obv_missing_close_raises(self) -> None:
        """ValueError bei fehlendem 'close'."""
        with pytest.raises(ValueError, match="Missing required"):
            OBV().compute({"volume": np.array([1.0])})

    def test_obv_missing_volume_raises(self) -> None:
        """ValueError bei fehlendem 'volume'."""
        close = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="Missing required"):
            OBV().compute({"close": close})

    def test_obv_zero_volume_first(self) -> None:
        """Erster OBV-Wert mit volume=0 ergibt 0."""
        close = np.array([100.0, 101.0])
        volume = np.array([0.0, 100.0])
        high = close + 1.0
        low = close - 1.0
        result = OBV().compute({"close": close, "volume": volume, "high": high, "low": low})
        assert result.values[0] == 0.0

    def test_obv_constant_data(self) -> None:
        """OBV auf konstanten Daten bleibt gleich."""
        n = 30
        close = np.full(n, 100.0)
        volume = np.full(n, 500.0)
        high = close + 1.0
        low = close - 1.0
        result = OBV().compute({"close": close, "volume": volume, "high": high, "low": low})
        for i in range(1, n):
            assert result.values[i] == result.values[i - 1]


class TestVWAP:
    """Testet Volume-Weighted Average Price."""

    def test_vwap_returns_result(self, ohlcv_data: dict[str, np.ndarray]) -> None:
        """Ergebnis hat Namen 'VWAP' und korrekte Länge."""
        result = VWAP().compute(ohlcv_data)
        assert result.name == "VWAP"
        assert len(result.values) == 100

    def test_vwap_first_value_correct(self) -> None:
        """Erster VWAP-Wert = typischer Preis des ersten Bars."""
        close = np.array([100.0])
        high = np.array([101.0])
        low = np.array([99.0])
        volume = np.array([1000.0])
        result = VWAP().compute({"close": close, "high": high, "low": low, "volume": volume})
        # typical_price = (100 + 101 + 99) / 3 = 100.0
        assert result.values[0] == 100.0

    def test_vwap_length_matches(self) -> None:
        """VWAP hat gleiche Länge wie Input."""
        n = 50
        rng = np.random.RandomState(42)
        close = 100.0 + np.cumsum(rng.randn(n) * 0.5)
        high = close + np.abs(rng.randn(n) * 0.3)
        low = close - np.abs(rng.randn(n) * 0.3)
        volume = np.ones(n)
        result = VWAP().compute({"close": close, "high": high, "low": low, "volume": volume})
        assert len(result.values) == n

    def test_vwap_missing_high_raises(self) -> None:
        """ValueError bei fehlendem 'high'."""
        close = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="Missing required"):
            VWAP().compute({"close": close, "low": close, "volume": np.ones(2)})

    def test_vwap_missing_low_raises(self) -> None:
        """ValueError bei fehlendem 'low'."""
        close = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="Missing required"):
            VWAP().compute({"close": close, "high": close, "volume": np.ones(2)})

    def test_vwap_missing_volume_raises(self) -> None:
        """ValueError bei fehlendem 'volume'."""
        close = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="Missing required"):
            VWAP().compute({"close": close, "high": close, "low": close})

    def test_vwap_ascending_prices(self) -> None:
        """VWAP steigt mit steigenden Preisen."""
        n = 30
        close = np.array([100.0 + i * 2.0 for i in range(n)])
        high = close + 1.0
        low = close - 1.0
        volume = np.ones(n)
        result = VWAP().compute({"close": close, "high": high, "low": low, "volume": volume})
        # Der VWAP sollte im Allgemeinen steigen (monoton nicht streng)
        valid = result.values[~np.isnan(result.values)]
        assert len(valid) > 0
        assert valid[-1] > valid[0]

    def test_vwap_constant_prices(self) -> None:
        """VWAP bleibt konstant bei konstanten Daten."""
        n = 30
        close = np.full(n, 100.0)
        high = np.full(n, 101.0)
        low = np.full(n, 99.0)
        volume = np.full(n, 500.0)
        result = VWAP().compute({"close": close, "high": high, "low": low, "volume": volume})
        assert np.allclose(result.values[1:], result.values[0])

    def test_vwap_values_reasonable(self, ohlcv_data: dict[str, np.ndarray]) -> None:
        """VWAP-Werte liegen in vernünftigen Bereichen."""
        result = VWAP().compute(ohlcv_data)
        valid = result.values[~np.isnan(result.values)]
        if len(valid) > 0:
            assert valid.min() < 200  # Unser Daten-Range ist ~90-115
            assert valid.max() > 90
