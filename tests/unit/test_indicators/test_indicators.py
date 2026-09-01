"""Tests - Technische Indikatoren.

Testet SMA, EMA, RSI, MACD, ADX, BollingerBands, Stochastic, ATR, OBV, VWAP.
"""

from __future__ import annotations

import numpy as np
import pytest
from packages.indicators import (
    ADX,
    ATR,
    EMA,
    MACD,
    OBV,
    RSI,
    SMA,
    VWAP,
    BollingerBands,
    StochasticOscillator,
)


@pytest.fixture
def sample_data() -> dict[str, np.ndarray]:
    """Erstellt synthetische Marktdaten für Tests."""
    np.random.seed(42)
    n = 100
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    volume = np.abs(np.random.randn(n) * 1000) + 500
    return {
        "open": close - np.random.randn(n) * 0.2,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


class TestBase:
    """Testet Indicator-Basisklasse."""

    def test_validate_data_missing_key(self, sample_data: dict[str, np.ndarray]) -> None:
        from packages.indicators.base import Indicator

        class Dummy(Indicator):
            name = "dummy"

            def compute(self, data: dict[str, np.ndarray]) -> Indicator.IndicatorResult:  # type: ignore[name-defined]
                return Indicator.IndicatorResult(name="dummy", values=np.array([0.0]))

        with pytest.raises(ValueError, match="Missing required"):
            Dummy()._validate_data({"close": sample_data["close"]}, ["open", "volume"])

    def test_check_lengths_inconsistent(self) -> None:
        from packages.indicators.base import Indicator

        data = {"close": np.array([1.0, 2.0]), "volume": np.array([1.0, 2.0, 3.0])}
        with pytest.raises(ValueError, match="Inconsistent array lengths"):
            Indicator._check_lengths(data)


class TestSMA:
    """Testet Simple Moving Average."""

    def test_defaults(self, sample_data: dict[str, np.ndarray]) -> None:
        sma = SMA(period=20)
        result = sma.compute(sample_data)
        assert result.name == "SMA"
        assert len(result.values) == 100
        assert np.isnan(result.values[:19]).all()
        assert not np.isnan(result.values[19])

    def test_period_5(self, sample_data: dict[str, np.ndarray]) -> None:
        sma = SMA(period=5)
        result = sma.compute(sample_data)
        assert len(result.values) == 100
        assert np.isnan(result.values[:4]).all()
        assert not np.isnan(result.values[4])

    def test_constant_values(self) -> None:
        close = np.full(30, 100.0)
        data = {"close": close, "volume": np.ones(30)}
        sma = SMA(period=5)
        result = sma.compute(data)
        assert np.allclose(result.values[4:], 100.0)

    def test_too_short(self) -> None:
        close = np.array([100.0, 101.0, 102.0])
        with pytest.raises(ValueError):
            SMA(period=10).compute({"close": close, "volume": np.ones(3)})


class TestEMA:
    """Testet Exponential Moving Average."""

    def test_defaults(self, sample_data: dict[str, np.ndarray]) -> None:
        ema = EMA(period=20)
        result = ema.compute(sample_data)
        assert result.name == "EMA"
        assert len(result.values) == 100

    def test_period_5(self, sample_data: dict[str, np.ndarray]) -> None:
        ema = EMA(period=5)
        result = ema.compute(sample_data)
        assert len(result.values) == 100
        assert not np.isnan(result.values[4])

    def test_constant_values(self) -> None:
        close = np.full(30, 100.0)
        ema = EMA(period=5)
        result = ema.compute({"close": close, "volume": np.ones(30)})
        assert np.allclose(result.values[4:], 100.0)


class TestRSI:
    """Testet Relative Strength Index."""

    def test_defaults(self, sample_data: dict[str, np.ndarray]) -> None:
        rsi = RSI(period=14)
        result = rsi.compute(sample_data)
        assert result.name == "RSI"
        assert len(result.values) == 100
        assert np.isnan(result.values[:14]).all()
        assert not np.isnan(result.values[14])

    def test_range(self, sample_data: dict[str, np.ndarray]) -> None:
        rsi = RSI(period=14)
        result = rsi.compute(sample_data)
        valid = result.values[~np.isnan(result.values)]
        assert np.all(valid >= 0)
        assert np.all(valid <= 100)

    def test_uptrend(self) -> None:
        close = np.array([100.0 + i * 0.5 for i in range(50)])
        result = RSI().compute({"close": close, "volume": np.ones(50)})
        valid = result.values[~np.isnan(result.values)]
        assert len(valid) > 0
        assert valid[-1] > 50

    def test_downtrend(self) -> None:
        close = np.array([100.0 - i * 0.5 for i in range(50)])
        result = RSI().compute({"close": close, "volume": np.ones(50)})
        valid = result.values[~np.isnan(result.values)]
        assert len(valid) > 0
        assert valid[-1] < 50

    def test_too_short(self) -> None:
        close = np.array([100.0] * 10)
        with pytest.raises(ValueError):
            RSI().compute({"close": close, "volume": np.ones(10)})


class TestMACD:
    """Testet MACD."""

    def test_defaults(self, sample_data: dict[str, np.ndarray]) -> None:
        macd = MACD()
        result = macd.compute(sample_data)
        assert result.name == "MACD"
        assert len(result.values) == 100

    def test_metadata(self, sample_data: dict[str, np.ndarray]) -> None:
        macd = MACD(fast_period=12, slow_period=26, signal_period=9)
        result = macd.compute(sample_data)
        meta = result.metadata
        assert meta["fast_period"] == 12
        assert meta["slow_period"] == 26
        assert meta["signal_period"] == 9

    def test_too_short(self) -> None:
        close = np.array([100.0] * 10)
        with pytest.raises(ValueError):
            MACD().compute({"close": close, "volume": np.ones(10)})


class TestBollingerBands:
    """Testet Bollinger Bands."""

    def test_defaults(self, sample_data: dict[str, np.ndarray]) -> None:
        bb = BollingerBands()
        result = bb.compute(sample_data)
        assert result.name == "BollingerBands"
        # Result is 2D: [upper, middle, lower] per row
        assert result.values.shape == (100, 3)

    def test_upper_above_middle(self, sample_data: dict[str, np.ndarray]) -> None:
        bb = BollingerBands()
        result = bb.compute(sample_data)
        valid = result.values[~np.isnan(result.values[:, 0])]
        assert len(valid) > 0
        assert np.all(valid[:, 0] >= valid[:, 1])
        assert np.all(valid[:, 1] >= valid[:, 2])

    def test_custom_std(self) -> None:
        close = np.full(40, 100.0)
        bb = BollingerBands(period=20, std_dev=3.0)
        result = bb.compute({"close": close, "volume": np.ones(40)})
        valid = result.values[~np.isnan(result.values[:, 0])]
        # With constant data, width should be 0
        assert np.allclose(valid[:, 0] - valid[:, 2], 0.0)

    def test_too_short(self) -> None:
        close = np.array([100.0] * 10)
        with pytest.raises(ValueError):
            BollingerBands().compute({"close": close, "volume": np.ones(10)})


class TestADX:
    """Testet Average Directional Index."""

    def test_defaults(self, sample_data: dict[str, np.ndarray]) -> None:
        adx = ADX()
        result = adx.compute(sample_data)
        assert result.name == "ADX"
        assert len(result.values) == 100

    def test_range(self, sample_data: dict[str, np.ndarray]) -> None:
        adx = ADX()
        result = adx.compute(sample_data)
        valid = result.values[~np.isnan(result.values)]
        assert np.all(valid >= 0)
        assert np.all(valid <= 100)

    def test_too_short(self) -> None:
        n = 10
        close = np.random.randn(n)
        high = close + 0.1
        low = close - 0.1
        with pytest.raises(ValueError):
            ADX().compute({"high": high, "low": low, "close": close, "volume": np.ones(n)})


class TestStochasticOscillator:
    """Testet Stochastic Oscillator."""

    def test_defaults(self, sample_data: dict[str, np.ndarray]) -> None:
        stoch = StochasticOscillator()
        result = stoch.compute(sample_data)
        assert result.name == "Stochastic"
        assert len(result.values) == 100

    def test_range(self, sample_data: dict[str, np.ndarray]) -> None:
        stoch = StochasticOscillator()
        result = stoch.compute(sample_data)
        valid = result.values[~np.isnan(result.values)]
        assert np.all(valid >= 0)
        assert np.all(valid <= 100)


class TestATR:
    """Testet Average True Range."""

    def test_defaults(self, sample_data: dict[str, np.ndarray]) -> None:
        atr = ATR()
        result = atr.compute(sample_data)
        assert result.name == "ATR"
        assert len(result.values) == 100

    def test_positive_values(self, sample_data: dict[str, np.ndarray]) -> None:
        atr = ATR()
        result = atr.compute(sample_data)
        valid = result.values[~np.isnan(result.values)]
        assert np.all(valid > 0)

    def test_constant_high_low(self) -> None:
        high = np.full(30, 101.0)
        low = np.full(30, 99.0)
        close = np.full(30, 100.0)
        result = ATR().compute({"high": high, "low": low, "close": close, "volume": np.ones(30)})
        valid = result.values[~np.isnan(result.values)]
        # True range should be 2.0 (high - low = 101 - 99)
        assert np.allclose(valid, 2.0)


class TestOBV:
    """Testet On-Balance Volume."""

    def test_defaults(self, sample_data: dict[str, np.ndarray]) -> None:
        obv = OBV()
        result = obv.compute(sample_data)
        assert result.name == "OBV"
        assert len(result.values) == 100

    def test_uptrend_increases(self) -> None:
        close = np.array([100.0, 101.0, 102.0, 103.0])
        volume = np.array([100.0, 100.0, 100.0, 100.0])
        result = OBV().compute({"close": close, "volume": volume, "high": close, "low": close})
        assert result.values[0] == 100.0
        assert result.values[1] > result.values[0]
        assert result.values[2] > result.values[1]

    def test_downtrend_decreases(self) -> None:
        close = np.array([103.0, 102.0, 101.0, 100.0])
        volume = np.array([100.0, 100.0, 100.0, 100.0])
        result = OBV().compute({"close": close, "volume": volume, "high": close, "low": close})
        assert result.values[0] == 100.0
        assert result.values[1] < result.values[0]


class TestVWAP:
    """Testet Volume-Weighted Average Price."""

    def test_defaults(self, sample_data: dict[str, np.ndarray]) -> None:
        vwap = VWAP()
        result = vwap.compute(sample_data)
        assert result.name == "VWAP"
        assert len(result.values) == 100

    def test_first_value(self) -> None:
        close = np.array([100.0])
        high = np.array([101.0])
        low = np.array([99.0])
        volume = np.array([1000.0])
        result = VWAP().compute({"close": close, "high": high, "low": low, "volume": volume})
        # First VWAP = typical_price = (100 + 101 + 99) / 3 = 100.0
        assert result.values[0] == 100.0
