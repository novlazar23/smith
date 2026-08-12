"""Tests for baseline strategies (packages/validation/baselines/)."""

import pytest
from packages.validation.baselines.base import (
    Baseline,
    BaselinePrediction,
)
from packages.validation.baselines.buy_hold import BuyHoldBaseline
from packages.validation.baselines.ma_cross import MACrossBaseline
from packages.validation.baselines.momentum import MomentumBaseline
from packages.validation.baselines.regime import RegimeBaseline
from packages.validation.baselines.rsi import RSIBaseline

# ── fixture helpers ──────────────────────────────────────────────────


def _bull_prices(n: int = 60, start: float = 100.0) -> list[float]:
    """Monotonically increasing price series."""
    return [start + i * 0.5 for i in range(n)]


def _bear_prices(n: int = 60, start: float = 100.0) -> list[float]:
    """Monotonically decreasing price series."""
    return [start - i * 0.5 for i in range(n)]


def _sideways_prices(n: int = 30, start: float = 100.0) -> list[float]:
    """Flat price series with small noise."""
    return [start + (i % 3) * 0.1 for i in range(n)]


# ── 1. BuyHold ───────────────────────────────────────────────────────


class TestBuyHoldBaseline:
    def test_always_up(self) -> None:
        bh = BuyHoldBaseline()
        pred = bh.predict(features={})
        assert pred.probabilities["UP"] == pytest.approx(1.0)
        assert pred.probabilities["DOWN"] == pytest.approx(0.0)
        assert pred.probabilities["RANGE"] == pytest.approx(0.0)

    def test_confidence_max(self) -> None:
        bh = BuyHoldBaseline()
        pred = bh.predict(features={})
        assert pred.confidence == pytest.approx(1.0)

    def test_baseline_id(self) -> None:
        bh = BuyHoldBaseline()
        assert bh.baseline_id == "buy_hold"


# ── 2. MACross ───────────────────────────────────────────────────────


class TestMACrossBaseline:
    def test_bullish_signal(self) -> None:
        ma = MACrossBaseline(short_window=5, long_window=10)
        prices = _bull_prices(n=20)
        pred = ma.predict(features={}, historical_prices=prices)
        assert pred.probabilities["UP"] > pred.probabilities["DOWN"]
        assert "Bullish" in pred.signal

    def test_bearish_signal(self) -> None:
        ma = MACrossBaseline(short_window=5, long_window=10)
        prices = _bear_prices(n=20)
        pred = ma.predict(features={}, historical_prices=prices)
        assert pred.probabilities["DOWN"] > pred.probabilities["UP"]
        assert "Bearish" in pred.signal

    def test_insufficient_data(self) -> None:
        ma = MACrossBaseline(short_window=5, long_window=10)
        pred = ma.predict(
            features={},
            historical_prices=[100.0, 101.0],
        )
        assert pred.probabilities["UP"] == pytest.approx(0.33)
        assert pred.probabilities["DOWN"] == pytest.approx(0.33)
        assert pred.probabilities["RANGE"] == pytest.approx(0.34)
        assert "Insufficient data" in pred.signal

    def test_no_prices(self) -> None:
        ma = MACrossBaseline(short_window=5, long_window=10)
        pred = ma.predict(features={})
        assert pred.probabilities["UP"] == pytest.approx(0.33)

    def test_baseline_id(self) -> None:
        ma = MACrossBaseline()
        assert ma.baseline_id == "ma_cross"


# ── 3. Momentum ──────────────────────────────────────────────────────


class TestMomentumBaseline:
    def test_positive_momentum(self) -> None:
        mom = MomentumBaseline(lookback=5)
        prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
        pred = mom.predict(features={}, historical_prices=prices)
        assert pred.probabilities["UP"] > pred.probabilities["DOWN"]
        assert "Positive" in pred.signal

    def test_negative_momentum(self) -> None:
        mom = MomentumBaseline(lookback=5)
        prices = [105.0, 104.0, 103.0, 102.0, 101.0, 100.0]
        pred = mom.predict(features={}, historical_prices=prices)
        assert pred.probabilities["DOWN"] > pred.probabilities["UP"]
        assert "Negative" in pred.signal

    def test_neutral_momentum(self) -> None:
        mom = MomentumBaseline(lookback=5)
        prices = [100.0, 100.001, 100.002, 100.003, 100.004, 100.0045]
        pred = mom.predict(features={}, historical_prices=prices)
        assert pred.probabilities["RANGE"] == pytest.approx(0.34)
        assert "Neutral" in pred.signal

    def test_insufficient_data(self) -> None:
        mom = MomentumBaseline(lookback=10)
        pred = mom.predict(features={}, historical_prices=[100.0])
        assert pred.probabilities["UP"] == pytest.approx(0.33)
        assert "Insufficient data" in pred.signal

    def test_baseline_id(self) -> None:
        mom = MomentumBaseline()
        assert mom.baseline_id == "momentum"


# ── 4. RSI ───────────────────────────────────────────────────────────


class TestRSIBaseline:
    def test_oversold_signal(self) -> None:
        rsi = RSIBaseline(period=10, oversold=30.0, overbought=70.0)
        # Steeply falling prices -> low RSI
        prices = [100.0, 95.0, 90.0, 85.0, 80.0, 75.0, 70.0, 65.0, 60.0, 55.0, 50.0]
        pred = rsi.predict(features={}, historical_prices=prices)
        assert pred.probabilities["UP"] > pred.probabilities["DOWN"]
        assert "Oversold" in pred.signal

    def test_overbought_signal(self) -> None:
        rsi = RSIBaseline(period=10, oversold=30.0, overbought=70.0)
        # Steeply rising prices -> high RSI
        prices = [50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0]
        pred = rsi.predict(features={}, historical_prices=prices)
        assert pred.probabilities["DOWN"] > pred.probabilities["UP"]
        assert "Overbought" in pred.signal

    def test_neutral_rsi(self) -> None:
        rsi = RSIBaseline(period=5)
        # Sideways prices -> RSI ~50
        prices = _sideways_prices(n=20)
        pred = rsi.predict(features={}, historical_prices=prices)
        assert "Neutral" in pred.signal

    def test_insufficient_prices(self) -> None:
        rsi = RSIBaseline(period=14)
        pred = rsi.predict(features={}, historical_prices=[100.0, 101.0])
        assert pred.signal == "RSI: Neutral (50.0)"

    def test_no_prices(self) -> None:
        rsi = RSIBaseline(period=14)
        pred = rsi.predict(features={})
        assert pred.signal == "RSI: Neutral (50.0)"

    def test_baseline_id(self) -> None:
        rsi = RSIBaseline()
        assert rsi.baseline_id == "rsi"


# ── 5. Regime ────────────────────────────────────────────────────────


class TestRegimeBaseline:
    def test_trend_up(self) -> None:
        reg = RegimeBaseline(trend_window=10, trend_threshold=0.0001)
        prices = [100.0 + i * 0.5 for i in range(20)]
        pred = reg.predict(features={}, historical_prices=prices)
        assert "Trend UP" in pred.signal
        assert pred.probabilities["UP"] > pred.probabilities["DOWN"]

    def test_trend_down(self) -> None:
        reg = RegimeBaseline(trend_window=10, trend_threshold=0.0001)
        prices = [100.0 - i * 0.5 for i in range(20)]
        pred = reg.predict(features={}, historical_prices=prices)
        assert "Trend DOWN" in pred.signal
        assert pred.probabilities["DOWN"] > pred.probabilities["UP"]

    def test_insufficient_data(self) -> None:
        reg = RegimeBaseline(trend_window=50)
        prices = [100.0] * 10
        pred = reg.predict(features={}, historical_prices=prices)
        assert "Trend" not in pred.signal
        assert "Mean-revert" not in pred.signal

    def test_no_prices(self) -> None:
        reg = RegimeBaseline()
        pred = reg.predict(features={})
        assert "Insufficient data" in pred.signal

    def test_baseline_id(self) -> None:
        reg = RegimeBaseline()
        assert reg.baseline_id == "regime"


# ── 6. Common / shared tests ─────────────────────────────────────────


class TestBaselineIdProperty:
    """All baselines must expose baseline_id."""

    @pytest.mark.parametrize(
        "baseline",
        [
            BuyHoldBaseline(),
            MACrossBaseline(),
            MomentumBaseline(),
            RSIBaseline(),
            RegimeBaseline(),
        ],
    )
    def test_baseline_id_exists(self, baseline: Baseline) -> None:
        assert isinstance(baseline.baseline_id, str)
        assert len(baseline.baseline_id) > 0

    @pytest.mark.parametrize(
        ("baseline", "expected_id"),
        [
            (BuyHoldBaseline(), "buy_hold"),
            (MACrossBaseline(), "ma_cross"),
            (MomentumBaseline(), "momentum"),
            (RSIBaseline(), "rsi"),
            (RegimeBaseline(), "regime"),
        ],
    )
    def test_baseline_id_values(self, baseline: Baseline, expected_id: str) -> (
        None
    ):
        assert baseline.baseline_id == expected_id


class TestProbabilitiesSum:
    """All probabilities must sum to ~1.0."""

    @pytest.mark.parametrize(
        ("baseline", "prices"),
        [
            (BuyHoldBaseline(), None),
            (MACrossBaseline(), _bull_prices(n=60)),
            (MomentumBaseline(), _bull_prices(n=20)),
            (RSIBaseline(), _bull_prices(n=30)),
            (RegimeBaseline(), _bull_prices(n=60)),
        ],
    )
    def test_probabilities_sum_to_one(self, baseline: Baseline, prices: list[float] | None) -> None:
        pred = baseline.predict(features={}, historical_prices=prices)
        total = sum(pred.probabilities.values())
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_buyhold_prob_sum(self) -> None:
        bh = BuyHoldBaseline()
        pred = bh.predict(features={})
        assert sum(pred.probabilities.values()) == pytest.approx(1.0)

    def test_ma_cross_prob_sum_bull(self) -> None:
        ma = MACrossBaseline()
        pred = ma.predict(features={}, historical_prices=_bull_prices(n=60))
        assert sum(pred.probabilities.values()) == pytest.approx(1.0)

    def test_momentum_prob_sum_bear(self) -> None:
        mom = MomentumBaseline()
        pred = mom.predict(features={}, historical_prices=_bear_prices(n=20))
        assert sum(pred.probabilities.values()) == pytest.approx(1.0)

    def test_rsi_prob_sum_oversold(self) -> None:
        rsi = RSIBaseline()
        prices = [100.0 - i for i in range(30)]
        pred = rsi.predict(features={}, historical_prices=prices)
        assert sum(pred.probabilities.values()) == pytest.approx(1.0)


class TestConfidenceRange:
    """All confidence values must be in [0, 1]."""

    @pytest.mark.parametrize(
        ("baseline", "prices"),
        [
            (BuyHoldBaseline(), None),
            (MACrossBaseline(), _bull_prices(n=60)),
            (MomentumBaseline(), _bull_prices(n=20)),
            (RSIBaseline(), _bull_prices(n=30)),
            (RegimeBaseline(), _bull_prices(n=60)),
        ],
    )
    def test_confidence_in_range(self, baseline: Baseline, prices: list[float] | None) -> None:
        pred = baseline.predict(features={}, historical_prices=prices)
        assert 0.0 <= pred.confidence <= 1.0


class TestBaselinePredictionImmutability:
    """BaselinePrediction must be frozen (immutable)."""

    def test_cannot_modify_attributes(self) -> None:
        pred = BaselinePrediction(
            baseline_id="test",
            baseline_version="1.0.0",
            probabilities={"UP": 0.5, "DOWN": 0.3, "RANGE": 0.2},
            confidence=0.7,
            signal="test signal",
        )
        with pytest.raises(Exception):
            pred.baseline_id = "changed"  # type: ignore[misc]

    def test_cannot_reassign_baseline_version(self) -> None:
        pred = BaselinePrediction(
            baseline_id="test",
            baseline_version="1.0.0",
            probabilities={"UP": 0.5, "DOWN": 0.3, "RANGE": 0.2},
            confidence=0.7,
        )
        with pytest.raises(Exception):
            pred.baseline_version = "2.0.0"  # type: ignore[misc]

    def test_cannot_modify_confidence(self) -> None:
        pred = BaselinePrediction(
            baseline_id="test",
            baseline_version="1.0.0",
            probabilities={"UP": 0.5, "DOWN": 0.3, "RANGE": 0.2},
            confidence=0.7,
        )
        with pytest.raises(Exception):
            pred.confidence = 0.99  # type: ignore[misc]

    def test_freeze_with_custom_signal(self) -> None:
        pred = BaselinePrediction(
            baseline_id="test",
            baseline_version="1.0.0",
            probabilities={"UP": 0.5, "DOWN": 0.3, "RANGE": 0.2},
            confidence=0.7,
            signal="custom",
        )
        assert pred.signal == "custom"
        with pytest.raises(Exception):
            pred.signal = "changed"  # type: ignore[misc]
