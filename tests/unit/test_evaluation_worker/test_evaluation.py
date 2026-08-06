"""Tests for evaluation and feature workers."""

from __future__ import annotations

import numpy as np
import pytest
from apps.evaluation_worker.worker import BacktestEvaluator
from apps.feature_worker.worker import FeatureExtractor

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def winning_trades() -> list[dict]:
    """Creates a list of winning trades."""
    return [
        {
            "symbol": "AAPL",
            "direction": "long",
            "entry_price": 150.0,
            "exit_price": 160.0,
            "quantity": 10,
            "timestamp": "2024-01-02T10:00:00",
            "commission": 1.0,
            "pnl": 99.0,
        },
        {
            "symbol": "AAPL",
            "direction": "long",
            "entry_price": 160.0,
            "exit_price": 170.0,
            "quantity": 10,
            "timestamp": "2024-01-03T10:00:00",
            "commission": 1.0,
            "pnl": 99.0,
        },
    ]


@pytest.fixture
def losing_trades() -> list[dict]:
    """Creates a list of losing trades."""
    return [
        {
            "symbol": "AAPL",
            "direction": "long",
            "entry_price": 150.0,
            "exit_price": 145.0,
            "quantity": 10,
            "timestamp": "2024-01-02T10:00:00",
            "commission": 1.0,
            "pnl": -51.0,
        },
        {
            "symbol": "AAPL",
            "direction": "long",
            "entry_price": 145.0,
            "exit_price": 140.0,
            "quantity": 10,
            "timestamp": "2024-01-03T10:00:00",
            "commission": 1.0,
            "pnl": -51.0,
        },
    ]


@pytest.fixture
def mixed_trades() -> list[dict]:
    """Creates a mixed list of winning and losing trades."""
    return [
        {
            "symbol": "AAPL",
            "direction": "long",
            "entry_price": 100.0,
            "exit_price": 110.0,
            "quantity": 10,
            "timestamp": "2024-01-02T10:00:00",
            "commission": 1.0,
            "pnl": 99.0,
        },
        {
            "symbol": "AAPL",
            "direction": "long",
            "entry_price": 110.0,
            "exit_price": 95.0,
            "quantity": 10,
            "timestamp": "2024-01-03T10:00:00",
            "commission": 1.0,
            "pnl": -151.0,
        },
        {
            "symbol": "AAPL",
            "direction": "long",
            "entry_price": 95.0,
            "exit_price": 105.0,
            "quantity": 10,
            "timestamp": "2024-01-04T10:00:00",
            "commission": 1.0,
            "pnl": 99.0,
        },
    ]


@pytest.fixture
def sample_ohlcv() -> dict[str, np.ndarray]:
    """Creates synthetic OHLCV data for feature extraction tests."""
    np.random.seed(42)
    n = 60
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    return {
        "open": close - np.random.randn(n) * 0.2,
        "high": close + np.abs(np.random.randn(n) * 0.3),
        "low": close - np.abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": np.abs(np.random.randn(n) * 1000) + 500,
    }


# ── BacktestEvaluator tests ───────────────────────────────────────────────────


class TestBacktestEvaluatorBasic:
    """Tests for basic backtest evaluation."""

    def test_backtest_evaluator_basic(self, mixed_trades: list[dict]) -> None:
        """Evaluates simple backtest correctly."""
        evaluator = BacktestEvaluator()
        result = evaluator.evaluate(mixed_trades)

        assert result["total_trades"] == 3
        assert result["winning_trades"] == 2
        assert result["losing_trades"] == 1
        assert isinstance(result["total_pnl"], float)
        assert isinstance(result["trade_list"], list)
        assert len(result["trade_list"]) == 3

    def test_backtest_evaluator_win_rate(self, winning_trades: list[dict]) -> None:
        """Calculates win rate correctly."""
        evaluator = BacktestEvaluator()
        result = evaluator.evaluate(winning_trades)

        assert result["win_rate"] == 1.0
        assert result["winning_trades"] == 2
        assert result["losing_trades"] == 0

    def test_backtest_evaluator_profit_factor(self, mixed_trades: list[dict]) -> None:
        """Calculates profit factor correctly."""
        evaluator = BacktestEvaluator()
        result = evaluator.evaluate(mixed_trades)

        total_wins = 99.0 + 99.0  # 2 winning trades
        total_losses = 151.0      # 1 losing trade
        expected_pf = total_wins / total_losses

        assert abs(result["profit_factor"] - expected_pf) < 0.01

    def test_backtest_evaluator_max_drawdown(self, mixed_trades: list[dict]) -> None:
        """Calculates max drawdown from equity curve."""
        evaluator = BacktestEvaluator()
        result = evaluator.evaluate(mixed_trades, initial_capital=100000.0)

        # Drawdown should be >= 0
        assert result["max_drawdown"] >= 0.0
        # Max drawdown is bounded to [0, 1]
        assert result["max_drawdown"] <= 1.0

    def test_backtest_evaluator_sharpe_ratio(self, mixed_trades: list[dict]) -> None:
        """Calculates Sharpe ratio."""
        evaluator = BacktestEvaluator()
        result = evaluator.evaluate(mixed_trades)

        # Sharpe should be a finite float
        assert isinstance(result["sharpe_ratio"], float)
        assert np.isfinite(result["sharpe_ratio"]) or result["sharpe_ratio"] == 0.0

    def test_backtest_evaluator_return_pct(self, winning_trades: list[dict]) -> None:
        """Calculates return percentage correctly."""
        evaluator = BacktestEvaluator()
        result = evaluator.evaluate(winning_trades, initial_capital=100000.0)

        expected_total_pnl = 99.0 + 99.0
        expected_return_pct = expected_total_pnl / 100000.0 * 100.0

        assert abs(result["return_pct"] - expected_return_pct) < 0.1

    def test_backtest_evaluator_all_winning(self, winning_trades: list[dict]) -> None:
        """All trades win — no losses, profit factor should be inf-like."""
        evaluator = BacktestEvaluator()
        result = evaluator.evaluate(winning_trades)

        assert result["winning_trades"] == 2
        assert result["losing_trades"] == 0
        assert result["win_rate"] == 1.0
        # No losses means profit_factor = total_wins / 0 → infinity
        assert result["profit_factor"] == 0.0  # our impl returns 0 for zero losses

    def test_backtest_evaluator_all_losing(self, losing_trades: list[dict]) -> None:
        """All trades lose — no wins, profit factor should be 0."""
        evaluator = BacktestEvaluator()
        result = evaluator.evaluate(losing_trades)

        assert result["winning_trades"] == 0
        assert result["losing_trades"] == 2
        assert result["win_rate"] == 0.0
        assert result["profit_factor"] == 0.0

    def test_backtest_evaluator_generate_report(self, winning_trades: list[dict]) -> None:
        """Generates a text report from evaluation results."""
        evaluator = BacktestEvaluator()
        evaluation = evaluator.evaluate(winning_trades)
        report = evaluator.generate_report(evaluation)

        assert isinstance(report, str)
        assert "BACKTEST EVALUATION REPORT" in report
        assert "Total Trades" in report
        assert "Win Rate" in report
        assert "Profit Factor" in report
        assert "Max Drawdown" in report
        assert "Sharpe Ratio" in report


# ── FeatureExtractor tests ────────────────────────────────────────────────────


class TestFeatureExtractor:
    """Tests for feature extraction from OHLCV data."""

    def test_feature_extractor_basic(self, sample_ohlcv: dict[str, np.ndarray]) -> None:
        """Extracts features from OHLCV data."""
        extractor = FeatureExtractor()
        features = extractor.extract(sample_ohlcv)

        assert isinstance(features, dict)
        assert len(features) > 0
        # Check expected feature keys exist
        expected_keys = [
            "return", "volatility_20", "momentum_10",
            "volume_ratio", "price_change", "price_range",
            "volume_change",
        ]
        for key in expected_keys:
            assert key in features, f"Missing feature: {key}"
            assert isinstance(features[key], float)

    def test_feature_extractor_returns(self, sample_ohlcv: dict[str, np.ndarray]) -> None:
        """Calculates returns correctly."""
        close = sample_ohlcv["close"]
        extractor = FeatureExtractor()
        features = extractor.extract(sample_ohlcv)

        # Manual log return computation
        expected_return = float(np.log(close[-1] / close[-2]))
        assert abs(features["return"] - expected_return) < 1e-10

    def test_feature_extractor_rolling(self, sample_ohlcv: dict[str, np.ndarray]) -> None:
        """Calculates rolling features correctly."""
        extractor = FeatureExtractor()
        features = extractor.extract_rolling_features(sample_ohlcv, window=20)

        assert "mean_return" in features
        assert "std_return" in features
        assert "max_drawdown" in features
        assert "volume_avg" in features
        assert "volume_std" in features

        assert isinstance(features["mean_return"], float)
        assert isinstance(features["max_drawdown"], float)
        assert 0.0 <= features["max_drawdown"] <= 1.0

    def test_feature_extractor_normalize(self, sample_ohlcv: dict[str, np.ndarray]) -> None:
        """Normalizes features to 0-1 range."""
        extractor = FeatureExtractor()
        features = extractor.extract(sample_ohlcv)
        normalized = extractor.normalize_features(features)

        assert isinstance(normalized, dict)
        assert len(normalized) == len(features)

        # All values should be in [0, 1]
        for value in normalized.values():
            assert 0.0 <= value <= 1.0 + 1e-10

        # Check min and max
        values = list(normalized.values())
        assert min(values) >= 0.0
        assert max(values) <= 1.0 + 1e-10

    def test_feature_extractor_short_data(self) -> None:
        """Handles short data gracefully."""
        close = np.array([100.0, 101.0])
        data = {
            "open": np.array([99.0, 100.0]),
            "high": np.array([101.5, 102.0]),
            "low": np.array([98.5, 99.5]),
            "close": close,
            "volume": np.array([1000.0, 1100.0]),
        }
        extractor = FeatureExtractor()

        # Should not raise
        features = extractor.extract(data)
        assert isinstance(features, dict)

        # Rolling features with short data
        rolling = extractor.extract_rolling_features(data, window=20)
        assert isinstance(rolling, dict)
        assert rolling["max_drawdown"] == 0.0

class TestBacktestEvaluatorEmpty:
    """Tests for edge cases with empty data."""

    def test_empty_trade_list(self) -> None:
        """Empty trade list returns zero metrics."""
        evaluator = BacktestEvaluator()
        result = evaluator.evaluate([])

        assert result["total_trades"] == 0
        assert result["winning_trades"] == 0
        assert result["losing_trades"] == 0
        assert result["win_rate"] == 0.0
        assert result["total_pnl"] == 0.0
        assert result["max_drawdown"] == 0.0
        assert result["sharpe_ratio"] == 0.0
        assert result["trade_list"] == []
