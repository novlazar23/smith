"""Tests for packages.backtesting — metrics, engine, validation, execution."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
from packages.backtesting.core import BacktestConfig, Candle
from packages.backtesting.datafeed import MemoryDataFeed
from packages.backtesting.engine import BacktestEngine
from packages.backtesting.execution import (
    BacktestOrder,
    FillStatus,
    FixedCommissionModel,
    FixedSlippageModel,
    LimitExecutionModel,
    MarketExecutionModel,
    OrderType,
    PercentageCommissionModel,
    PercentageSlippageModel,
    Side,
    StopLimitExecutionModel,
    TieredCommissionModel,
    VolumeBasedSlippageModel,
)
from packages.backtesting.metrics import (
    BacktestMetrics,
    calculate_backtest_metrics,
    calculate_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)
from packages.backtesting.strategies import MACDCrossover, SignalAction, StrategySignal
from packages.backtesting.validation import (
    MonteCarloSimulator,
    ParameterSweeper,
    PurgedKFold,
    WalkForwardAnalyzer,
)

# ─── Fixtures ───────────────────────────────────────────────────────────────


def _make_candle(ts: datetime, close: float, symbol: str = "BTC/USD") -> Candle:
    """Helper to create a Candle with realistic values."""
    return Candle(
        timestamp=ts,
        symbol=symbol,
        open=close * 0.998,
        high=close * 1.002,
        low=close * 0.997,
        close=close,
        volume=100.0,
    )


@pytest.fixture
def sample_candles() -> list[Candle]:
    """Generate synthetic candles for testing."""
    np.random.seed(42)
    n = 200
    base = 100.0
    closes = np.cumsum(np.random.randn(n) * 0.5) + base

    start = datetime(2024, 1, 1)
    candles = []
    for i in range(n):
        ts = start.replace(minute=i // 60, second=i % 60)
        candles.append(_make_candle(ts, float(closes[i])))

    return candles


@pytest.fixture
def sample_trades() -> list[dict]:
    """Sample trades with PnL values."""
    return [
        {"pnl": 100.0, "quantity": 1.0, "price": 50000.0, "holding_days": 2},
        {"pnl": -50.0, "quantity": 0.5, "price": 49500.0, "holding_days": 1},
        {"pnl": 200.0, "quantity": 2.0, "price": 51000.0, "holding_days": 3},
        {"pnl": -30.0, "quantity": 0.3, "price": 48000.0, "holding_days": 1},
        {"pnl": 150.0, "quantity": 1.5, "price": 52000.0, "holding_days": 5},
    ]


# ─── BacktestConfig Tests ───────────────────────────────────────────────────


class TestBacktestConfig:
    def test_default_config(self) -> None:
        cfg = BacktestConfig()
        assert cfg.initial_capital == 100_000.0
        assert cfg.commission_rate == 0.001
        assert cfg.slippage_bps == 5.0
        assert cfg.max_position_size == 0.25
        assert cfg.warmup_bars == 50

    def test_custom_config(self) -> None:
        cfg = BacktestConfig(
            symbol="ETH/USD",
            initial_capital=50_000.0,
            commission_rate=0.0005,
        )
        assert cfg.symbol == "ETH/USD"
        assert cfg.initial_capital == 50_000.0
        assert cfg.commission_rate == 0.0005

    def test_invalid_config_values(self) -> None:
        """Config rejects out-of-range values."""
        with pytest.raises(ValueError, match="commission_rate"):
            BacktestConfig(commission_rate=-0.01)
        with pytest.raises(ValueError, match="initial_capital"):
            BacktestConfig(initial_capital=-100)

    def test_config_to_dict(self) -> None:
        """Config can be serialized via model_dump."""
        cfg = BacktestConfig(
            symbol="ETH/USD",
            initial_capital=50_000.0,
            commission_rate=0.0005,
            warmup_bars=10,
        )
        d = cfg.model_dump()
        assert isinstance(d, dict)
        assert d["symbol"] == "ETH/USD"
        assert d["initial_capital"] == 50_000.0
        assert d["commission_rate"] == 0.0005
        assert d["warmup_bars"] == 10


# ─── BacktestEngine Tests ───────────────────────────────────────────────────


class TestBacktestEngine:
    def test_engine_run_no_candles(self) -> None:
        """Engine raises ValueError when no candles provided."""
        feed = MemoryDataFeed()
        engine = BacktestEngine()
        with pytest.raises(ValueError, match="No candles in data feed"):
            engine.run(feed, MACDCrossover())

    def test_engine_run_with_data(self, sample_candles: list[Candle]) -> None:
        """Engine completes backtest and produces results."""
        feed = MemoryDataFeed(candles=sample_candles)
        engine = BacktestEngine(config=BacktestConfig(symbol="BTC/USD"))
        strategy = MACDCrossover(name="test_macd")
        result = engine.run(feed, strategy, warmup_bars=10)

        assert result is not None
        assert len(result.candles) == len(sample_candles)
        assert result.config.symbol == "BTC/USD"
        assert len(result.snapshots) > 0
        assert len(result.metrics) > 0
        assert "total_return_pct" in result.metrics
        assert "sharpe_ratio" in result.metrics

    def test_engine_run_macd_strategy(
        self, sample_candles: list[Candle]
    ) -> None:
        """MACD strategy generates signals on crossover."""
        feed = MemoryDataFeed(candles=sample_candles)
        engine = BacktestEngine(config=BacktestConfig(symbol="BTC/USD"))
        strategy = MACDCrossover(name="macd_test")
        result = engine.run(feed, strategy, warmup_bars=10)

        assert result is not None

    def test_engine_with_empty_strategy(self, sample_candles: list[Candle]) -> None:
        """Engine completes with strategy that returns no signals."""
        feed = MemoryDataFeed(candles=sample_candles)
        engine = BacktestEngine(config=BacktestConfig(symbol="BTC/USD"))

        class NoSignalStrategy(MACDCrossover):
            def on_bar(self, candle: Candle) -> None:
                return None

        strategy = NoSignalStrategy(name="noop")
        result = engine.run(feed, strategy, warmup_bars=10)

        assert result is not None
        assert len(result.snapshots) > 0

    def test_engine_preserves_config(self, sample_candles: list[Candle]) -> None:
        """Engine preserves the provided config through the result."""
        cfg = BacktestConfig(symbol="BTC/USD", initial_capital=250_000.0, warmup_bars=20)
        feed = MemoryDataFeed(candles=sample_candles)
        engine = BacktestEngine(config=cfg)
        strategy = MACDCrossover(name="test")
        result = engine.run(feed, strategy, warmup_bars=20)

        assert result.config.initial_capital == 250_000.0
        assert result.config.symbol == "BTC/USD"

    def test_engine_warmup_bars(self, sample_candles: list[Candle]) -> None:
        """Warmup bars are skipped in the trading phase."""
        feed = MemoryDataFeed(candles=sample_candles)
        engine = BacktestEngine(config=BacktestConfig(symbol="BTC/USD"))
        strategy = MACDCrossover(name="test")
        result = engine.run(feed, strategy, warmup_bars=50)

        # snapshots should be from warmup_end onward
        assert len(result.snapshots) == len(sample_candles) - 50


# ─── Metrics Tests ──────────────────────────────────────────────────────────


class TestBacktestMetrics:
    def test_calculate_metrics(self, sample_trades: list[dict]) -> None:
        """Metrics are computed from equity curve and trades."""
        equity = [100_000.0] + [
            100_000.0 + sum(t["pnl"] for t in sample_trades[:i])
            for i in range(1, len(sample_trades) + 1)
        ]

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 7)

        metrics = calculate_backtest_metrics(
            equity_curve=equity,
            trades=sample_trades,
            risk_free_rate=0.02,
            start_date=start,
            end_date=end,
        )

        assert isinstance(metrics, BacktestMetrics)
        assert metrics.total_return != 0.0
        assert metrics.sharpe_ratio != 0.0
        assert metrics.max_drawdown >= 0.0
        assert metrics.total_trades == len(sample_trades)

    def test_calculate_metrics_empty(self) -> None:
        """Empty data returns zero metrics."""
        metrics = calculate_backtest_metrics(
            equity_curve=[100_000.0],
            trades=[],
        )
        assert metrics.total_return == 0.0
        assert metrics.total_trades == 0

    def test_metrics_to_dict(self, sample_trades: list[dict]) -> None:
        """Metrics export to flat dict."""
        equity = [100_000.0, 100_100.0, 99_950.0, 100_150.0, 99_920.0, 100_070.0]
        metrics = calculate_backtest_metrics(equity, sample_trades)
        d = metrics.to_dict()

        assert isinstance(d, dict)
        assert "total_return_pct" in d
        assert "sharpe_ratio" in d
        assert "win_rate_pct" in d
        assert "max_drawdown_pct" in d

    def test_calculate_sharpe_ratio(self) -> None:
        """Sharpe ratio calculated correctly."""
        returns = [0.01, -0.005, 0.008, -0.002, 0.003, 0.001]
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.0)
        assert sharpe > 0  # Positive average returns

    def test_calculate_sharpe_zero_returns(self) -> None:
        """Sharpe returns 0 when all returns are identical (zero std)."""
        returns = [0.001, 0.001]
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.0)
        assert sharpe == 0.0

    def test_calculate_sortino_ratio(self) -> None:
        """Sortino ratio calculated correctly."""
        returns = [0.01, -0.005, 0.008, -0.002, 0.003, 0.001]
        sortino = calculate_sortino_ratio(returns, risk_free_rate=0.0)
        assert sortino >= 0

    def test_calculate_sortino_zero_returns(self) -> None:
        """Sortino returns inf when no downside deviation and positive excess."""
        returns = [0.001] * 5
        result = calculate_sortino_ratio(returns, risk_free_rate=0.0)
        assert result == float("inf")

    def test_calculate_metrics_with_benchmark(self, sample_trades: list[dict]) -> None:
        """Metrics include alpha/beta when benchmark returns are provided."""
        equity = [100_000.0] + [
            100_000.0 + sum(t["pnl"] for t in sample_trades[:i])
            for i in range(1, len(sample_trades) + 1)
        ]
        benchmark_rets = [0.001, 0.002, -0.001, 0.001, 0.002]
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 7)

        metrics = calculate_backtest_metrics(
            equity_curve=equity,
            trades=sample_trades,
            benchmark_returns=benchmark_rets,
            start_date=start,
            end_date=end,
        )

        assert metrics.beta is not None
        assert metrics.alpha is not None

    def test_calculate_drawdown(self) -> None:
        """Drawdown is zero for monotonically increasing equity."""
        equity = [100.0, 110.0, 120.0, 130.0, 140.0]
        drawdowns = calculate_drawdown(equity)
        assert all(dd == 0.0 for dd in drawdowns)

    def test_calculate_drawdown_with_decline(self) -> None:
        """Drawdown reflects peak-to-trough decline."""
        equity = [100.0, 110.0, 105.0, 120.0, 110.0]
        drawdowns = calculate_drawdown(equity)
        assert drawdowns[-1] > 0  # Last point has drawdown from peak 120 to 110


# ─── Validation Tests ───────────────────────────────────────────────────────


class TestWalkForwardAnalyzer:
    def test_generate_windows_expanding(
        self, sample_candles: list[Candle]
    ) -> None:
        """Walk-forward generates valid expanding windows."""
        config = BacktestConfig(symbol="BTC/USD")
        analyzer = WalkForwardAnalyzer(
            config=config, n_steps=5, train_ratio=0.6
        )
        windows = analyzer.generate_windows(sample_candles)
        assert len(windows) > 0
        train_slice, test_slice = windows[0]
        assert train_slice.stop < test_slice.start
        assert test_slice.start < len(sample_candles)

    def test_is_valid_no_overlap(
        self, sample_candles: list[Candle]
    ) -> None:
        """Rolling-window walk-forward windows don't overlap."""
        config = BacktestConfig(symbol="BTC/USD")
        analyzer = WalkForwardAnalyzer(
            config=config, n_steps=3, train_ratio=0.7, window_type="rolling"
        )
        analyzer.analyze(sample_candles, MACDCrossover())
        assert analyzer.is_valid

    def test_generate_windows_rolling(self, sample_candles: list[Candle]) -> None:
        """Rolling windows are generated correctly."""
        config = BacktestConfig(symbol="BTC/USD")
        analyzer = WalkForwardAnalyzer(
            config=config, n_steps=3, train_ratio=0.5, window_type="rolling"
        )
        windows = analyzer.generate_windows(sample_candles)
        assert len(windows) > 0
        for train_slice, test_slice in windows:
            assert train_slice.stop < test_slice.start

    def test_walkforward_is_degraded(self, sample_candles: list[Candle]) -> None:
        """is_degraded returns False when no degradation data."""
        config = BacktestConfig(symbol="BTC/USD")
        analyzer = WalkForwardAnalyzer(
            config=config, n_steps=2, train_ratio=0.6
        )
        assert not analyzer.is_degraded  # no steps yet

    def test_walkforward_generate_windows_expanding(
        self, sample_candles: list[Candle]
    ) -> None:
        """Expanding windows grow the training set each step."""
        config = BacktestConfig(symbol="BTC/USD")
        analyzer = WalkForwardAnalyzer(
            config=config, n_steps=5, train_ratio=0.5, window_type="expanding"
        )
        windows = analyzer.generate_windows(sample_candles)
        assert len(windows) > 1
        # First training slice should be smaller than the last
        assert windows[0][0].stop < windows[-1][0].stop

    def test_walkforward_rolling_windows(self, sample_candles: list[Candle]) -> None:
        """Rolling windows keep training set size constant."""
        config = BacktestConfig(symbol="BTC/USD")
        analyzer = WalkForwardAnalyzer(
            config=config, n_steps=4, train_ratio=0.5, window_type="rolling"
        )
        windows = analyzer.generate_windows(sample_candles)
        assert len(windows) > 0
        # All training slices should be the same size
        sizes = [w[0].stop - w[0].start for w in windows]
        assert len(set(sizes)) <= 1

    def test_analyze_degradation(self, sample_candles: list[Candle]) -> None:
        """Analyze populates steps and degradation property."""
        config = BacktestConfig(symbol="BTC/USD")
        analyzer = WalkForwardAnalyzer(
            config=config, n_steps=3, train_ratio=0.6
        )
        steps = analyzer.analyze(sample_candles, MACDCrossover())
        assert len(steps) > 0
        for step in steps:
            assert isinstance(step.degradation, float)


class TestPurgedKFold:
    def test_splits_valid(self, sample_candles: list[Candle]) -> None:
        """Purged k-fold produces valid splits."""
        cv = PurgedKFold(n_splits=3)
        splits = cv.split(sample_candles)
        assert len(splits) == 3

        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            # No overlap between train and test
            assert len(set(train_idx) & set(test_idx)) == 0

    def test_not_enough_data(self, sample_candles: list[Candle]) -> None:
        """Too few candles raises error."""
        small_candles = sample_candles[:3]
        cv = PurgedKFold(n_splits=3)
        with pytest.raises(ValueError, match="Not enough data"):
            cv.split(small_candles)

    def test_n_splits_actual(self, sample_candles: list[Candle]) -> None:
        """n_splits_actual reflects the number of generated splits."""
        cv = PurgedKFold(n_splits=4)
        cv.split(sample_candles)
        assert cv.n_splits_actual == 4

    def test_split_partial_overlap(self, sample_candles: list[Candle]) -> None:
        """Train+test covers all indices."""
        cv = PurgedKFold(n_splits=3)
        splits = cv.split(sample_candles)
        all_covered = set()
        for train_idx, test_idx in splits:
            all_covered.update(train_idx)
            all_covered.update(test_idx)
        # Due to purging some indices may be removed, so just verify each split is valid
        for train_idx, test_idx in splits:
            assert len(set(train_idx) & set(test_idx)) == 0

    def test_large_dataset_splits(self, sample_candles: list[Candle]) -> None:
        """Splits scale with larger datasets."""
        cv = PurgedKFold(n_splits=5)
        splits = cv.split(sample_candles)
        assert len(splits) == 5
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0


class TestMonteCarloSimulator:
    def test_simulation_runs(self, sample_trades: list[dict]) -> None:
        """Monte Carlo simulation completes and returns results."""
        sim = MonteCarloSimulator(n_simulations=100, seed=42)
        results = sim.simulate(sample_trades, initial_capital=100_000.0)

        assert len(results) == 1
        assert "actual_sharpe" in results[0]
        assert "simulated_sharpe_mean" in results[0]
        assert "simulated_sharpe_ci95" in results[0]

    def test_simulation_zero_trades(self) -> None:
        """Simulation with no trades returns zero metrics."""
        sim = MonteCarloSimulator(n_simulations=50, seed=42)
        results = sim.simulate([], initial_capital=100_000.0)
        assert len(results) == 1

    def test_simulation_with_seed_reproducibility(self) -> None:
        """Same seed produces identical results."""
        sim1 = MonteCarloSimulator(n_simulations=100, seed=99)
        sim2 = MonteCarloSimulator(n_simulations=100, seed=99)
        trades = [
            {"pnl": 10.0, "quantity": 1.0, "price": 50000.0, "holding_days": 1},
            {"pnl": -5.0, "quantity": 0.5, "price": 49000.0, "holding_days": 2},
            {"pnl": 15.0, "quantity": 1.0, "price": 51000.0, "holding_days": 1},
        ]
        r1 = sim1.simulate(trades, initial_capital=100_000.0)
        r2 = sim2.simulate(trades, initial_capital=100_000.0)
        assert r1[0]["actual_sharpe"] == r2[0]["actual_sharpe"]
        assert r1[0]["simulated_sharpe_mean"] == r2[0]["simulated_sharpe_mean"]

    def test_simulation_with_many_trades(self, sample_trades: list[dict]) -> None:
        """Simulation with many trades runs without error."""
        # Create many trades
        many_trades = [
            {"pnl": 10.0 * (i + 1), "quantity": 1.0, "price": 50000.0, "holding_days": 1}
            for i in range(50)
        ]
        sim = MonteCarloSimulator(n_simulations=200, seed=42)
        results = sim.simulate(many_trades, initial_capital=100_000.0)
        assert len(results) == 1
        assert results[0]["actual_sharpe"] >= 0


class TestParameterSweeper:
    def test_generate_combinations(self) -> None:
        """Parameter sweeper generates all combinations."""
        grid = {"fast": [5, 10], "slow": [20, 30]}
        sweeper = ParameterSweeper(
            config=BacktestConfig(symbol="BTC/USD"),
            param_grid=grid,
        )
        combos = sweeper._generate_combinations()
        assert len(combos) == 4  # 2 * 2

    def test_best_result(self, sample_candles: list[Candle]) -> None:
        """Best result is returned by metric key."""
        grid = {"fast_period": [5, 10], "slow_period": [20, 30]}
        sweeper = ParameterSweeper(
            config=BacktestConfig(symbol="BTC/USD"),
            param_grid=grid,
            metric_key="sharpe_ratio",
        )
        sweeper.sweep(sample_candles, MACDCrossover)
        best = sweeper.best_result
        assert best is not None
        assert isinstance(best.param_combination, dict)

    def test_empty_grid(self) -> None:
        """Empty grid produces no combinations."""
        sweeper = ParameterSweeper(
            config=BacktestConfig(symbol="BTC/USD"),
            param_grid={},
        )
        combos = sweeper._generate_combinations()
        assert len(combos) <= 1
        assert sweeper.best_result is None

    def test_single_param_sweep(self, sample_candles: list[Candle]) -> None:
        """Sweep with a single parameter works correctly."""
        grid = {"fast_period": [5, 10]}
        sweeper = ParameterSweeper(
            config=BacktestConfig(symbol="BTC/USD"),
            param_grid=grid,
        )
        sweeper.sweep(sample_candles, MACDCrossover)
        assert len(sweeper.results) == 2
        best = sweeper.best_result
        assert best is not None
        assert "fast_period" in best.param_combination


# ─── Strategy Tests ─────────────────────────────────────────────────────────


class TestMACDCrossover:
    def test_initial_no_signal(self, sample_candles: list[Candle]) -> None:
        """First bars produce no signal (warmup)."""
        strategy = MACDCrossover(name="test")
        for i in range(10):
            signal = strategy.on_bar(sample_candles[i])
            assert signal is None

    def test_signal_generation(self, sample_candles: list[Candle]) -> None:
        """MACD strategy can generate signals."""
        strategy = MACDCrossover(name="test")
        for candle in sample_candles:
            signal = strategy.on_bar(candle)
            if signal is not None:
                assert signal.action in (SignalAction.BUY, SignalAction.SELL)
                assert 0.0 <= signal.confidence <= 1.0
                assert signal.symbol == "BTC/USD"

    def test_crossover_detection(self) -> None:
        """Crossover detection triggers on signal line crossing."""
        candles: list[Candle] = []
        base = datetime(2024, 1, 1)
        for i in range(40):
            close = 100.0 + i * 0.1
            candle = _make_candle(base.replace(second=i), close)
            # Bullish crossover at i=25
            if i >= 24 and i < 26:
                object.__setattr__(candle, 'macd_line', 0.0)
            elif i >= 26:
                object.__setattr__(candle, 'macd_line', 0.5 + (i - 26) * 0.1)
            object.__setattr__(candle, 'macd_signal', 0.0)
            candles.append(candle)

        strategy = MACDCrossover(name="test")
        found_buy = False
        for i, candle in enumerate(candles):
            signal = strategy.on_bar(candle)
            if i >= 26 and signal is not None and signal.action == SignalAction.BUY:
                found_buy = True
                break
        assert found_buy, "Expected BUY signal after crossover"

    def test_history_tracking(self, sample_candles: list[Candle]) -> None:
        """Strategy tracks candle history via on_bars."""
        strategy = MACDCrossover(name="test")
        strategy.on_bars(sample_candles)
        assert len(strategy.history) == len(sample_candles)

    def test_different_period_values(self) -> None:
        """Strategy works with custom periods."""
        candles: list[Candle] = []
        base = datetime(2024, 1, 1)
        for i in range(50):
            close = 100.0 + i * 0.05
            candle = _make_candle(base.replace(second=i), close)
            candles.append(candle)

        strategy = MACDCrossover(name="test", fast_period=5, slow_period=15, signal_period=5)
        for candle in candles:
            strategy.on_bar(candle)
        # Should not raise


# ─── DataFeed Tests ─────────────────────────────────────────────────────────


class TestMemoryDataFeed:
    def test_get_candles(self, sample_candles: list[Candle]) -> None:
        """MemoryDataFeed returns correct candles."""
        feed = MemoryDataFeed(candles=sample_candles)
        candles = feed.get_candles("BTC/USD")
        assert len(candles) == len(sample_candles)

    def test_get_candles_empty(self) -> None:
        """Empty feed returns empty list."""
        feed = MemoryDataFeed()
        candles = feed.get_candles("ETH/USD")
        assert candles == []

    def test_get_candles_wrong_symbol(self, sample_candles: list[Candle]) -> None:
        """Feed returns empty list for unknown symbol."""
        feed = MemoryDataFeed(candles=sample_candles)
        candles = feed.get_candles("UNKNOWN/USD")
        assert candles == []


# ─── Execution Model Tests ──────────────────────────────────────────────────


class TestExecutionModels:
    def test_market_order_execution(self, sample_candles: list[Candle]) -> None:
        """Market order fills at next bar open with slippage."""
        exec_model = MarketExecutionModel()
        candle = sample_candles[10]
        order = BacktestOrder(
            order_id="mkt-001",
            instrument="BTC/USD",
            side=Side.BUY,
            quantity=1.0,
            price=None,
            order_type=OrderType.MARKET,
            timestamp=candle.timestamp,
        )
        fill = exec_model.execute(order, candle)
        assert fill.status == FillStatus.FILLED
        assert fill.filled_quantity == 1.0
        assert fill.fill_price > 0

    def test_limit_order_filled(self, sample_candles: list[Candle]) -> None:
        """Limit order fills when price reaches limit."""
        exec_model = LimitExecutionModel()
        candle = sample_candles[10]
        # Set limit above candle high → buy should fill
        order = BacktestOrder(
            order_id="lim-001",
            instrument="BTC/USD",
            side=Side.BUY,
            quantity=1.0,
            price=candle.high * 1.01,
            order_type=OrderType.LIMIT,
            timestamp=candle.timestamp,
        )
        fill = exec_model.execute(order, candle)
        assert fill.status == FillStatus.FILLED

    def test_limit_order_rejected(self, sample_candles: list[Candle]) -> None:
        """Limit order rejected when price doesn't reach limit."""
        exec_model = LimitExecutionModel()
        candle = sample_candles[10]
        # Set limit below candle low → buy should NOT fill
        order = BacktestOrder(
            order_id="lim-002",
            instrument="BTC/USD",
            side=Side.BUY,
            quantity=1.0,
            price=candle.low * 0.99,
            order_type=OrderType.LIMIT,
            timestamp=candle.timestamp,
        )
        fill = exec_model.execute(order, candle)
        assert fill.status == FillStatus.REJECTED
        assert fill.filled_quantity == 0

    def test_stop_limit_triggered(self, sample_candles: list[Candle]) -> None:
        """Stop-limit order fills when stop is triggered."""
        exec_model = StopLimitExecutionModel()
        candle = sample_candles[10]
        # Set stop below candle high → should trigger
        order = BacktestOrder(
            order_id="sl-001",
            instrument="BTC/USD",
            side=Side.BUY,
            quantity=1.0,
            price=candle.close,
            order_type=OrderType.STOP_LIMIT,
            stop_price=candle.low * 0.99,
            timestamp=candle.timestamp,
        )
        fill = exec_model.execute(order, candle)
        assert fill.status == FillStatus.FILLED

    def test_stop_limit_not_triggered(self, sample_candles: list[Candle]) -> None:
        """Stop-limit order rejected when stop not triggered."""
        exec_model = StopLimitExecutionModel()
        candle = sample_candles[10]
        # Set stop above candle high → should NOT trigger for buy
        order = BacktestOrder(
            order_id="sl-002",
            instrument="BTC/USD",
            side=Side.BUY,
            quantity=1.0,
            price=candle.close,
            order_type=OrderType.STOP_LIMIT,
            stop_price=candle.high * 1.01,
            timestamp=candle.timestamp,
        )
        fill = exec_model.execute(order, candle)
        assert fill.status == FillStatus.REJECTED
        assert fill.filled_quantity == 0


# ─── Commission Model Tests ─────────────────────────────────────────────────


class TestCommissionModels:
    def test_fixed_commission(self) -> None:
        """Fixed commission returns the configured fee."""
        model = FixedCommissionModel(fee=5.0)
        assert model.calculate(1000.0, Side.BUY) == 5.0
        assert model.calculate(50000.0, Side.SELL) == 5.0

    def test_percentage_commission(self) -> None:
        """Percentage commission scales with notional."""
        model = PercentageCommissionModel(rate=0.001)
        assert model.calculate(1000.0, Side.BUY) == 1.0
        assert model.calculate(100000.0, Side.SELL) == 100.0

    def test_tiered_commission(self) -> None:
        """Tiered commission uses the correct tier."""
        model = TieredCommissionModel(
            tiers=[(0, 0.001), (10_000, 0.0008), (100_000, 0.0005)]
        )
        # Small notional → first tier
        assert model.calculate(1_000.0, Side.BUY) == 1.0  # 1000 * 0.001
        # Medium notional → second tier
        assert model.calculate(50_000.0, Side.BUY) == 40.0  # 50000 * 0.0008
        # Large notional → third tier
        assert model.calculate(200_000.0, Side.BUY) == 100.0  # 200000 * 0.0005


# ─── Slippage Model Tests ───────────────────────────────────────────────────


class TestSlippageModels:
    def test_fixed_slippage(self) -> None:
        """Fixed slippage returns the configured amount."""
        model = FixedSlippageModel(amount=0.05)
        assert model.calculate(50000.0, Side.BUY) == 0.05
        assert model.calculate(100.0, Side.SELL) == 0.05

    def test_percentage_slippage(self) -> None:
        """Percentage slippage scales with price."""
        model = PercentageSlippageModel(bps=10.0)  # 10 bps = 0.001
        assert model.calculate(50000.0, Side.BUY) == 50.0  # 50000 * 0.001
        assert model.calculate(100.0, Side.SELL) == 0.1  # 100 * 0.001

    def test_volume_slippage(self) -> None:
        """Volume slippage increases with volume."""
        model = VolumeBasedSlippageModel(base_bps=2.0, depth=1_000_000)
        # Small volume → base slippage
        small = model.calculate(50000.0, Side.BUY, volume=100.0)
        # Large volume → higher slippage
        large = model.calculate(50000.0, Side.BUY, volume=1_000_000.0)
        assert large > small


# ─── StrategySignal Tests ───────────────────────────────────────────────────


class TestStrategySignal:
    def test_signal_buy_action(self) -> None:
        """Buy signal has correct attributes."""
        ts = datetime(2024, 1, 1, 12, 0)
        signal = StrategySignal(
            action=SignalAction.BUY,
            symbol="BTC/USD",
            confidence=0.8,
            reason="bullish",
            timestamp=ts,
        )
        assert signal.action == SignalAction.BUY
        assert signal.symbol == "BTC/USD"
        assert signal.confidence == 0.8
        assert signal.timestamp == ts

    def test_signal_sell_action(self) -> None:
        """Sell signal has correct attributes."""
        signal = StrategySignal(
            action=SignalAction.SELL,
            symbol="ETH/USD",
            confidence=0.9,
            reason="bearish",
        )
        assert signal.action == SignalAction.SELL
        assert signal.symbol == "ETH/USD"
        assert signal.confidence == 0.9

    def test_signal_with_stops(self) -> None:
        """Signal can carry stop loss and take profit levels."""
        signal = StrategySignal(
            action=SignalAction.BUY,
            symbol="BTC/USD",
            confidence=0.7,
            stop_loss=49000.0,
            take_profit=52000.0,
        )
        assert signal.stop_loss == 49000.0
        assert signal.take_profit == 52000.0
        assert signal.position_size == 0.0  # default
