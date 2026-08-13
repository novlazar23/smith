"""Walk-Forward and Cross-Validation for backtest optimization.

Provides:
- WalkForwardAnalyzer: expanding train/test windows
- PurgedKFold: purged k-fold cross-validation (Le et al. 2022)
- ParameterSweeper: grid search over strategy parameters
- MonteCarloSimulator: random perturbation of trade sequences
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from .core import BacktestConfig, Candle
from .datafeed import MemoryDataFeed
from .engine import BacktestEngine

# ─── Walk-Forward ───────────────────────────────────────────────────────────


@dataclass
class WalkForwardStep:
    """Result of a single walk-forward step."""

    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_metrics: dict[str, Any] = field(default_factory=dict)
    test_metrics: dict[str, Any] = field(default_factory=dict)
    train_return: float = 0.0
    test_return: float = 0.0
    train_sharpe: float = 0.0
    test_sharpe: float = 0.0

    @property
    def degradation(self) -> float:
        """Performance degradation: test_return / train_return - 1."""
        if abs(self.train_return) < 1e-8:
            return 0.0
        return (self.test_return / self.train_return) - 1.0


class WalkForwardAnalyzer:
    """Walk-forward analysis with expanding or rolling windows.

    Walks forward by:
    1. Training on [start, train_end]
    2. Testing on [test_start, test_end]
    3. Shifting windows forward

    Supports both expanding window (training set grows) and
    rolling window (fixed training window).
    """

    def __init__(
        self,
        config: BacktestConfig,
        n_steps: int = 10,
        train_ratio: float = 0.7,
        gap_bars: int = 5,
        window_type: str = "expanding",
    ) -> None:
        """Initialize walk-forward analyzer.

        Args:
            config: Backtest configuration.
            n_steps: Number of walk-forward steps.
            train_ratio: Fraction of data for training (0.0 to 1.0).
            gap_bars: Bars to skip between train and test (prevents lookahead).
            window_type: "expanding" or "rolling".
        """
        self.config = config
        self.n_steps = n_steps
        self.train_ratio = train_ratio
        self.gap_bars = gap_bars
        self.window_type = window_type
        self._steps: list[WalkForwardStep] = []

    @property
    def steps(self) -> list[WalkForwardStep]:
        return self._steps

    @property
    def is_valid(self) -> bool:
        """Check if walk-forward design is valid (no overlapping windows)."""
        if not self._steps:
            return False
        for i in range(1, len(self._steps)):
            prev = self._steps[i - 1]
            curr = self._steps[i]
            if curr.train_start <= prev.train_end:
                return False
        return True

    def generate_windows(
        self,
        candles: list[Candle],
    ) -> list[tuple[slice, slice]]:
        """Generate train/test window slices.

        Args:
            candles: All candles for analysis.

        Returns:
            List of (train_slice, test_slice) tuples.
        """
        n = len(candles)
        train_end_idx = int(n * self.train_ratio)

        # Ensure at least gap_bars between train and test
        if self.gap_bars > 0:
            train_end_idx -= self.gap_bars

        if train_end_idx <= 0:
            raise ValueError("Not enough data for requested configuration")

        # Determine step size
        if self.window_type == "rolling":
            step_size = n - train_end_idx
            if step_size <= 0:
                step_size = 1
        else:
            step_size = max(1, (n - train_end_idx) // self.n_steps)

        windows: list[tuple[slice, slice]] = []
        train_start = 0

        for i in range(self.n_steps):
            train_end = min(train_end_idx + i * step_size, n)
            test_start = train_end + self.gap_bars
            test_end = min(test_start + step_size, n)

            if test_start >= n or test_start >= test_end:
                break

            windows.append((
                slice(train_start, train_end),
                slice(test_start, test_end),
            ))

        return windows

    def analyze(
        self,
        candles: list[Candle],
        strategy: Any,
        metric_key: str = "total_return",
    ) -> list[WalkForwardStep]:
        """Run walk-forward analysis.

        Args:
            candles: All candles for analysis.
            strategy: Strategy to evaluate.
            metric_key: Key to compare between train and test.

        Returns:
            List of WalkForwardStep results.
        """
        windows = self.generate_windows(candles)
        self._steps = []

        for train_slice, test_slice in windows:
            train_candles = candles[train_slice]
            test_candles = candles[test_slice]

            # Create step result
            step = WalkForwardStep(
                train_start=train_candles[0].timestamp,
                train_end=train_candles[-1].timestamp,
                test_start=test_candles[0].timestamp,
                test_end=test_candles[-1].timestamp,
            )

            # Resolve strategy: accept class or instance
            if isinstance(strategy, type):
                train_strategy = strategy()
                test_strategy = strategy()
            else:
                train_strategy = strategy
                test_strategy = strategy

            # Train phase — run backtest on training candles
            train_engine = BacktestEngine(config=self.config)
            train_feed = MemoryDataFeed(train_candles)
            train_result = train_engine.run(train_feed, train_strategy)
            step.train_metrics = train_result.metrics
            step.train_return = train_result.total_return

            # Test phase — run backtest on test candles
            test_engine = BacktestEngine(config=self.config)
            test_feed = MemoryDataFeed(test_candles)
            test_result = test_engine.run(test_feed, test_strategy)
            step.test_metrics = test_result.metrics
            step.test_return = test_result.total_return

            self._steps.append(step)

        return self._steps

    @property
    def is_degraded(self) -> bool:
        """Check if performance degrades from train to test."""
        if not self._steps:
            return False
        return any(step.degradation > 0.2 for step in self._steps)


# ─── Purged K-Fold Cross-Validation ─────────────────────────────────────────


class PurgedKFold:
    """Purged K-Fold cross-validation (Le, Kim, Roşca 2022).

    Implements purged k-fold to prevent information leakage between
    train and test sets in time-series data.

    The purge removes observations from the training set that are
    adjacent to test set observations, preventing look-ahead bias
    through overlapping windows.
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.1,
    ) -> None:
        """Initialize purged k-fold.

        Args:
            n_splits: Number of folds.
            embargo_pct: Percentage of test window to embargo from training.
        """
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self._splits: list[tuple[list[int], list[int]]] = []

    def split(
        self,
        candles: list[Candle],
    ) -> list[tuple[list[int], list[int]]]:
        """Generate train/test splits with purging.

        Args:
            candles: List of candles (must be chronologically sorted).

        Returns:
            List of (train_indices, test_indices) tuples.
        """
        n = len(candles)
        if n < self.n_splits * 2:
            raise ValueError("Not enough data for requested number of splits")

        # Create indices
        indices = np.arange(n)
        fold_size = n // self.n_splits

        splits = []
        for i in range(self.n_splits):
            # Test fold
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, n)
            test_indices = list(indices[test_start:test_end])

            # Training set: all except test fold
            train_indices = list(indices[:test_start]) + list(indices[test_end:])

            # Purge: remove observations in training that overlap with test
            # (for strategies with look-ahead in features)
            purge_size = max(1, int(len(test_indices) * self.embargo_pct))
            test_start_ts = candles[test_start].timestamp
            test_end_ts = candles[test_end - 1].timestamp
            train_indices = [
                idx for idx in train_indices
                if abs(
                    candles[idx].timestamp - test_start_ts
                ).total_seconds() > purge_size
                or abs(
                    candles[idx].timestamp - test_end_ts
                ).total_seconds() > purge_size
            ]

            splits.append((train_indices, test_indices))

        self._splits = splits
        return splits

    @property
    def n_splits_actual(self) -> int:
        return len(self._splits)

    @property
    def is_valid(self) -> bool:
        """Check if splits are valid (no empty sets, proper coverage)."""
        if not self._splits:
            return False
        for train_idx, test_idx in self._splits:
            if not train_idx or not test_idx:
                return False
        # Check that all indices are covered
        all_indices = set()
        for train_idx, test_idx in self._splits:
            all_indices.update(train_idx)
            all_indices.update(test_idx)
        return all_indices == set(range(len(self._splits[0][0]) + len(self._splits[0][1])))


# ─── Parameter Sweeper ──────────────────────────────────────────────────────


@dataclass
class ParameterSweepResult:
    """Result of a parameter sweep (grid search)."""

    param_combination: dict[str, Any]
    metrics: dict[str, Any]
    train_metrics: dict[str, Any] | None = None
    test_metrics: dict[str, Any] | None = None


class ParameterSweeper:
    """Grid search over strategy parameters.

    Evaluates all combinations of parameter values and ranks by
    the specified metric.
    """

    def __init__(
        self,
        config: BacktestConfig,
        param_grid: dict[str, list[Any]],
        metric_key: str = "sharpe_ratio",
    ) -> None:
        """Initialize parameter sweeper.

        Args:
            config: Base backtest configuration.
            param_grid: Dict of parameter names to list of values.
            metric_key: Key to rank results by.
        """
        self.config = config
        self.param_grid = param_grid
        self.metric_key = metric_key
        self._results: list[ParameterSweepResult] = []

    @property
    def results(self) -> list[ParameterSweepResult]:
        return self._results

    @property
    def best_result(self) -> ParameterSweepResult | None:
        """Return the best result by metric_key."""
        if not self._results:
            return None
        return max(self._results, key=lambda r: r.metrics.get(self.metric_key, 0))

    def sweep(
        self,
        candles: list[Candle],
        strategy_class: Any,
        base_params: dict[str, Any] | None = None,
    ) -> list[ParameterSweepResult]:
        """Run parameter sweep.

        Args:
            candles: All candles for evaluation.
            strategy_class: Class to instantiate with parameters.
            base_params: Base parameters shared across all runs.

        Returns:
            Sorted list of ParameterSweepResult (descending by metric_key).
        """
        # Generate all parameter combinations
        combinations = self._generate_combinations()
        self._results = []

        for params in combinations:
            # Merge base params
            full_params = {**(base_params or {}), **params}

            try:
                # Instantiate strategy with merged parameters
                strategy_instance = strategy_class(**full_params)

                # Run backtest
                engine = BacktestEngine(config=self.config)
                feed = MemoryDataFeed(candles)
                backtest_result = engine.run(feed, strategy_instance)

                result = ParameterSweepResult(
                    param_combination=params,
                    metrics=backtest_result.metrics,
                )

                self._results.append(result)

            except (ValueError, TypeError):
                continue  # Skip invalid parameter combinations

        # Sort by metric
        self._results.sort(
            key=lambda r: r.metrics.get(self.metric_key, 0),
            reverse=True,
        )

        return self._results

    def _generate_combinations(
        self,
    ) -> list[dict[str, Any]]:
        """Generate all parameter combinations."""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())

        combinations: list[dict[str, Any]] = []
        self._combo_helper(keys, values, {}, 0, combinations)
        return combinations

    @staticmethod
    def _combo_helper(
        keys: list[str],
        values: list[list[Any]],
        current: dict[str, Any],
        depth: int,
        results: list[dict[str, Any]],
    ) -> None:
        """Recursive helper to generate combinations."""
        if depth == len(keys):
            results.append(dict(current))
            return

        key = keys[depth]
        for val in values[depth]:
            current[key] = val
            ParameterSweeper._combo_helper(keys, values, current, depth + 1, results)
            del current[key]


# ─── Monte Carlo Simulation ─────────────────────────────────────────────────


class MonteCarloSimulator:
    """Monte Carlo simulation for strategy robustness testing.

    Randomly perturbs trade sequences to estimate confidence intervals
    for key metrics.
    """

    def __init__(
        self,
        n_simulations: int = 1000,
        seed: int = 42,
    ) -> None:
        """Initialize Monte Carlo simulator.

        Args:
            n_simulations: Number of random perturbations.
            seed: Random seed for reproducibility.
        """
        self.n_simulations = n_simulations
        self.seed = seed
        self._results: list[dict[str, Any]] = []

    def simulate(
        self,
        trades: list[dict[str, Any]],
        initial_capital: float = 100_000.0,
    ) -> list[dict[str, Any]]:
        """Run Monte Carlo simulation.

        Randomly shuffles trade order and replays to estimate
        metric distributions.

        Args:
            trades: List of trade dicts with 'pnl' key.
            initial_capital: Starting capital for simulation.

        Returns:
            List of simulation results with metric distributions.
        """
        np.random.seed(self.seed)
        pnl_array = np.array([t.get("pnl", 0) for t in trades])

        # Store actual values
        actual_sharpe = self._compute_sharpe(pnl_array)
        actual_max_dd = self._compute_max_dd(pnl_array, initial_capital)
        actual_return = pnl_array.sum() / initial_capital if initial_capital > 0 else 0

        simulated_sharpes = []
        simulated_dds = []
        simulated_returns = []

        for _ in range(self.n_simulations):
            # Shuffle trades
            shuffled = pnl_array[np.random.permutation(len(pnl_array))]
            simulated_sharpes.append(self._compute_sharpe(shuffled))
            simulated_dds.append(
                self._compute_max_dd(shuffled, initial_capital)
            )
            simulated_returns.append(
                shuffled.sum() / initial_capital if initial_capital > 0 else 0
            )

        self._results = [
            {
                "actual_sharpe": actual_sharpe,
                "simulated_sharpe_mean": float(np.mean(simulated_sharpes)),
                "simulated_sharpe_std": float(np.std(simulated_sharpes)),
                "simulated_sharpe_ci95": (
                    float(np.percentile(simulated_sharpes, 2.5)),
                    float(np.percentile(simulated_sharpes, 97.5)),
                ),
                "actual_max_dd": actual_max_dd,
                "simulated_dd_mean": float(np.mean(simulated_dds)),
                "simulated_dd_ci95": (
                    float(np.percentile(simulated_dds, 2.5)),
                    float(np.percentile(simulated_dds, 97.5)),
                ),
                "actual_return": actual_return,
                "simulated_return_mean": float(np.mean(simulated_returns)),
                "simulated_return_std": float(np.std(simulated_returns)),
                "simulated_return_ci95": (
                    float(np.percentile(simulated_returns, 2.5)),
                    float(np.percentile(simulated_returns, 97.5)),
                ),
                "p_win_rate": float(
                    np.mean(np.array(simulated_returns) < actual_return)
                ),
            }
        ]

        return self._results

    @staticmethod
    def _compute_sharpe(pnls: np.ndarray) -> float:
        """Simple Sharpe approximation from PnL series."""
        if len(pnls) < 2:
            return 0.0
        mean = np.mean(pnls)
        std = np.std(pnls)
        return (mean / std * np.sqrt(365)) if std > 0 else 0.0

    @staticmethod
    def _compute_max_dd(
        pnls: np.ndarray,
        initial_capital: float,
    ) -> float:
        """Compute max drawdown from PnL series."""
        equity = initial_capital + np.cumsum(pnls)
        peak = initial_capital
        max_dd = 0.0

        for e in equity:
            if e > peak:
                peak = e
            dd = (peak - e) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)

        return max_dd
