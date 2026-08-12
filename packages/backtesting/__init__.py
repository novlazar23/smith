"""Backtesting Engine — Historische Simulations-Plattform.

Simuliert Orderausführung mit konfigurierbaren Gebühren, Slippage und
Look-ahead-Protection auf historischen OHLCV-Daten.

Nutzt PaperExecutor aus packages/paper für realistische Ausführung.
"""

from __future__ import annotations

from .core import (
    BacktestConfig,
    BacktestResult,
    Candle,
    OrderBookSimulator,
    PortfolioSnapshot,
    Position,
    Trade,
)
from .datafeed import (
    CsvDataFeed,
    DataFeed,
    DataFeedFactory,
    compute_indicators,
)
from .engine import BacktestEngine
from .execution import (
    BacktestOrder,
    CommissionModel,
    ExecutionModel,
    FixedCommissionModel,
    FixedSlippageModel,
    LimitExecutionModel,
    MarketExecutionModel,
    OrderType,
    PercentageCommissionModel,
    PercentageSlippageModel,
    Side,
    SimulatedFill,
    StopLimitExecutionModel,
    TieredCommissionModel,
    VolumeBasedSlippageModel,
)
from .metrics import (
    BacktestMetrics,
    calculate_backtest_metrics,
    calculate_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)
from .optimization import BayesianOptimizer, OptimizationResult
from .report import BacktestReport
from .strategies import (
    BaseStrategy,
    HeuristicAgent,
    MACDCrossover,
    RandomAgent,
    SignalAction,
    SignalDrivenStrategy,
    SimulatedAgent,
    StrategySignal,
)
from .validation import (
    MonteCarloSimulator,
    ParameterSweeper,
    ParameterSweepResult,
    PurgedKFold,
    WalkForwardAnalyzer,
    WalkForwardStep,
)

__all__ = [
    # Core
    "BacktestConfig",
    # Engine
    "BacktestEngine",
    # Metrics
    "BacktestMetrics",
    # Execution
    "BacktestOrder",
    # Report
    "BacktestReport",
    "BacktestResult",
    # Strategies
    "BaseStrategy",
    "Candle",
    "CommissionModel",
    # Data
    "CsvDataFeed",
    "DataFeed",
    "DataFeedFactory",
    "ExecutionModel",
    "FixedCommissionModel",
    "FixedSlippageModel",
    "HeuristicAgent",
    "LimitExecutionModel",
    "MACDCrossover",
    "MarketExecutionModel",
    # Validation
    "MonteCarloSimulator",
    "OrderBookSimulator",
    "OrderType",
    "ParameterSweepResult",
    "ParameterSweeper",
    "PercentageCommissionModel",
    "PercentageSlippageModel",
    "PortfolioSnapshot",
    "Position",
    "PurgedKFold",
    "RandomAgent",
    "Side",
    "SignalAction",
    "SignalDrivenStrategy",
    "SimulatedAgent",
    "SimulatedFill",
    "StopLimitExecutionModel",
    "StrategySignal",
    "TieredCommissionModel",
    "Trade",
    "VolumeBasedSlippageModel",
    "WalkForwardAnalyzer",
    "WalkForwardStep",
    # Optimization
    "BayesianOptimizer",
    "OptimizationResult",
    "calculate_backtest_metrics",
    "calculate_drawdown",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "compute_indicators",
]
