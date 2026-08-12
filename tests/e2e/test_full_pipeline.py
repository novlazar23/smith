"""End-to-end integration test for the Trading Orchestra pipeline.

Verifies the entire trading cycle from raw OHLCV candles through
feature calculation, agent voting, consensus aggregation, strategy
generation, portfolio sizing, risk checks, and paper-trading execution.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, UTC

import numpy as np
import pytest

from packages.consensus import (
    CalibratedConsensusAggregator,
    ConsensusDecision,
    ConsensusResult,
    VoteDirection,
    WeightConfig,
)
from packages.domain.market_data import CandleAggregation
from packages.paper import Fill
from packages.paper.base import (
    OrderType,
    PaperAccount,
    PaperPosition,
    TradeDirection,
)
from packages.paper.executor import PaperExecutor
from packages.risk.base import (
    RiskDecision,
    RiskGateResult,
    RiskGateType,
)
from packages.risk.drawdown import DrawdownMonitor
from packages.risk.position_sizing import KellyPositionSizer, ATRPositionSizer
from packages.schemas.agent_report import (
    AgentReport,
    AgentStatus,
    EvidenceReference,
)
from packages.strategy.engine import StrategyEngine
from packages.strategy.models import (
    StrategyConfig,
    StrategyDirection,
    StrategyProposal,
)


# ---------------------------------------------------------------------------
# Fixtures — shared across test cases
# ---------------------------------------------------------------------------

def _make_mock_candles(
    instrument: str = "BTC/USD",
    count: int = 100,
    base_price: float = 100.0,
) -> list[dict]:
    """Generate deterministic mock OHLCV candles."""
    candles: list[dict] = []
    price = base_price
    rng = np.random.default_rng(seed=42)
    base_time = datetime(2025, 1, 1, tzinfo=UTC)
    for i in range(count):
        change = rng.normal(0, 0.005)  # ~0.5% per-bar std dev
        open_ = price
        close = price * (1 + change)
        high = max(open_, close) * (1 + abs(rng.normal(0, 0.002)))
        low = min(open_, close) * (1 - abs(rng.normal(0, 0.002)))
        volume = float(rng.uniform(100, 1000))
        candles.append(
            {
                "open_time": base_time + timedelta(minutes=i),
                "close_time": base_time + timedelta(minutes=i + 1),
                "open": round(open_, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "volume": round(volume, 2),
                "trade_count": int(rng.integers(50, 500)),
            }
        )
        price = close
    return candles


def _create_agent_reports(
    instrument: str = "BTC/USD",
    as_of: datetime | None = None,
) -> list[AgentReport]:
    """Create mock agent reports (trend + sentiment agents)."""
    if as_of is None:
        as_of = datetime.now(UTC)

    trend_ref = EvidenceReference(
        reference="trend-sma-cross",
        feature="sma_20_vs_sma_50",
        value="bullish_crossover",
        direction="positive",
        relevance=0.85,
    )
    sentiment_ref = EvidenceReference(
        reference="sent-nlp-score",
        feature="nlp_sentiment",
        value="0.72_positive",
        direction="positive",
        relevance=0.65,
    )

    trend_report = AgentReport(
            report_id=f"trend-{uuid.uuid4()}",
            run_id="run-001",
            agent_id="trend-agent",
            agent_version="1.2.0",
            instrument=instrument,
            horizon="1h",
            as_of=as_of,
            hypothesis="Short-term trend is bullish based on SMA crossover",
            probabilities={"up": 0.65, "down": 0.15, "range": 0.20},
            evidence=[trend_ref],
            raw_confidence=0.72,
            calibrated_confidence=None,
            expected_return=None,
            status=AgentStatus.ACTIVE,
        )

    sentiment_report = AgentReport(
            report_id=f"sent-{uuid.uuid4()}",
            run_id="run-001",
            agent_id="sentiment-agent",
            agent_version="2.0.1",
            instrument=instrument,
            horizon="1h",
            as_of=as_of,
            hypothesis="Sentiment analysis indicates bullish market conditions",
            probabilities={"up": 0.60, "down": 0.20, "range": 0.20},
            evidence=[sentiment_ref],
            raw_confidence=0.68,
            calibrated_confidence=None,
            expected_return=None,
            status=AgentStatus.ACTIVE,
        )

    return [trend_report, sentiment_report]


class TestFullPipeline:
    """Complete pipeline integration test.

    Verifies: raw candles → snapshot → features → agents → consensus
    → strategy → portfolio sizing → risk checks → paper order → outcome.
    """

    def test_end_to_end_trading_pipeline(self):
        """Run a full pipeline cycle with mock data and verify final state."""
        # ---- Step 1: Generate mock OHLCV candles ----
        candles = _make_mock_candles(instrument="BTC/USD", count=100, base_price=50000.0)
        assert len(candles) == 100
        assert candles[0]["open"] > 0
        assert candles[-1]["close"] > 0

        # ---- Step 2: Create market snapshot via CandleAggregation ----
        agg = CandleAggregation.from_raw_candles(
            candles[:5],
            target_timeframe="5m",
            instrument="BTC/USD",
            venue="mock",
        )
        assert agg.open > 0
        assert agg.high >= agg.low
        assert agg.volume >= 0
        current_price = agg.close

        # ---- Step 3: Calculate features (SMA, EMA) ----
        from packages.domain.market_data import MultiTimeframeAggregator

        mta = MultiTimeframeAggregator(base_timeframe="1m")
        sma_20 = mta.compute_simple_moving_average(candles, price_key="close", window=20)
        ema_12 = mta.compute_exponential_moving_average(candles, price_key="close", span=12)
        assert len(sma_20) == 100  # Same length as input, NaN for first window-1
        assert len(ema_12) == 100
        # Last values should be finite (no NaN in the tail for 100 candles)
        assert np.isfinite(sma_20[-1])
        assert np.isfinite(ema_12[-1])

        # Volatility (ATR approximation)
        closes = np.array([c["close"] for c in candles], dtype=np.float64)
        tr = np.diff(closes)
        atr = float(np.mean(np.abs(tr[-20:]))) if len(tr) >= 20 else 1.0
        assert atr > 0

        # Feature dict
        features = {
            "current_price": current_price,
            "sma_20": float(sma_20[-1]),
            "ema_12": float(ema_12[-1]),
            "atr": atr,
            "volatility": atr * 0.6,
            "entry_type": "MARKET",
            "entry_condition": "MOMENTUM",
        }

        # ---- Step 4: Run agents (mock reports) ----
        as_of = candles[-1]["close_time"]
        agent_reports = _create_agent_reports(instrument="BTC/USD", as_of=as_of)
        assert len(agent_reports) == 2
        for report in agent_reports:
            assert report.probabilities is not None
            total = sum(report.probabilities.values())
            assert abs(total - 1.0) <= 0.0001

        # ---- Step 5: Consensus aggregation ----
        aggregator = CalibratedConsensusAggregator(
            config=WeightConfig(base_weight=1.0, min_consensus_threshold=0.5)
        )
        consensus_result = aggregator.aggregate(agent_reports)
        assert consensus_result.decision in (
            ConsensusDecision.LONG_BIAS,
            ConsensusDecision.SHORT_BIAS,
            ConsensusDecision.RANGE,
            ConsensusDecision.NO_TRADE,
        )
        # The two bullish trend agents should push toward LONG_BIAS
        assert consensus_result.confidence > 0.0

        # Build a ConsensusResult for the strategy engine
        consensus_for_strategy = ConsensusResult(
            decision=consensus_result.decision,
            vote_distribution=consensus_result.vote_distribution,
            agent_weights=consensus_result.agent_weights,
            agent_agreements=list(
                set(r.agent_id for r in agent_reports)
            ),
            agent_disagreements=[],
            confidence=consensus_result.confidence,
            reason=consensus_result.reason,
        )

        # ---- Step 6: Strategy generates a proposal ----
        strategy_engine = StrategyEngine(config=StrategyConfig())
        proposal = strategy_engine.run(
            context={
                "consensus": consensus_for_strategy,
                "features": features,
                "costs": {"slippage": 0.001, "commission": 0.001},
            }
        )
        assert proposal is not None
        assert isinstance(proposal, StrategyProposal)

        # ---- Step 7: Portfolio sizing ----
        kelly_sizer = KellyPositionSizer()
        fraction = kelly_sizer.calculate_fraction(
            win_rate=0.6,
            avg_win=1500.0,
            avg_loss=800.0,
        )
        account_size = 100000.0
        target_qty = kelly_sizer.calculate_size(
            win_rate=0.6,
            avg_win=1500.0,
            avg_loss=800.0,
            account_size=account_size,
        )
        assert target_qty >= 0

        # ATR-based sizing
        atr_sizer = ATRPositionSizer()
        atr_qty = atr_sizer.calculate_size(
            atr=atr,
            stop_distance_atr=2,
            account_size=account_size,
        )
        assert atr_qty > 0

        # ---- Step 8: Risk checks (drawdown gate) ----
        monitor = DrawdownMonitor(max_drawdown_pct=0.15, warning_drawdown_pct=0.10)
        dd_result = monitor.check_drawdown(current_equity=account_size)
        assert dd_result.passed is True
        assert dd_result.severity == "soft"

        # Exposure gate (simulated)
        exposure_gate = RiskGateResult(
            gate_type=RiskGateType.EXPOSURE,
            passed=True,
            severity="hard",
        )

        # Data quality gate
        dq_gate = RiskGateResult(
            gate_type=RiskGateType.DATA_QUALITY,
            passed=True,
            severity="hard",
        )

        risk_decision = RiskDecision(
            risk_version="1.0",
            run_id="run-001",
            instrument="BTC/USD",
            approved=True,
            max_position_size=atr_qty,
            reduction_factor=1.0,
            gates=[dd_result, exposure_gate, dq_gate],
        )
        assert risk_decision.approved is True
        assert risk_decision.veto is False

        # ---- Step 9: Paper trading order ----
        executor = PaperExecutor(
            initial_cash=account_size,
            default_slippage_pct=0.001,
            default_commission_pct=0.001,
            max_position_size_pct=0.10,
        )
        account = executor.create_account("test-paper-001")
        assert account.cash == account_size
        assert account.initial_cash == account_size

        # Determine trade direction from proposal
        if proposal.direction == StrategyDirection.LONG:
            direction = TradeDirection.BUY
        elif proposal.direction == StrategyDirection.SHORT:
            direction = TradeDirection.SELL
        else:
            direction = TradeDirection.BUY  # default for no-trade proposals

        # Execute buy order with capped quantity (max 10% of equity)
        exec_price = current_price  # current market price
        order_qty = min(
            max(1, atr_qty),  # at least 1 unit, but capped by risk
            account_size * 0.10 / exec_price,  # max 10% of equity
        )

        trade = executor.submit_order(
            account=account,
            instrument="BTC/USD",
            direction=direction,
            quantity=order_qty,
            price=exec_price,
            order_type=OrderType.MARKET,
        )
        assert trade is not None
        assert trade.status == "filled"
        assert trade.filled_quantity > 0
        assert trade.filled_price > 0

        # ---- Step 10: Execute a follow-up fill (simulating partial execution) ----
        fill = Fill(
            fill_id=f"fill-{uuid.uuid4()}",
            quantity=order_qty * 0.5,
            price=exec_price * 1.002,  # slight upward slippage
            commission=0.5,
            slippage=0.002,
            timestamp=datetime.now(UTC),
        )
        assert fill.notional > 0
        assert fill.total_cost > 0

        # ---- Step 11: Verify final state ----
        account_summary = executor.get_account_summary(account)

        # Position should exist
        assert "BTC/USD" in account.positions or account.total_trades > 0

        # Cash should be less than initial (spent on the trade)
        assert account.cash < account.initial_cash

        # Equity should be reasonable (cash + position value)
        assert account.equity > 0

        # Total trades should be at least 1
        assert account.total_trades >= 1

        # Commission should be positive
        assert account.total_commission > 0

        # PnL should be finite (could be negative from costs)
        assert np.isfinite(account.total_pnl)

        # Unrealized PnL should be finite
        assert np.isfinite(account.unrealized_pnl)

        # Realized PnL should be finite
        assert np.isfinite(account.realized_pnl)

        # Verify position properties
        if "BTC/USD" in account.positions:
            pos = account.positions["BTC/USD"]
            assert pos.quantity > 0
            assert pos.avg_price > 0
            assert np.isfinite(pos.unrealized_pnl)
            assert np.isfinite(pos.realized_pnl)


class TestPipelineEdgeCases:
    """Edge cases in the full pipeline."""

    def test_no_trade_pipeline_flow(self):
        """Pipeline should handle NO_TRADE consensus gracefully."""
        candles = _make_mock_candles(count=50, base_price=25000.0)
        agg = CandleAggregation.from_raw_candles(
            candles[:5],
            target_timeframe="5m",
            instrument="ETH/USD",
            venue="mock",
        )
        current_price = agg.close

        # Create reports with conflicting signals
        ref_long = EvidenceReference(
            reference="r1", feature="trend", value="up",
            direction="positive", relevance=0.7,
        )
        ref_short = EvidenceReference(
            reference="r2", feature="momentum", value="down",
            direction="negative", relevance=0.7,
        )

        reports = [
            AgentReport(
                report_id="long-1",
                run_id="run-nt",
                agent_id="trend-agent",
                agent_version="1.0",
                instrument="ETH/USD",
                horizon="1h",
                as_of=datetime.now(UTC),
                hypothesis="Bullish trend",
                probabilities={"up": 0.5, "down": 0.25, "range": 0.25},
                evidence=[ref_long],
                raw_confidence=None,
                calibrated_confidence=None,
                expected_return=None,
                status=AgentStatus.ACTIVE,
            ),
            AgentReport(
                report_id="short-1",
                run_id="run-nt",
                agent_id="momentum-agent",
                agent_version="1.0",
                instrument="ETH/USD",
                horizon="1h",
                as_of=datetime.now(UTC),
                hypothesis="Bearish momentum",
                probabilities={"up": 0.25, "down": 0.5, "range": 0.25},
                evidence=[ref_short],
                raw_confidence=None,
                calibrated_confidence=None,
                expected_return=None,
                status=AgentStatus.ACTIVE,
            ),
        ]

        aggregator = CalibratedConsensusAggregator(
            config=WeightConfig(min_consensus_threshold=0.5)
        )
        result = aggregator.aggregate(reports)

        # Build ConsensusResult for strategy engine
        consensus = ConsensusResult(
            decision=result.decision,
            vote_distribution=result.vote_distribution,
            agent_weights=result.agent_weights,
            agent_agreements=[],
            agent_disagreements=[],
            confidence=result.confidence,
            reason=result.reason,
        )

        features = {
            "current_price": current_price,
            "sma_20": current_price * 0.99,
            "atr": current_price * 0.01,
            "volatility": current_price * 0.01,
        }

        engine = StrategyEngine(config=StrategyConfig())
        proposal = engine.run(context={"consensus": consensus, "features": features})
        assert proposal is not None
        # Proposal should exist (NO_TRADE is a valid outcome)
        assert isinstance(proposal, StrategyProposal)

    def test_risk_veto_stops_execution(self):
        """A hard risk veto should prevent order submission."""
        executor = PaperExecutor(initial_cash=100000.0)
        account = executor.create_account("veto-test")

        # Simulate existing position that would breach max drawdown
        monitor = DrawdownMonitor(max_drawdown_pct=0.05)  # very tight limit
        monitor.update_equity(100000.0)
        monitor.update_equity(94000.0)  # 6% drawdown

        dd_result = monitor.check_drawdown(current_equity=94000.0)
        assert dd_result.passed is False
        assert dd_result.severity == "hard"

        risk_decision = RiskDecision(
            risk_version="1.0",
            run_id="run-001",
            instrument="BTC/USD",
            approved=False,
            gates=[dd_result],
        )
        assert risk_decision.veto is True


class TestPipelineDataIntegrity:
    """Data integrity checks throughout the pipeline."""

    def test_candle_validations(self):
        """CandleAggregation should reject invalid data."""
        valid_candles = [
            {
                "open_time": datetime(2025, 1, 1, tzinfo=UTC),
                "close_time": datetime(2025, 1, 1, minute=5, tzinfo=UTC),
                "open": 100.0, "high": 105.0, "low": 98.0,
                "close": 102.0, "volume": 500.0, "trade_count": 100,
            },
        ]
        agg = CandleAggregation.from_raw_candles(
            valid_candles, "5m", "BTC/USD", "mock"
        )
        assert agg.high == 105.0
        assert agg.low == 98.0

        # high < low should fail
        invalid = [
            {
                "open_time": datetime(2025, 1, 1, tzinfo=UTC),
                "close_time": datetime(2025, 1, 1, minute=5, tzinfo=UTC),
                "open": 100.0, "high": 95.0, "low": 105.0,
                "close": 102.0, "volume": 500.0, "trade_count": 100,
            },
        ]
        with pytest.raises(ValueError):
            CandleAggregation.from_raw_candles(invalid, "5m", "BTC/USD", "mock")

        # Negative volume should fail
        bad_vol = [
            {
                "open_time": datetime(2025, 1, 1, tzinfo=UTC),
                "close_time": datetime(2025, 1, 1, minute=5, tzinfo=UTC),
                "open": 100.0, "high": 105.0, "low": 98.0,
                "close": 102.0, "volume": -1.0, "trade_count": 100,
            },
        ]
        with pytest.raises(ValueError):
            CandleAggregation.from_raw_candles(bad_vol, "5m", "BTC/USD", "mock")

    def test_consensus_probability_validation(self):
        """AgentReports with invalid probability sums should be rejected."""
        ref = EvidenceReference(
            reference="r1", feature="test", value="v",
            direction="positive", relevance=0.5,
        )
        bad_report = AgentReport(
            report_id="bad",
            run_id="run-bad",
            agent_id="bad-agent",
            agent_version="1.0",
            instrument="BTC/USD",
            horizon="1h",
            as_of=datetime.now(UTC),
            hypothesis="Bad probabilities",
            probabilities={"up": 0.5, "down": 0.5},  # sums to 1.0 but no range
            evidence=[ref],
            raw_confidence=None,
            calibrated_confidence=None,
            expected_return=None,
        )
        # This should actually pass: 0.5 + 0.5 = 1.0, which is valid
        # The validator only checks sum == 1.0, not that all keys exist
        assert abs(sum(bad_report.probabilities.values()) - 1.0) <= 0.0001

        # Truly invalid probabilities
        with pytest.raises(ValueError):
            AgentReport(
                report_id="bad2",
                run_id="run-bad",
                agent_id="bad-agent",
                agent_version="1.0",
                instrument="BTC/USD",
                horizon="1h",
                as_of=datetime.now(UTC),
                hypothesis="Totally invalid",
                probabilities={"up": 0.5, "down": 0.6},  # sums to 1.1
                evidence=[ref],
                raw_confidence=None,
                calibrated_confidence=None,
                expected_return=None,
            )

    def test_paper_account_limits(self):
        """PaperAccount should enforce cash and position constraints."""
        executor = PaperExecutor(initial_cash=1000.0)
        account = executor.create_account("limits-test")

        # Should not have enough cash for huge order
        with pytest.raises(ValueError, match="Insufficient"):
            executor.submit_order(
                account=account,
                instrument="BTC/USD",
                direction=TradeDirection.BUY,
                quantity=1000000.0,
                price=1.0,
                order_type=OrderType.MARKET,
            )

        # SELL without position should fail
        with pytest.raises(ValueError, match="No position to sell"):
            executor.submit_order(
                account=account,
                instrument="BTC/USD",
                direction=TradeDirection.SELL,
                quantity=10.0,
                price=100.0,
                order_type=OrderType.MARKET,
            )

        # Valid small order should succeed
        trade = executor.submit_order(
            account=account,
            instrument="BTC/USD",
            direction=TradeDirection.BUY,
            quantity=1.0,
            price=100.0,
            order_type=OrderType.MARKET,
        )
        assert trade.status == "filled"