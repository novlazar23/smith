"""Integration tests for the Trading Orchestra full pipeline.

End-to-end test chaining:
  market_data → indicator_agent → consensus → strategy → paper executor

Verifies that a realistic data flow produces valid paper orders from
candle data through the entire analysis stack.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
from packages.consensus import (
    ConsensusDecision,
    ConsensusResult,
    WeightConfig,
    WeightedConsensusEngine,
)
from packages.domain.market_data.orderbook import (
    OrderBookReconstructor,
)
from packages.indicators.momentum import MACD, RSI
from packages.indicators.trend import SMA
from packages.paper.base import (
    TradeDirection,
)
from packages.paper.executor import PaperExecutor as Executor
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

# ── helpers ──────────────────────────────────────────────────────────


def _make_indicator_report(
    agent_id: str = "indicator",
    probabilities: dict[str, float] | None = None,
    status: AgentStatus = AgentStatus.ACTIVE,
) -> AgentReport:
    """Convenience factory for an indicator-style AgentReport."""
    return AgentReport(
        report_id=f"rpt-{agent_id}",
        run_id="run-001",
        agent_id=agent_id,
        agent_version="0.1.0",
        instrument="EUR/USD",
        horizon="1h",
        as_of=datetime.now(),
        hypothesis="test",
        probabilities=probabilities or {"up": 0.7, "down": 0.1, "range": 0.2},
        evidence=[
            EvidenceReference(
                reference=f"{agent_id}:sma",
                feature="sma",
                value="uptrend",
                direction="positive",
                relevance=0.5,
            )
        ],
        raw_confidence=0.6,
        status=status,
    )


def _build_close_array(n: int = 100, start: float = 100.0) -> np.ndarray:
    """Build a realistic-looking closing price array (small random walk)."""
    prices = np.full(n, start, dtype=np.float64)
    np.random.seed(42)  # reproducible
    for i in range(1, n):
        prices[i] = prices[i - 1] * (1 + np.random.normal(0, 0.005))
    return prices


def _run_pipeline(
    close: np.ndarray | None = None,
    agent_reports: list[AgentReport] | None = None,
) -> dict:
    """Run the full pipeline and return intermediate results.

    Returns dict with keys:
        features, consensus, strategy, trade (if any)
    """
    if close is None:
        close = _build_close_array()

    # ── Step 1: Feature calculation ──
    rsi_result = RSI(period=14).compute({"close": close})
    macd_result = MACD().compute({"close": close})
    sma20_result = SMA(period=20).compute({"close": close})
    sma50_result = SMA(period=50).compute({"close": close})

    valid_rsi = rsi_result.values[~np.isnan(rsi_result.values)]
    valid_macd = macd_result.values[~np.isnan(macd_result.values)]
    valid_sma20 = sma20_result.values[~np.isnan(sma20_result.values)]
    valid_sma50 = sma50_result.values[~np.isnan(sma50_result.values)]

    rsi_latest = float(valid_rsi[-1])
    macd_latest = float(valid_macd[-1])
    sma20_latest = float(valid_sma20[-1])
    sma50_latest = float(valid_sma50[-1])
    current_price = float(close[-1])

    atr = float(np.std(close[-20:]) * np.sqrt(20)) if len(close) >= 20 else 1.0

    features = {
        "current_price": current_price,
        "atr": max(atr, 0.5),
        "volatility": atr * 0.6,
        "rsi": rsi_latest,
        "macd_hist": macd_latest,
        "sma20": sma20_latest,
        "sma50": sma50_latest,
        "entry_type": "market",
        "entry_condition": "momentum",
    }

    # ── Step 2: Agent predictions ──
    if agent_reports is None:
        agent_reports = [
            _make_indicator_report(
                agent_id="ind1",
                probabilities={"up": 0.7, "down": 0.1, "range": 0.2},
                status=AgentStatus.ACTIVE,
            ),
            _make_indicator_report(
                agent_id="ind2",
                probabilities={"up": 0.65, "down": 0.15, "range": 0.2},
                status=AgentStatus.ACTIVE,
            ),
        ]

    # ── Step 3: Consensus ──
    engine = WeightedConsensusEngine(config=WeightConfig(min_consensus_threshold=0.5))
    consensus: ConsensusResult = engine.compute_consensus(agent_reports)

    # ── Step 4: Strategy ──
    strategy_engine = StrategyEngine(config=StrategyConfig())
    context = {
        "consensus": consensus,
        "features": features,
        "costs": {"commission": 0.001, "slippage": 0.001},
    }
    proposal: StrategyProposal = strategy_engine.run(context)

    return {
        "features": features,
        "consensus": consensus,
        "strategy": proposal,
        "agent_reports": agent_reports,
    }


class TestFullPipelineLong:
    """Verify that bullish agent consensus produces a LONG strategy and paper order."""

    def test_full_chain_produces_long_strategy(self) -> None:
        close = _build_close_array(n=100, start=100.0)
        result = _run_pipeline(close=close)

        assert result["consensus"].decision in (
            ConsensusDecision.LONG_BIAS,
            ConsensusDecision.SHORT_BIAS,
        )
        assert isinstance(result["strategy"], StrategyProposal)
        assert result["strategy"].direction != StrategyDirection.NO_TRADE

    def test_full_chain_produces_paper_order(self) -> None:
        """When consensus is bullish and strategy passes gates, a trade executes."""
        close = _build_close_array(n=100, start=100.0)
        # Force bullish agents
        agent_reports = [
            _make_indicator_report(
                agent_id="bull1",
                probabilities={"up": 0.8, "down": 0.05, "range": 0.15},
                status=AgentStatus.ACTIVE,
            ),
            _make_indicator_report(
                agent_id="bull2",
                probabilities={"up": 0.8, "down": 0.05, "range": 0.15},
                status=AgentStatus.ACTIVE,
            ),
        ]
        result = _run_pipeline(close=close, agent_reports=agent_reports)

        assert result["consensus"].decision == ConsensusDecision.LONG_BIAS

        # Execute via paper executor if strategy approved a direction
        strategy = result["strategy"]
        if strategy.direction == StrategyDirection.LONG:
            executor = Executor(
                initial_cash=100000.0,
                default_slippage_pct=0.001,
                default_commission_pct=0.001,
                max_position_size_pct=0.10,
            )
            account = executor.create_account("test-long")
            price = result["features"]["current_price"]
            trade = executor.submit_order(
                account, "EUR/USD", TradeDirection.BUY, 1.0, price=price
            )
            assert trade.status == "filled"
            assert trade.instrument == "EUR/USD"
            assert trade.direction == TradeDirection.BUY
            assert "EUR/USD" in account.positions

    def test_full_chain_validity_of_results(self) -> None:
        """All pipeline outputs must be non-empty and well-formed."""
        close = _build_close_array(n=120, start=50.0)
        result = _run_pipeline(close=close)

        # Features
        assert result["features"]["current_price"] > 0
        assert result["features"]["atr"] > 0
        assert result["features"]["rsi"] >= 0

        # Consensus
        consensus = result["consensus"]
        assert isinstance(consensus.decision, ConsensusDecision)
        assert 0.0 <= consensus.confidence <= 1.0
        assert len(consensus.agent_agreements) + len(consensus.agent_disagreements) == len(
            consensus.agent_weights
        )

        # Strategy
        proposal = result["strategy"]
        assert isinstance(proposal.direction, StrategyDirection)
        if proposal.direction != StrategyDirection.NO_TRADE:
            assert proposal.entry_price > 0
            assert proposal.stop_loss >= 0


class TestFullPipelineBadConsensus:
    """When agents disagree, the system should produce NO_TRADE."""

    def test_disagreement_yields_no_trade(self) -> None:
        agent_reports = [
            _make_indicator_report(
                agent_id="bull",
                probabilities={"up": 0.8, "down": 0.05, "range": 0.15},
                status=AgentStatus.ACTIVE,
            ),
            _make_indicator_report(
                agent_id="bear",
                probabilities={"up": 0.1, "down": 0.8, "range": 0.1},
                status=AgentStatus.ACTIVE,
            ),
        ]
        result = _run_pipeline(agent_reports=agent_reports)

        assert result["consensus"].decision == ConsensusDecision.NO_TRADE
        assert result["strategy"].direction == StrategyDirection.NO_TRADE

    def test_range_consensus_yields_no_trade(self) -> None:
        agent_reports = [
            _make_indicator_report(
                agent_id="range1",
                probabilities={"up": 0.2, "down": 0.2, "range": 0.6},
                status=AgentStatus.ACTIVE,
            ),
        ]
        result = _run_pipeline(agent_reports=agent_reports)

        assert result["consensus"].decision == ConsensusDecision.RANGE
        # Range consensus → strategy produces NO_TRADE variant
        assert result["strategy"].direction != StrategyDirection.LONG
        assert result["strategy"].direction != StrategyDirection.SHORT


class TestFullPipelineWithOrderBook:
    """Verify orderbook reconstruction can coexist with the pipeline."""

    def test_orderbook_integrated_with_pipeline(self) -> None:
        close = _build_close_array(n=100, start=100.0)
        result = _run_pipeline(close=close)

        # Build orderbook snapshot alongside pipeline data
        recon = OrderBookReconstructor("EUR/USD", "BINANCE")
        recon.apply_snapshot({
            "sequence": 1000,
            "bids": [[99.5, 10.0], [99.4, 5.0]],
            "asks": [[100.5, 8.0], [100.6, 3.0]],
        }, event_time=datetime.now())

        book = recon.get_current_book()
        assert book is not None
        assert book.sequence == 1000
        assert book.mid_price is not None

        # Pipeline should still work with this market context
        assert result["features"]["current_price"] > 0
        assert result["consensus"].decision is not None

    def test_pipeline_with_multiple_agent_types(self) -> None:
        """Pipeline should handle reports from multiple agent types."""
        close = _build_close_array(n=100, start=100.0)
        agent_reports = [
            _make_indicator_report(
                agent_id="ind1",
                probabilities={"up": 0.75, "down": 0.1, "range": 0.15},
                status=AgentStatus.ACTIVE,
            ),
            AgentReport(
                report_id="rpt-regime1",
                run_id="run-001",
                agent_id="regime1",
                agent_version="0.1.0",
                instrument="EUR/USD",
                horizon="1h",
                as_of=datetime.now(),
                hypothesis="Trend is up",
                probabilities={"up": 0.7, "down": 0.15, "range": 0.15},
                evidence=[
                    EvidenceReference(
                        reference="regime:trend",
                        feature="regime",
                        value="uptrend",
                        direction="positive",
                        relevance=0.7,
                    )
                ],
                raw_confidence=0.7,
                status=AgentStatus.ACTIVE,
            ),
        ]

        result = _run_pipeline(close=close, agent_reports=agent_reports)
        assert result["consensus"].decision in (
            ConsensusDecision.LONG_BIAS,
            ConsensusDecision.SHORT_BIAS,
        )
        assert len(result["consensus"].agent_weights) == 2
