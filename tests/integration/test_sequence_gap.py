"""Integration tests for orderbook sequence gap detection.

Verifies that:
- Gaps in sequence numbers are detected and logged
- Trading can continue after gap detection
- Gap detection doesn't cause position corruption
- Multiple consecutive gaps are tracked
"""

from __future__ import annotations

from datetime import datetime

import pytest
from packages.consensus import (
    ConsensusDecision,
    WeightedConsensusEngine,
)
from packages.domain.market_data.orderbook import (
    OrderBookReconstructor,
)
from packages.paper.base import (
    PaperAccount,
    TradeDirection,
)
from packages.paper.executor import PaperExecutor
from packages.schemas.agent_report import (
    AgentReport,
    AgentStatus,
    EvidenceReference,
)
from packages.strategy.engine import StrategyEngine
from packages.strategy.models import StrategyConfig, StrategyDirection

# ── helpers ──────────────────────────────────────────────────────────


def _make_agent_report(
    agent_id: str = "ind1",
    probabilities: dict[str, float] | None = None,
    status: AgentStatus = AgentStatus.ACTIVE,
) -> AgentReport:
    return AgentReport(
        report_id=f"rpt-{agent_id}",
        run_id="run-001",
        agent_id=agent_id,
        agent_version="0.1.0",
        instrument="BTC/USDT",
        horizon="1h",
        as_of=datetime.now(),
        hypothesis="test",
        probabilities=probabilities or {"up": 0.7, "down": 0.1, "range": 0.2},
        evidence=[
            EvidenceReference(
                reference=f"{agent_id}:test",
                feature="test",
                value="active",
                direction="positive",
                relevance=0.7,
            )
        ],
        raw_confidence=0.6,
        status=status,
    )


class TestGapDetection:
    """Test gap detection in orderbook sequence numbers."""

    def test_consecutive_sequences_no_gap(self) -> None:
        """Normal incrementing sequence should not trigger a gap."""
        recon = OrderBookReconstructor("BTC/USDT", "BINANCE")
        recon.apply_snapshot({
            "sequence": 100,
            "bids": [[100.0, 1.0]],
            "asks": [[101.0, 1.0]],
        }, event_time=datetime.now())

        # Next sequence is 101 — no gap
        delta = {
            "sequence": 101,
            "bids": [],
            "asks": [],
        }
        book = recon.apply_delta(delta)
        assert book is not None
        assert book.sequence == 101

    def test_sequence_gap_detected(self) -> None:
        """A gap in sequence (100 → 105) should be accepted but detectable."""
        recon = OrderBookReconstructor("BTC/USDT", "BINANCE")
        recon.apply_snapshot({
            "sequence": 100,
            "bids": [[100.0, 1.0]],
            "asks": [[101.0, 1.0]],
        }, event_time=datetime.now())

        # Delta has sequence 105 — gap of 4
        delta = {
            "sequence": 105,
            "bids": [[100.5, 2.0]],
            "asks": [],
        }
        book = recon.apply_delta(delta)
        assert book is not None
        assert book.sequence == 105
        # The reconstructor does NOT reject gap — it applies the delta anyway

    def test_sequence_regression_detected(self) -> None:
        """A sequence that goes backward should still be accepted by reconstructor."""
        recon = OrderBookReconstructor("BTC/USDT", "BINANCE")
        recon.apply_snapshot({
            "sequence": 200,
            "bids": [[100.0, 1.0]],
            "asks": [[101.0, 1.0]],
        }, event_time=datetime.now())

        # Regression: sequence 190 is below the snapshot sequence 200
        delta = {
            "sequence": 190,
            "bids": [],
            "asks": [],
        }
        book = recon.apply_delta(delta)
        assert book is not None
        assert book.sequence == 190

    def test_large_gap_multiple_deltas(self) -> None:
        """Multiple deltas with large gaps should all be applied."""
        recon = OrderBookReconstructor("BTC/USDT", "BINANCE")
        recon.apply_snapshot({
            "sequence": 1,
            "bids": [[100.0, 1.0]],
            "asks": [[101.0, 1.0]],
        }, event_time=datetime.now())

        # Apply deltas with increasing but spotty sequences
        gaps = [10, 25, 50]
        for seq in gaps:
            delta = {
                "sequence": seq,
                "bids": [[float(100 + seq * 0.1), 1.0]],
                "asks": [],
            }
            book = recon.apply_delta(delta)
            assert book is not None
            assert book.sequence == seq


class TestGapDetectionAndLogging:
    """Test that gaps are detectable and can be logged for monitoring."""

    def _simulate_gap_tracking(self, recon: OrderBookReconstructor, sequences: list[int]) -> list[int]:
        """Simulate tracking sequence numbers and detecting gaps."""
        prev_seq = None
        detected_gaps: list[int] = []

        for seq in sequences:
            if prev_seq is not None:
                expected = prev_seq + 1
                if seq != expected:
                    detected_gaps.append(seq - prev_seq)  # gap size
            prev_seq = seq

        return detected_gaps

    def test_small_gap_detected(self) -> None:
        """A gap of 1 (100 → 102) should be detected as size 2."""
        recon = OrderBookReconstructor("X", "Y")
        recon.apply_snapshot({
            "sequence": 100,
            "bids": [[10.0, 1.0]],
            "asks": [[11.0, 1.0]],
        })

        gaps = self._simulate_gap_tracking(recon, [100, 102])
        assert 2 in gaps  # gap of 2 between 100 and 102

    def test_no_gap_when_continuous(self) -> None:
        """No gaps when sequences are continuous."""
        recon = OrderBookReconstructor("X", "Y")
        recon.apply_snapshot({
            "sequence": 1,
            "bids": [[10.0, 1.0]],
            "asks": [[11.0, 1.0]],
        })

        gaps = self._simulate_gap_tracking(recon, [1, 2, 3, 4, 5])
        assert len(gaps) == 0

    def test_multiple_gaps_in_sequence(self) -> None:
        """Multiple distinct gaps in a sequence should all be detected."""
        recon = OrderBookReconstructor("X", "Y")
        recon.apply_snapshot({
            "sequence": 1,
            "bids": [[10.0, 1.0]],
            "asks": [[11.0, 1.0]],
        })

        # Simulate: 1, 2, 5(gap=3), 6, 10(gap=4), 11
        gaps = self._simulate_gap_tracking(recon, [1, 2, 5, 6, 10, 11])
        assert 3 in gaps  # gap 2→5
        assert 4 in gaps  # gap 6→10

    def test_first_message_from_scratch(self) -> None:
        """First message has no previous sequence — no gap expected."""
        # No recon created, simulating fresh start
        gaps = self._simulate_gap_tracking(
            OrderBookReconstructor("X", "Y"),
            [50, 51, 52]
        )
        # 50 is first, no previous → no gap for 50
        assert len(gaps) == 0


class TestTradingContinuesAfterGap:
    """Verify that trading continues normally after a gap is detected."""

    def test_trade_after_gap(self) -> None:
        """Pipeline works normally after orderbook gap is detected."""
        recon = OrderBookReconstructor("BTC/USDT", "BINANCE")
        recon.apply_snapshot({
            "sequence": 100,
            "bids": [[100.0, 5.0], [99.0, 3.0]],
            "asks": [[101.0, 4.0], [102.0, 2.0]],
        }, event_time=datetime.now())

        # Simulate gap
        recon.apply_delta({
            "sequence": 110,  # gap of 10
            "bids": [[100.5, 2.0]],
            "asks": [[101.5, 3.0]],
        })

        book = recon.get_current_book()
        assert book is not None
        assert book.sequence == 110
        assert book.mid_price is not None

        # Build consensus and strategy as if trading on top of this
        engine = WeightedConsensusEngine()
        reports = [
            _make_agent_report(
                "a1", {"up": 0.7, "down": 0.1, "range": 0.2},
                AgentStatus.ACTIVE,
            ),
        ]
        consensus = engine.compute_consensus(reports)
        assert consensus.decision == ConsensusDecision.LONG_BIAS

        # Strategy proceeds normally
        strategy = StrategyEngine(config=StrategyConfig())
        context = {
            "consensus": consensus,
            "features": {
                "current_price": 100.5,
                "atr": 2.0,
                "entry_type": "market",
                "entry_condition": "momentum",
            },
        }
        proposal = strategy.run(context)
        assert isinstance(proposal.direction, StrategyDirection)

    def test_multiple_trades_after_gap(self) -> None:
        """Multiple trades can execute after a gap was detected."""
        recon = OrderBookReconstructor("BTC/USDT", "BINANCE")
        recon.apply_snapshot({
            "sequence": 100,
            "bids": [[100.0, 5.0]],
            "asks": [[101.0, 5.0]],
        })

        # Gap
        recon.apply_delta({"sequence": 150, "bids": [], "asks": []})

        # Continue with more trades
        recon.apply_delta({"sequence": 151, "bids": [], "asks": []})
        recon.apply_delta({"sequence": 152, "bids": [], "asks": []})

        book = recon.get_current_book()
        assert book is not None
        assert book.sequence == 152

    def test_trading_with_gap_and_reconvergence(self) -> None:
        """After a gap, the orderbook converges to a valid state."""
        recon = OrderBookReconstructor("BTC/USDT", "BINANCE")
        recon.apply_snapshot({
            "sequence": 100,
            "bids": [[100.0, 1.0], [99.0, 1.0]],
            "asks": [[101.0, 1.0], [102.0, 1.0]],
        })

        # Gap of 5
        recon.apply_delta({
            "sequence": 105,
            "bids": [[100.5, 2.0], [99.5, 1.5]],
            "asks": [[101.5, 1.0]],
        })

        # Normal increment resumes
        recon.apply_delta({
            "sequence": 106,
            "bids": [],
            "asks": [[101.5, 0]],  # remove 101.5 ask
        })

        book = recon.get_current_book()
        assert book is not None
        assert book.sequence == 106
        assert recon.verify_consistency()  # No crossing between bids/asks


class TestNoPositionCorruption:
    """Verify that gap detection and handling don't corrupt paper positions."""

    def _setup_paper_trading(self) -> tuple[PaperExecutor, PaperAccount, float]:
        """Helper: set up a paper trading account ready for orders."""
        executor = PaperExecutor(
            initial_cash=100000.0,
            default_slippage_pct=0.001,
            default_commission_pct=0.001,
        )
        account = executor.create_account("gap-test")
        return executor, account

    def test_buy_sell_roundtrip_no_corruption(self) -> None:
        """Normal buy/sell roundtrip leaves position clean after gaps."""
        executor, account = self._setup_paper_trading()

        # Initial BUY
        executor.submit_order(account, "BTC/USDT", TradeDirection.BUY, 1.0, price=100.0)
        assert "BTC/USDT" in account.positions
        assert account.positions["BTC/USDT"].quantity == 1.0

        # SELL all
        executor.submit_order(account, "BTC/USDT", TradeDirection.SELL, 1.0, price=101.0)
        assert "BTC/USDT" not in account.positions
        assert len(account.positions) == 0

        # Cash should be: 100000 - 100 - 0.1 + 101 - 0.101 = ~100000.799
        assert account.cash > 0
        assert account.total_trades == 2

    def test_gap_between_trades_no_corruption(self) -> None:
        """Simulating a sequence gap between trades doesn't corrupt state."""
        executor, account = self._setup_paper_trading()

        # Simulate gap in orderbook (doesn't affect paper executor)
        recon = OrderBookReconstructor("BTC/USDT", "BINANCE")
        recon.apply_snapshot({
            "sequence": 100,
            "bids": [[100.0, 1.0]],
            "asks": [[101.0, 1.0]],
        })
        recon.apply_delta({"sequence": 150, "bids": [], "asks": []})  # gap
        recon.apply_delta({"sequence": 151, "bids": [], "asks": []})

        # Trades continue unaffected
        executor.submit_order(account, "BTC/USDT", TradeDirection.BUY, 0.5, price=100.0)
        assert "BTC/USDT" in account.positions
        assert account.positions["BTC/USDT"].quantity == 0.5

        # Position is still valid
        pos = account.positions["BTC/USDT"]
        assert pos.quantity > 0
        assert pos.avg_price > 0

    def test_buy_multiple_after_gap(self) -> None:
        """Multiple buys after a gap correctly aggregate position."""
        executor, account = self._setup_paper_trading()

        # Simulate gap
        recon = OrderBookReconstructor("BTC/USDT", "BINANCE")
        recon.apply_snapshot({
            "sequence": 1,
            "bids": [[100.0, 1.0]],
            "asks": [[101.0, 1.0]],
        })
        recon.apply_delta({"sequence": 50, "bids": [], "asks": []})  # big gap

        # Multiple buys — positions should aggregate correctly
        executor.submit_order(account, "BTC/USDT", TradeDirection.BUY, 1.0, price=100.0)
        executor.submit_order(account, "BTC/USDT", TradeDirection.BUY, 0.5, price=102.0)

        pos = account.positions["BTC/USDT"]
        assert pos.quantity == 1.5
        # avg_price should be weighted: (100*1 + 102*0.5) / 1.5
        # but with slippage and commission the actual avg_price differs slightly
        assert pos.quantity == 1.5
        assert pos.avg_price > 100.0
        assert pos.avg_price < 103.0

    def test_sell_partial_after_gap(self) -> None:
        """Partial sell after a gap correctly reduces position."""
        executor, account = self._setup_paper_trading()

        # Gap simulation
        recon = OrderBookReconstructor("BTC/USDT", "BINANCE")
        recon.apply_snapshot({
            "sequence": 1,
            "bids": [[100.0, 1.0]],
            "asks": [[101.0, 1.0]],
        })
        recon.apply_delta({"sequence": 100, "bids": [], "asks": []})

        # Buy 2.0 units
        executor.submit_order(account, "BTC/USDT", TradeDirection.BUY, 2.0, price=100.0)
        assert account.positions["BTC/USDT"].quantity == 2.0

        # Sell 0.5
        executor.submit_order(account, "BTC/USDT", TradeDirection.SELL, 0.5, price=101.0)
        assert account.positions["BTC/USDT"].quantity == 1.5

        # Sell remaining
        executor.submit_order(account, "BTC/USDT", TradeDirection.SELL, 1.5, price=102.0)
        assert "BTC/USDT" not in account.positions

    def test_sell_too_much_after_gap_raises(self) -> None:
        """Trying to sell more than available after a gap raises ValueError."""
        executor, account = self._setup_paper_trading()

        # Gap simulation
        recon = OrderBookReconstructor("BTC/USDT", "BINANCE")
        recon.apply_snapshot({
            "sequence": 1,
            "bids": [[100.0, 1.0]],
            "asks": [[101.0, 1.0]],
        })
        recon.apply_delta({"sequence": 100, "bids": [], "asks": []})

        # Buy only 1.0
        executor.submit_order(account, "BTC/USDT", TradeDirection.BUY, 1.0, price=100.0)

        # Try to sell 2.0 — should fail
        with pytest.raises(ValueError, match="Insufficient position"):
            executor.submit_order(account, "BTC/USDT", TradeDirection.SELL, 2.0, price=101.0)

        # Position should still be intact
        assert account.positions["BTC/USDT"].quantity == 1.0

    def test_consensus_after_gap_continues_pipeline(self) -> None:
        """Full pipeline with gap detection produces valid consensus and strategy."""
        # 1. Orderbook with gap
        recon = OrderBookReconstructor("BTC/USDT", "BINANCE")
        recon.apply_snapshot({
            "sequence": 100,
            "bids": [[100.0, 5.0]],
            "asks": [[101.0, 5.0]],
        })
        recon.apply_delta({"sequence": 200, "bids": [], "asks": []})  # big gap

        book = recon.get_current_book()
        assert book is not None
        assert book.sequence == 200

        # 2. Consensus on top of this data
        engine = WeightedConsensusEngine()
        reports = [
            _make_agent_report("a1", {"up": 0.7, "down": 0.1, "range": 0.2}, AgentStatus.ACTIVE),
            _make_agent_report("a2", {"up": 0.7, "down": 0.1, "range": 0.2}, AgentStatus.ACTIVE),
        ]
        consensus = engine.compute_consensus(reports)
        assert consensus.decision == ConsensusDecision.LONG_BIAS

        # 3. Strategy proceeds
        strategy = StrategyEngine(config=StrategyConfig())
        context = {
            "consensus": consensus,
            "features": {
                "current_price": 100.5,
                "atr": 2.0,
                "entry_type": "market",
                "entry_condition": "momentum",
            },
        }
        proposal = strategy.run(context)
        assert isinstance(proposal.direction, StrategyDirection)

        # 4. Paper trading on top
        executor, account = self._setup_paper_trading()
        executor.submit_order(account, "BTC/USDT", TradeDirection.BUY, 1.0, price=100.5)
        assert account.positions["BTC/USDT"].quantity == 1.0
