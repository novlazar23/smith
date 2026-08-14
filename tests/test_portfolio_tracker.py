from __future__ import annotations

import threading

from trading_harness.models import (
    PaperPosition,
    PaperPositionStatus,
    PortfolioState,
)
from trading_harness.services.portfolio_tracker import (
    InMemoryPortfolioStore,
    PersistedPortfolioStore,
    PortfolioTracker,
)


def _make_position(
    symbol: str = "AAPL",
    side: str = "LONG",
    entry_price: float = 100.0,
    quantity: float = 10.0,
    unrealized_pnl: float = 0.0,
    realized_pnl: float = 0.0,
    status: PaperPositionStatus = PaperPositionStatus.OPEN,
) -> PaperPosition:
    from datetime import UTC, datetime
    from uuid import uuid4

    return PaperPosition(
        id=f"paper-pos-{uuid4()}",
        trade_id="trade-1",
        run_id="run-1",
        symbol=symbol,
        side=side,
        entry_price=entry_price,
        quantity=quantity,
        fees=0.0,
        current_price=entry_price + unrealized_pnl / quantity if quantity > 0 else entry_price,
        unrealized_pnl=unrealized_pnl,
        realized_pnl=realized_pnl,
        stop_price=entry_price * 0.95,
        target_price=entry_price * 1.05,
        status=status,
        open_timestamp=datetime.now(UTC),
    )


class TestInitialState:
    """Initial state with no positions."""

    def test_initial_state_no_positions(self) -> None:
        """Given no positions, equity equals start_equity and drawdown is zero."""
        tracker = PortfolioTracker(start_equity=100000.0)
        state = tracker.update([])

        assert state.current_equity == 100000.0
        assert state.total_realized_pnl == 0.0
        assert state.total_unrealized_pnl == 0.0
        assert state.current_drawdown == 0.0
        assert state.max_drawdown == 0.0
        assert state.peak_equity == 100000.0
        assert state.positions == {}
        assert state.symbols == []


class TestProfitableTrades:
    """Equity increases with profitable positions."""

    def test_profitable_position_increases_equity(self) -> None:
        """Given a profitable position, equity reflects unrealized gains."""
        tracker = PortfolioTracker(start_equity=100000.0)
        pos = _make_position(
            symbol="AAPL", entry_price=100.0, quantity=10.0, unrealized_pnl=500.0
        )
        state = tracker.update([pos])

        assert state.current_equity == 100500.0
        assert state.total_unrealized_pnl == 500.0
        assert state.total_realized_pnl == 0.0


class TestLosingTrades:
    """Equity decreases with losing positions."""

    def test_losing_position_decreases_equity(self) -> None:
        """Given a losing position, equity reflects unrealized losses."""
        tracker = PortfolioTracker(start_equity=100000.0)
        pos = _make_position(
            symbol="AAPL", entry_price=100.0, quantity=10.0, unrealized_pnl=-300.0
        )
        state = tracker.update([pos])

        assert state.current_equity == 99700.0
        assert state.total_unrealized_pnl == -300.0


class TestDrawdown:
    """Drawdown calculation and tracking."""

    def test_drawdown_after_equity_drop(self) -> None:
        """Equity drops after first update — drawdown should track."""
        tracker = PortfolioTracker(start_equity=100000.0)

        # First update: equity rises
        pos1 = _make_position(
            symbol="AAPL", entry_price=100.0, quantity=10.0, unrealized_pnl=500.0
        )
        tracker.update([pos1])

        # Second update: equity drops
        pos2 = _make_position(
            symbol="AAPL", entry_price=100.0, quantity=10.0, unrealized_pnl=-1500.0
        )
        state = tracker.update([pos2])

        assert state.current_equity == 98500.0
        assert state.current_drawdown > 0.0
        assert state.max_drawdown == state.current_drawdown

    def test_max_drawdown_across_updates(self) -> None:
        """Multiple updates track the deepest drawdown."""
        tracker = PortfolioTracker(start_equity=100000.0)

        # Rise to 105000
        tracker.update([_make_position(symbol="AAPL", entry_price=100.0, quantity=10.0, unrealized_pnl=500.0)])
        # Drop to 90000
        tracker.update([_make_position(symbol="AAPL", entry_price=100.0, quantity=10.0, unrealized_pnl=-10500.0)])

        tracker._store.get(tracker._store.by_run("")[0].id) if tracker._store.by_run("") else None
        # Max drawdown should be from peak 105000 to 90000
        states = tracker._store.all()
        latest = max(states, key=lambda s: s.timestamp)
        assert latest.max_drawdown > 0.10  # > 10%

    def test_no_drawdown_when_equity_rises(self) -> None:
        """Drawdown is zero when equity never drops below peak."""
        tracker = PortfolioTracker(start_equity=100000.0)

        tracker.update([_make_position(symbol="AAPL", entry_price=100.0, quantity=10.0, unrealized_pnl=500.0)])
        tracker.update([_make_position(symbol="AAPL", entry_price=100.0, quantity=10.0, unrealized_pnl=1500.0)])

        states = tracker._store.all()
        latest = max(states, key=lambda s: s.timestamp)
        assert latest.current_drawdown == 0.0
        assert latest.max_drawdown == 0.0


class TestExposureTracking:
    """Exposure calculation by symbol."""

    def test_exposure_sum_of_notional(self) -> None:
        """Exposure = sum of abs(quantity) * entry_price per symbol."""
        tracker = PortfolioTracker(start_equity=100000.0)
        pos = _make_position(symbol="AAPL", entry_price=100.0, quantity=10.0)
        exposure = tracker.calculate_exposure([pos])

        assert exposure["AAPL"] == 1000.0


class TestMixedPositions:
    """Update with mixed LONG/SHORT positions."""

    def test_mixed_long_short_positions(self) -> None:
        """LONG and SHORT positions contribute independently to equity."""
        tracker = PortfolioTracker(start_equity=100000.0)

        long_pos = _make_position(
            symbol="AAPL", side="LONG", entry_price=100.0, quantity=10.0, unrealized_pnl=500.0
        )
        short_pos = _make_position(
            symbol="SPY", side="SHORT", entry_price=400.0, quantity=5.0, unrealized_pnl=200.0
        )
        state = tracker.update([long_pos, short_pos])

        assert state.current_equity == 100700.0
        assert state.total_unrealized_pnl == 700.0
        assert set(state.symbols) == {"AAPL", "SPY"}


class TestStartEquityConfigurable:
    """Start equity can be configured."""

    def test_custom_start_equity(self) -> None:
        """Configurable start_equity is used in calculations."""
        tracker = PortfolioTracker(start_equity=50000.0)
        state = tracker.update([])

        assert state.start_equity == 50000.0
        assert state.current_equity == 50000.0
        assert tracker.start_equity == 50000.0


class TestDeterminism:
    """Same positions produce same state."""

    def test_same_positions_same_state(self) -> None:
        """Identical positions yield identical equity and drawdown."""
        tracker_a = PortfolioTracker(start_equity=100000.0)
        tracker_b = PortfolioTracker(start_equity=100000.0)

        pos = _make_position(symbol="AAPL", entry_price=100.0, quantity=10.0, unrealized_pnl=300.0)
        state_a = tracker_a.update([pos])
        state_b = tracker_b.update([pos])

        assert state_a.current_equity == state_b.current_equity
        assert state_a.total_realized_pnl == state_b.total_realized_pnl
        assert state_a.total_unrealized_pnl == state_b.total_unrealized_pnl
        assert state_a.max_drawdown == state_b.max_drawdown


class TestInMemoryStore:
    """InMemoryPortfolioStore add/get/by_run."""

    def test_add_get(self) -> None:
        store = InMemoryPortfolioStore()
        state = PortfolioState(run_id="run-1", start_equity=100000.0)
        stored = store.add(state)
        assert stored.id == state.id

        retrieved = store.get(state.id)
        assert retrieved is not None
        assert retrieved.run_id == "run-1"

    def test_get_missing_returns_none(self) -> None:
        store = InMemoryPortfolioStore()
        assert store.get("nonexistent") is None

    def test_by_run_filters(self) -> None:
        store = InMemoryPortfolioStore()
        s1 = PortfolioState(run_id="run-1", start_equity=100000.0)
        s2 = PortfolioState(run_id="run-2", start_equity=200000.0)
        store.add(s1)
        store.add(s2)

        assert len(store.by_run("run-1")) == 1
        assert len(store.by_run("run-2")) == 1
        assert store.by_run("run-1")[0].start_equity == 100000.0


class TestThreadSafety:
    """Thread-safe concurrent access."""

    def test_concurrent_updates(self) -> None:
        """Multiple threads updating concurrently produce correct count."""
        tracker = PortfolioTracker(start_equity=100000.0)
        results: list[PortfolioState] = []
        lock = threading.Lock()

        def update_thread(i: int) -> None:
            pos = _make_position(
                symbol="SYM", entry_price=100.0, quantity=1.0,
                unrealized_pnl=float(i * 100),
            )
            state = tracker.update([pos])
            with lock:
                results.append(state)

        threads = [threading.Thread(target=update_thread, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 5

        for s in results:
            assert s.current_equity > 0.0
            assert s.positions == {"SYM": 1.0}


class TestPersistedStoreFallback:
    """PersistedPortfolioStore falls back to in-memory when DB unavailable."""

    def test_fallback_when_db_unavailable(self) -> None:
        store = PersistedPortfolioStore(db=None)
        state = PortfolioState(run_id="run-1", start_equity=100000.0)
        store.add(state)

        retrieved = store.get(state.id)
        assert retrieved is not None
        assert retrieved.run_id == "run-1"

    def test_by_run_fallback(self) -> None:
        store = PersistedPortfolioStore(db=None)
        s1 = PortfolioState(run_id="run-a", start_equity=100000.0)
        s2 = PortfolioState(run_id="run-b", start_equity=200000.0)
        store.add(s1)
        store.add(s2)

        assert len(store.by_run("run-a")) == 1
        assert len(store.all()) == 2


class TestGetStateAndGetHistory:
    """PortfolioTracker get_state and get_history methods."""

    def test_get_state_returns_latest(self) -> None:
        store = InMemoryPortfolioStore()
        tracker = PortfolioTracker(start_equity=100000.0, store=store)

        pos = _make_position(symbol="AAPL", entry_price=100.0, quantity=10.0, unrealized_pnl=500.0)
        state = tracker.update([pos])

        # Manually set run_id on the stored state
        state.run_id = "test-run"
        store.add(state)

        retrieved = tracker.get_state("test-run")
        assert retrieved is not None
        assert retrieved.run_id == "test-run"

    def test_get_state_missing_returns_none(self) -> None:
        store = InMemoryPortfolioStore()
        tracker = PortfolioTracker(start_equity=100000.0, store=store)
        assert tracker.get_state("nonexistent") is None

    def test_get_history_returns_all_for_run(self) -> None:
        store = InMemoryPortfolioStore()
        tracker = PortfolioTracker(start_equity=100000.0, store=store)

        pos1 = _make_position(symbol="AAPL", entry_price=100.0, quantity=10.0, unrealized_pnl=100.0)
        tracker.update([pos1])
        pos2 = _make_position(symbol="AAPL", entry_price=100.0, quantity=10.0, unrealized_pnl=200.0)
        tracker.update([pos2])

        # States stored with empty run_id by default; set run_id for querying
        all_states = store.all()
        for s in all_states:
            s.run_id = "history-run"
            store.add(s)

        history = tracker.get_history("history-run")
        assert len(history) == 2