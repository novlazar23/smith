"""Tests: shadow_decision — deterministische Signal-Aggregation und Decision-Bildung (WI-ST-04).

Sichert die Akzeptanzkriterien von Spec ST.6 und Epic WI-ST-04 ab:

- ST.6 (a): 2 LONG + 1 SHORT -> LONG (Majorität der NON-NO_TRADE-Signale)
- ST.6 (b): 1 LONG + 1 SHORT -> NO_TRADE (Gleichstand)
- ST.6 (c): mittlere Confidence 0.5 < shadow_min_confidence 0.6 -> NO_TRADE
- ST.6 (d): requested_quantity > max_position_size -> auf max gedeckelt
- ST.6 (e): gleiche Eingaben -> bit-identisches Ergebnis (Determinismus)
- Z4: Agenten-Statusfilter (ACTIVE/CHAMPION) passiert im Loop, NICHT hier;
  leere Eingabe = keine qualifizierenden Agenten -> NO_TRADE mit Grund
  "NO_ACTIVE_CHAMPIONS"
- Z5: aggregate_signals ist der MVP-Ersatz für ein (nicht existierendes)
  Consensus-Modul
- Epic E2: NO_TRADE-Gründervokabular ist exakt
  {NO_ACTIVE_CHAMPIONS, BELOW_MIN_CONFIDENCE}
- Ticker-Vertrag: Entry-Preis kommt vom festen Schlüssel "ticker" in
  snapshot.data (Spec ST.1)
- decision_id ist deterministisch: f"shadow-dec-{{snapshot.id}}-{{symbol}}"

Alle Funktionen unter Test sind pure: kein I/O, kein Zufall, keine Uhr.
"""

from __future__ import annotations

from typing import Any

import pytest

from trading_harness.models import (
    AgentSignal,
    MarketSnapshot,
    PortfolioState,
    RiskDecision,
    SignalAggregation,
)
from trading_harness.services.shadow_decision import (
    aggregate_signals,
    build_trade_proposal,
    no_trade_reason,
)


def _signal(direction: str, confidence: float, agent_id: str = "agent-1") -> AgentSignal:
    """Minimal gültiges AgentSignal für Aggregations-Tests."""
    return AgentSignal(
        run_id="run-1",
        agent_id=agent_id,
        snapshot_id="snap-1",
        category="technical",
        direction=direction,
        confidence=confidence,
        reasoning="r",
    )


def _aggregation(direction: str = "LONG", confidence: float = 0.8) -> SignalAggregation:
    return SignalAggregation(
        direction=direction,
        confidence=confidence,
        signal_count=2,
        no_trade_count=0,
        agent_ids=["agent-1", "agent-2"],
    )


def _snapshot(
    symbol: str = "BTC/USDT",
    snapshot_id: str = "snap-d",
    data: dict[str, Any] | None = None,
) -> MarketSnapshot:
    return MarketSnapshot(
        id=snapshot_id,
        symbol=symbol,
        data=data if data is not None else {"ticker": 100.0},
    )


def _portfolio(equity: float = 100000.0, positions: dict[str, float] | None = None) -> PortfolioState:
    return PortfolioState(
        run_id="run-1",
        current_equity=equity,
        positions=positions if positions is not None else {},
    )


def _risk_decision(
    max_position_size: float = 10000.0, risk_fraction: float = 0.005
) -> RiskDecision:
    # Nur max_position_size und risk_fraction werden von build_trade_proposal
    # gelesen; die übrigen Felder sind hier plausible Test-Dummies.
    return RiskDecision(
        approved=True,
        reason="OK",
        max_position_size=max_position_size,
        risk_amount=100000.0 * risk_fraction,
        risk_fraction=risk_fraction,
        risk_reward=2.0,
    )


# ---------------------------------------------------------------------------
# aggregate_signals
# ---------------------------------------------------------------------------


def test_aggregate_signals_majority_long() -> None:
    """ST.6 (a): 2 LONG + 1 SHORT -> LONG, Confidence = Mittel der LONGs."""
    signals = [_signal("LONG", 0.8, "a-long-1"), _signal("SHORT", 0.9, "a-short-1"), _signal("LONG", 0.7, "a-long-2")]
    agg = aggregate_signals(signals, "BTC/USDT")
    assert agg.direction == "LONG"
    assert agg.confidence == pytest.approx((0.8 + 0.7) / 2.0)
    assert agg.signal_count == 3
    assert agg.no_trade_count == 0


def test_aggregate_signals_majority_short() -> None:
    """1 LONG + 2 SHORT -> SHORT, Confidence = Mittel der SHORTs."""
    signals = [_signal("LONG", 0.9), _signal("SHORT", 0.7, "a-short-1"), _signal("SHORT", 0.8, "a-short-2")]
    agg = aggregate_signals(signals, "BTC/USDT")
    assert agg.direction == "SHORT"
    assert agg.confidence == pytest.approx((0.7 + 0.8) / 2.0)
    assert agg.signal_count == 3
    assert agg.no_trade_count == 0


def test_aggregate_signals_tie_no_trade() -> None:
    """ST.6 (b): 1 LONG + 1 SHORT (Gleichstand) -> NO_TRADE, Confidence 0.0."""
    signals = [_signal("LONG", 0.9), _signal("SHORT", 0.8)]
    agg = aggregate_signals(signals, "BTC/USDT")
    assert agg.direction == "NO_TRADE"
    assert agg.confidence == 0.0
    assert agg.signal_count == 2
    assert agg.no_trade_count == 0


def test_aggregate_signals_all_no_trade() -> None:
    """Alle NO_TRADE -> NO_TRADE, signal_count=0, no_trade_count=3."""
    signals = [_signal("NO_TRADE", 0.5, "a-1"), _signal("NO_TRADE", 0.4, "a-2"), _signal("NO_TRADE", 0.6, "a-3")]
    agg = aggregate_signals(signals, "BTC/USDT")
    assert agg.direction == "NO_TRADE"
    assert agg.confidence == 0.0
    assert agg.signal_count == 0
    assert agg.no_trade_count == 3


def test_aggregate_signals_empty_input_no_trade() -> None:
    """Z4: leere Eingabe = keine qualifizierenden Agenten -> NO_TRADE."""
    agg = aggregate_signals([], "BTC/USDT")
    assert agg.direction == "NO_TRADE"
    assert agg.confidence == 0.0
    assert agg.signal_count == 0
    assert agg.no_trade_count == 0
    assert agg.agent_ids == []


def test_aggregate_signals_confidence_is_direction_mean() -> None:
    """Confidence ist exakt das arithmetische Mittel der gewählten Richtung."""
    signals = [_signal("LONG", 0.9), _signal("LONG", 0.7, "a-2"), _signal("SHORT", 0.99, "a-3")]
    agg = aggregate_signals(signals, "BTC/USDT")
    assert agg.direction == "LONG"
    # Der SHORT-Wert (0.99) darf die Confidence nicht beeinflussen.
    assert agg.confidence == (0.9 + 0.7) / 2.0
    assert agg.confidence == pytest.approx(0.8)


def test_aggregate_signals_below_min_confidence_no_trade() -> None:
    """ST.6 (c): Mittel 0.525 < 0.6 -> NO_TRADE (Default und explizit)."""
    signals = [_signal("LONG", 0.5), _signal("LONG", 0.55, "a-2")]
    agg_default = aggregate_signals(signals, "BTC/USDT")
    assert agg_default.direction == "NO_TRADE"
    # Die berechnete Confidence bleibt als Information erhalten.
    assert agg_default.confidence == pytest.approx(0.525)

    agg_explicit = aggregate_signals(signals, "BTC/USDT", min_confidence=0.6)
    assert agg_explicit.direction == "NO_TRADE"

    # Niedrigerer Schwellwert: gleiche Signale bleiben LONG.
    agg_custom = aggregate_signals(signals, "BTC/USDT", min_confidence=0.5)
    assert agg_custom.direction == "LONG"
    assert agg_custom.confidence == pytest.approx(0.525)


def test_aggregate_signals_exactly_min_confidence_passes() -> None:
    """Grenze: Mittel exakt 0.6 mit Default-Schwelle 0.6 -> bleibt LONG (streng <)."""
    signals = [_signal("LONG", 0.6)]
    agg = aggregate_signals(signals, "BTC/USDT")
    assert agg.direction == "LONG"
    assert agg.confidence == 0.6


def test_aggregate_signals_deterministic() -> None:
    """ST.6 (e): gleiche Eingaben -> bit-identisches SignalAggregation."""
    def fresh_signals() -> list[AgentSignal]:
        return [_signal("LONG", 0.8, "a-1"), _signal("SHORT", 0.9, "a-2"), _signal("LONG", 0.7, "a-3")]

    first = aggregate_signals(fresh_signals(), "BTC/USDT")
    second = aggregate_signals(fresh_signals(), "BTC/USDT")
    assert first.model_dump() == second.model_dump()


def test_aggregate_signals_agent_ids_all_participants_in_order() -> None:
    """agent_ids enthält JEDES Eingabe-Signal in Eingabe-Reihenfolge (Audit)."""
    signals = [
        _signal("LONG", 0.8, "a-1"),
        _signal("NO_TRADE", 0.3, "a-2"),
        _signal("SHORT", 0.9, "a-3"),
        _signal("LONG", 0.7, "a-4"),
        _signal("NO_TRADE", 0.4, "a-5"),
    ]
    agg = aggregate_signals(signals, "BTC/USDT")
    assert agg.direction == "LONG"  # 2 LONG > 1 SHORT
    assert agg.agent_ids == ["a-1", "a-2", "a-3", "a-4", "a-5"]


# ---------------------------------------------------------------------------
# no_trade_reason
# ---------------------------------------------------------------------------


def test_no_trade_reasons() -> None:
    """E2: exakt zweiwertiges NO_TRADE-Gründervokabular."""
    # Keine qualifizierenden Agenten (Z4) -> NO_ACTIVE_CHAMPIONS
    assert no_trade_reason(aggregate_signals([], "BTC/USDT")) == "NO_ACTIVE_CHAMPIONS"
    # Alle NO_TRADE (signal_count == 0) ebenfalls -> NO_ACTIVE_CHAMPIONS
    assert no_trade_reason(aggregate_signals([_signal("NO_TRADE", 0.5)], "BTC/USDT")) == "NO_ACTIVE_CHAMPIONS"
    # Unterhalb des Confidence-Gates -> BELOW_MIN_CONFIDENCE
    below = aggregate_signals([_signal("LONG", 0.5), _signal("LONG", 0.55, "a-2")], "BTC/USDT")
    assert no_trade_reason(below) == "BELOW_MIN_CONFIDENCE"
    # Gleichstand (tie) -> BELOW_MIN_CONFIDENCE
    tie = aggregate_signals([_signal("LONG", 0.9), _signal("SHORT", 0.8)], "BTC/USDT")
    assert no_trade_reason(tie) == "BELOW_MIN_CONFIDENCE"
    # Freigebene Richtung -> None
    assert no_trade_reason(aggregate_signals([_signal("LONG", 0.9), _signal("LONG", 0.8, "a-2")], "BTC/USDT")) is None


# ---------------------------------------------------------------------------
# build_trade_proposal
# ---------------------------------------------------------------------------


def test_build_trade_proposal_caps_quantity() -> None:
    """ST.6 (d): rohe Quantity über max_position_size -> exakt auf max gedeckelt."""
    snapshot = _snapshot()
    # raw = 100000 * 0.01 / (100 * 0.02) = 500 > 100
    capped = build_trade_proposal(
        _aggregation(),
        snapshot,
        _portfolio(),
        _risk_decision(max_position_size=100.0, risk_fraction=0.01),
    )
    assert capped.requested_quantity == 100.0

    # raw = 100000 * 0.005 / 2.0 = 250 < 500 -> Rohwert bleibt erhalten
    uncapped = build_trade_proposal(
        _aggregation(),
        snapshot,
        _portfolio(),
        _risk_decision(max_position_size=500.0, risk_fraction=0.005),
    )
    assert uncapped.requested_quantity == 250.0


def test_build_trade_proposal_long_prices() -> None:
    """LONG: entry 100, slf 0.02, rrr 2.0 -> stop 98.0, target 104.0, side BUY."""
    proposal = build_trade_proposal(_aggregation(), _snapshot(), _portfolio(), _risk_decision())
    assert proposal.side == "BUY"
    assert proposal.entry_price == 100.0
    assert proposal.stop_price == 98.0
    assert proposal.target_price == 104.0


def test_build_trade_proposal_short_prices() -> None:
    """SHORT: entry 100, slf 0.02, rrr 2.0 -> stop 102.0, target 96.0, side SELL."""
    proposal = build_trade_proposal(_aggregation(direction="SHORT"), _snapshot(), _portfolio(), _risk_decision())
    assert proposal.side == "SELL"
    assert proposal.entry_price == 100.0
    assert proposal.stop_price == 102.0
    assert proposal.target_price == 96.0


def test_build_trade_proposal_quantity_formula() -> None:
    """equity 100000, risk_fraction 0.005, entry 100, slf 0.02 -> quantity 250.0."""
    proposal = build_trade_proposal(
        _aggregation(),
        _snapshot(),
        _portfolio(equity=100000.0),
        _risk_decision(max_position_size=10000.0, risk_fraction=0.005),
    )
    assert proposal.requested_quantity == 250.0
    assert proposal.equity == 100000.0
    assert proposal.open_positions == 0


def test_build_trade_proposal_rejects_no_trade_direction() -> None:
    """NO_TRADE-Aggregation -> ValueError (Loop ruft nur freigegebene Trades)."""
    no_trade = _aggregation(direction="NO_TRADE", confidence=0.0)
    with pytest.raises(ValueError):
        build_trade_proposal(no_trade, _snapshot(), _portfolio(), _risk_decision())


def test_build_trade_proposal_missing_ticker_raises() -> None:
    """Fehlender, nicht-numerischer oder nicht-positiver Ticker -> ValueError."""
    for bad_data in ({"price": 100.0}, {"ticker": "abc"}, {"ticker": None}, {"ticker": -5.0}):
        with pytest.raises(ValueError):
            build_trade_proposal(_aggregation(), _snapshot(data=bad_data), _portfolio(), _risk_decision())


def test_build_trade_proposal_deterministic() -> None:
    """ST.14: gleiche Eingaben -> bit-identisches TradeProposal, stabiles decision_id."""
    aggregation = _aggregation()
    snapshot = _snapshot()
    portfolio_state = _portfolio()
    risk_decision = _risk_decision()
    first = build_trade_proposal(aggregation, snapshot, portfolio_state, risk_decision)
    second = build_trade_proposal(aggregation, snapshot, portfolio_state, risk_decision)
    assert first.model_dump() == second.model_dump()
    assert first.decision_id == "shadow-dec-snap-d-BTC/USDT"


def test_build_trade_proposal_rejects_non_positive_equity() -> None:
    """equity <= 0 -> ValueError (TradeProposal verlangt gt=0)."""
    for equity in (0.0, -1000.0):
        with pytest.raises(ValueError):
            build_trade_proposal(_aggregation(), _snapshot(), _portfolio(equity=equity), _risk_decision())


def test_build_trade_proposal_rejects_zero_max_position_size() -> None:
    """max_position_size 0 -> gedeckelte Quantity 0 <= 0 -> ValueError (defensiv)."""
    with pytest.raises(ValueError):
        build_trade_proposal(_aggregation(), _snapshot(), _portfolio(), _risk_decision(max_position_size=0.0))
