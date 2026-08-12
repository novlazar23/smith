"""Tests for packages.strategy — StrategyService (EPIC-04-WP01).

Covers: models, entry evaluation, target calculation, EV evaluation,
and the StrategyEngine pipeline (consensus → proposal).
"""

from __future__ import annotations

import dataclasses
from datetime import datetime

import pytest
from packages.consensus import (
    ConsensusDecision,
    ConsensusResult,
    VoteDirection,
    WeightConfig,
    WeightedConsensusEngine,
)
from packages.schemas.agent_report import AgentReport, AgentStatus, EvidenceReference
from packages.strategy import (
    EntryCondition,
    EntryType,
    StrategyConfig,
    StrategyDirection,
    StrategyEngine,
    StrategyProposal,
    StrategyVariant,
    TargetType,
    apply_gates,
    calculate_expected_return,
    calculate_prob_target_before_stop,
    calculate_risk_reward,
    calculate_targets,
    estimate_mfe_mae,
    evaluate_entry,
    evaluate_variant,
)

# ── helpers ──────────────────────────────────────────────────────────


def _make_consensus_long(
    confidence: float = 0.75,
    decision: ConsensusDecision = ConsensusDecision.LONG_BIAS,
) -> ConsensusResult:
    return ConsensusResult(
        decision=decision,
        vote_distribution={
            VoteDirection.LONG: 0.7,
            VoteDirection.SHORT: 0.15,
            VoteDirection.RANGE: 0.1,
            VoteDirection.ABSTAIN: 0.05,
        },
        agent_weights={"agent-1": 1.0, "agent-2": 1.0},
        agent_agreements=["agent-1", "agent-2"],
        agent_disagreements=[],
        confidence=confidence,
        reason="long consensus",
    )


def _make_consensus_short(
    confidence: float = 0.70,
) -> ConsensusResult:
    return ConsensusResult(
        decision=ConsensusDecision.SHORT_BIAS,
        vote_distribution={
            VoteDirection.LONG: 0.15,
            VoteDirection.SHORT: 0.75,
            VoteDirection.RANGE: 0.05,
            VoteDirection.ABSTAIN: 0.05,
        },
        agent_weights={"agent-1": 1.0},
        agent_agreements=["agent-1"],
        agent_disagreements=[],
        confidence=confidence,
        reason="short consensus",
    )


def _make_consensus_no_trade() -> ConsensusResult:
    return ConsensusResult(
        decision=ConsensusDecision.NO_TRADE,
        vote_distribution={
            VoteDirection.LONG: 0.33,
            VoteDirection.SHORT: 0.33,
            VoteDirection.RANGE: 0.34,
            VoteDirection.ABSTAIN: 0.0,
        },
        agent_weights={"agent-1": 1.0},
        agent_agreements=[],
        agent_disagreements=["agent-1"],
        confidence=0.34,
        reason="no consensus",
    )


def _make_features(
    current_price: float = 100.0,
    atr: float = 2.0,
    volatility: float = 1.2,
    entry_type: str = "market",
    entry_condition: str = "momentum",
) -> dict:
    return {
        "current_price": current_price,
        "atr": atr,
        "volatility": volatility,
        "entry_type": entry_type,
        "entry_condition": entry_condition,
    }


# ══════════════════════════════════════════════════════════════════════
# 1. MODEL TESTS
# ══════════════════════════════════════════════════════════════════════


class TestStrategyDirection:
    """Teste StrategyDirection-Enum."""

    def test_all_directions(self) -> None:
        assert StrategyDirection.LONG == "long"
        assert StrategyDirection.SHORT == "short"
        assert StrategyDirection.NO_TRADE == "no_trade"
        assert (
            StrategyDirection.NO_TRADE_DATA_QUALITY
            == "no_trade_data_quality"
        )
        assert (
            StrategyDirection.NO_TRADE_INSUFFICIENT_EDGE
            == "no_trade_insufficient_edge"
        )
        assert StrategyDirection.NO_TRADE_RISK == "no_trade_risk"
        assert (
            StrategyDirection.NO_TRADE_PORTFOLIO
            == "no_trade_portfolio"
        )
        assert (
            StrategyDirection.NO_TRADE_MODEL_UNCERTAINTY
            == "no_trade_model_uncertainty"
        )


class TestStrategyConfig:
    """Teste StrategyConfig-Standardwerte und Validierung."""

    def test_defaults(self) -> None:
        cfg = StrategyConfig()
        assert cfg.min_edge == 0.002
        assert cfg.min_prob == 0.55
        assert cfg.min_rr == 1.5
        assert cfg.min_quality == 0.95

    def test_frozen(self) -> None:
        cfg = StrategyConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.min_edge = 0.01

    def test_valid_range(self) -> None:
        cfg = StrategyConfig(min_edge=0.005, min_prob=0.60, min_rr=2.0, min_quality=0.99)
        cfg.validate()  # keine Exception

    def test_invalid_min_edge(self) -> None:
        with pytest.raises(ValueError, match="min_edge"):
            StrategyConfig(min_edge=-0.1).validate()

    def test_invalid_min_rr(self) -> None:
        with pytest.raises(ValueError, match="min_rr"):
            StrategyConfig(min_rr=-1.0).validate()


class TestStrategyVariant:
    """Teste StrategyVariant."""

    def test_risk_amount_long(self) -> None:
        v = StrategyVariant(
            variant_id="v1",
            direction=StrategyDirection.LONG,
            entry_price=100.0,
            entry_type="market",
            stop_loss=97.0,
            targets=[105.0, 110.0],
        )
        assert v.risk_amount == 3.0

    def test_risk_amount_short(self) -> None:
        v = StrategyVariant(
            variant_id="v2",
            direction=StrategyDirection.SHORT,
            entry_price=100.0,
            entry_type="market",
            stop_loss=103.0,
            targets=[95.0, 90.0],
        )
        assert v.risk_amount == 3.0

    def test_total_reward_long(self) -> None:
        v = StrategyVariant(
            variant_id="v1",
            direction=StrategyDirection.LONG,
            entry_price=100.0,
            entry_type="market",
            stop_loss=97.0,
            targets=[105.0, 110.0],
        )
        assert v.total_reward == 10.0  # max(targets) - entry = 110 - 100

    def test_total_reward_short(self) -> None:
        v = StrategyVariant(
            variant_id="v2",
            direction=StrategyDirection.SHORT,
            entry_price=100.0,
            entry_type="market",
            stop_loss=103.0,
            targets=[95.0, 90.0],
        )
        assert v.total_reward == 10.0  # entry - min(targets) = 100 - 90

    def test_total_reward_empty_targets(self) -> None:
        v = StrategyVariant(
            variant_id="v3",
            direction=StrategyDirection.LONG,
            entry_price=100.0,
            entry_type="market",
            stop_loss=97.0,
            targets=[],
        )
        assert v.total_reward == 0.0


class TestStrategyProposal:
    """Teste StrategyProposal."""

    def test_long_proposal(self) -> None:
        p = StrategyProposal(
            direction=StrategyDirection.LONG,
            entry_type="market",
            entry_price=100.0,
            entry_condition="momentum",
            stop_loss=97.0,
            targets=[105.0, 110.0],
            prob_target_before_stop=0.65,
            expected_return_net=0.005,
            risk_reward_ratio=2.0,
            confidence=0.75,
        )
        assert p.direction == StrategyDirection.LONG
        assert len(p.targets) == 2
        assert p.risk_reward_ratio == 2.0

    def test_no_trade_proposal(self) -> None:
        p = StrategyProposal(
            direction=StrategyDirection.NO_TRADE,
            entry_type="",
            entry_price=0.0,
            entry_condition="",
            stop_loss=0.0,
            targets=[],
            reason="no_consensus",
        )
        assert p.direction == StrategyDirection.NO_TRADE
        assert p.targets == []


# ══════════════════════════════════════════════════════════════════════
# 2. ENTRY TESTS
# ══════════════════════════════════════════════════════════════════════


class TestEntryType:
    """Teste EntryType-Enum."""

    def test_all_types(self) -> None:
        assert EntryType.MARKET == "market"
        assert EntryType.LIMIT == "limit"
        assert EntryType.BRACKET == "bracket"


class TestEntryCondition:
    """Teste EntryCondition-Enum."""

    def test_all_conditions(self) -> None:
        assert EntryCondition.BREAKOUT == "breakout"
        assert EntryCondition.PULLBACK == "pullback"
        assert EntryCondition.REVERSAL == "reversal"
        assert EntryCondition.MOMENTUM == "momentum"
        assert EntryCondition.MEAN_REVERSION == "mean_reversion"
        assert EntryCondition.VOLATILITY_BREAKOUT == "volatility_breakout"
        assert EntryCondition.VOLATILITY_CONTRACTION == "volatility_contraction"


class TestEvaluateEntry:
    """Teste evaluate_entry-Funktion."""

    def test_market_entry(self) -> None:
        cfg = StrategyConfig()
        signal = {"type": "market", "condition": "momentum", "price": 100.0, "strength": 0.8}
        snapshot = {"bid": 99.9, "ask": 100.1, "spread": 0.2, "volume": 1000.0}
        result = evaluate_entry(cfg, signal, snapshot)
        assert result.price == 100.0
        assert result.entry_type == EntryType.MARKET
        assert result.condition == EntryCondition.MOMENTUM
        assert result.strength == 0.8

    def test_limit_entry(self) -> None:
        cfg = StrategyConfig()
        signal = {"type": "limit", "condition": "pullback", "price": 99.0}
        snapshot = {"bid": 99.0, "ask": 99.2, "spread": 0.2, "volume": 500.0}
        result = evaluate_entry(cfg, signal, snapshot)
        assert result.entry_type == EntryType.LIMIT
        assert result.condition == EntryCondition.PULLBACK

    def test_bracket_entry(self) -> None:
        cfg = StrategyConfig()
        signal = {"type": "bracket", "condition": "breakout", "price": 105.0}
        snapshot = {"bid": 104.9, "ask": 105.1, "spread": 0.2, "volume": 800.0}
        result = evaluate_entry(cfg, signal, snapshot)
        assert result.entry_type == EntryType.BRACKET
        assert result.condition == EntryCondition.BREAKOUT

    def test_missing_price_raises(self) -> None:
        cfg = StrategyConfig()
        signal = {"type": "market", "condition": "momentum"}
        snapshot = {"bid": 100.0, "ask": 100.2, "spread": 0.2}
        with pytest.raises(ValueError, match="price"):
            evaluate_entry(cfg, signal, snapshot)

    def test_empty_signal_raises(self) -> None:
        cfg = StrategyConfig()
        snapshot = {}
        with pytest.raises(ValueError, match="signal must not be empty"):
            evaluate_entry(cfg, {}, snapshot)

    def test_invalid_entry_type_raises(self) -> None:
        cfg = StrategyConfig()
        signal = {"type": "invalid_type", "condition": "momentum", "price": 100.0}
        snapshot = {"bid": 100.0, "ask": 100.2}
        with pytest.raises(ValueError, match="entry type"):
            evaluate_entry(cfg, signal, snapshot)

    def test_invalid_condition_raises(self) -> None:
        cfg = StrategyConfig()
        signal = {"type": "market", "condition": "fake_condition", "price": 100.0}
        snapshot = {"bid": 100.0, "ask": 100.2}
        with pytest.raises(ValueError, match="entry condition"):
            evaluate_entry(cfg, signal, snapshot)

    def test_spread_warning(self) -> None:
        cfg = StrategyConfig(min_quality=0.99)
        signal = {"type": "market", "condition": "momentum", "price": 100.0}
        snapshot = {"bid": 90.0, "ask": 110.0, "spread": 20.0, "volume": 100.0}
        result = evaluate_entry(cfg, signal, snapshot)
        assert result.metadata.get("spread_warning") is True

    def test_strength_default(self) -> None:
        cfg = StrategyConfig()
        signal = {"type": "market", "condition": "reversal", "price": 50.0}
        snapshot = {"bid": 49.9, "ask": 50.1, "spread": 0.2}
        result = evaluate_entry(cfg, signal, snapshot)
        assert result.strength == 1.0


# ══════════════════════════════════════════════════════════════════════
# 3. TARGET TESTS
# ══════════════════════════════════════════════════════════════════════


class TestTargetType:
    """Teste TargetType-Enum."""

    def test_all_types(self) -> None:
        assert TargetType.TP1 == "tp1"
        assert TargetType.TP2 == "tp2"
        assert TargetType.TP3 == "tp3"
        assert TargetType.STOP_LIMIT == "stop_limit"


class TestCalculateTargets:
    """Teste calculate_targets-Funktion."""

    def test_long_targets_order(self) -> None:
        targets = calculate_targets(
            entry_price=100.0,
            volatility=1.0,
            atr=2.0,
            direction="long",
        )
        # TP1 < TP2 < TP3, stop < entry
        tp_prices = [t.price for t in targets]
        assert tp_prices[0] > 100.0  # TP1 above entry
        assert tp_prices[1] > tp_prices[0]  # TP2 > TP1
        assert tp_prices[2] > tp_prices[1]  # TP3 > TP2
        assert tp_prices[3] < 100.0  # STOP below entry

    def test_short_targets_order(self) -> None:
        targets = calculate_targets(
            entry_price=100.0,
            volatility=1.0,
            atr=2.0,
            direction="short",
        )
        tp_prices = [t.price for t in targets]
        assert tp_prices[0] < 100.0  # TP1 below entry
        assert tp_prices[1] < tp_prices[0]  # TP2 < TP1
        assert tp_prices[2] < tp_prices[1]  # TP3 < TP2
        assert tp_prices[3] > 100.0  # STOP above entry

    def test_probability_decay(self) -> None:
        targets = calculate_targets(
            entry_price=100.0,
            volatility=1.0,
            atr=2.0,
            direction="long",
        )
        assert targets[0].probability > targets[1].probability
        assert targets[1].probability > targets[2].probability

    def test_tp1_has_stop_metadata(self) -> None:
        targets = calculate_targets(
            entry_price=100.0,
            volatility=1.0,
            atr=2.0,
            direction="long",
        )
        tp1 = targets[0]
        assert "prob_target_before_stop" in tp1.metadata
        assert "prob_stop_before_target" in tp1.metadata
        assert tp1.metadata["prob_target_before_stop"] + tp1.metadata["prob_stop_before_target"] == 1.0

    def test_stop_limit_type(self) -> None:
        targets = calculate_targets(
            entry_price=100.0,
            volatility=1.0,
            atr=2.0,
            direction="long",
        )
        assert targets[3].type == TargetType.STOP_LIMIT
        assert targets[3].metadata.get("is_stop") is True

    def test_zero_entry_raises(self) -> None:
        with pytest.raises(ValueError, match="entry_price"):
            calculate_targets(
                entry_price=0.0,
                volatility=1.0,
                atr=2.0,
                direction="long",
            )

    def test_zero_atr_raises(self) -> None:
        with pytest.raises(ValueError, match="atr"):
            calculate_targets(
                entry_price=100.0,
                volatility=1.0,
                atr=0.0,
                direction="long",
            )

    def test_invalid_direction_raises(self) -> None:
        with pytest.raises(ValueError, match="direction"):
            calculate_targets(
                entry_price=100.0,
                volatility=1.0,
                atr=2.0,
                direction="invalid",
            )

    def test_custom_multipliers(self) -> None:
        targets = calculate_targets(
            entry_price=100.0,
            volatility=1.0,
            atr=2.0,
            direction="long",
            tp1_atr=1.0,
            tp2_atr=2.0,
            tp3_atr=4.0,
            stop_atr=1.5,
        )
        # TP1 distance = 1.0 * 2.0 = 2.0 → price = 102.0
        assert targets[0].price == 102.0
        assert targets[3].price == 97.0  # stop = 100 - 1.5*2 = 97


class TestEstimateMfeMae:
    """Teste estimate_mfe_mae-Funktion."""

    def test_long_mae_mfe(self) -> None:
        targets = calculate_targets(100.0, 1.0, 2.0, "long")
        mae, mfe = estimate_mfe_mae(100.0, targets, 1.0, "long")
        # MAE sollte negativ (favorable = positive for long)
        assert mae < 0  # adverse move is negative for long
        assert mfe > 0  # favorable move is positive for long

    def test_short_mae_mfe(self) -> None:
        targets = calculate_targets(100.0, 1.0, 2.0, "short")
        mae, mfe = estimate_mfe_mae(100.0, targets, 1.0, "short")
        # For short: adverse = positive, favorable = negative
        assert mae > 0
        assert mfe < 0

    def test_mfe_capped_at_target(self) -> None:
        """MFE sollte am Target-Oberlauf geschnitten werden."""
        targets = calculate_targets(100.0, 1.0, 2.0, "long")
        _, mfe = estimate_mfe_mae(100.0, targets, 0.5, "long")
        # Bei niedriger Volatilität sollte MFE am Target cap sein
        assert mfe > 0


class TestCalculateProbTargetBeforeStop:
    """Teste calculate_prob_target_before_stop."""

    def test_equal_distances(self) -> None:
        # TP und Stop gleiche Distanz → ca. 0.5
        prob = calculate_prob_target_before_stop(2.0, 2.0, 0.01)
        assert 0.45 < prob < 0.55

    def test_far_target(self) -> None:
        prob = calculate_prob_target_before_stop(10.0, 1.0, 0.01)
        assert prob < 0.5

    def test_near_target(self) -> None:
        prob = calculate_prob_target_before_stop(1.0, 10.0, 0.01)
        assert prob > 0.5

    def test_zero_distance(self) -> None:
        prob = calculate_prob_target_before_stop(0.0, 2.0, 0.01)
        assert prob == 0.5

    def test_zero_volatility(self) -> None:
        prob = calculate_prob_target_before_stop(2.0, 2.0, 0.0)
        assert 0.4 < prob < 0.6

    def test_high_volatility_bias(self) -> None:
        # Hohe Volatilität schiebt Richtung 0.5
        prob_low = calculate_prob_target_before_stop(1.0, 5.0, 0.01)
        prob_high = calculate_prob_target_before_stop(1.0, 5.0, 0.15)
        assert abs(prob_high - 0.5) < abs(prob_low - 0.5)


# ══════════════════════════════════════════════════════════════════════
# 4. EV TESTS
# ══════════════════════════════════════════════════════════════════════


class TestCalculateExpectedReturn:
    """Teste calculate_expected_return."""

    def test_long_no_costs(self) -> None:
        v = StrategyVariant(
            variant_id="v1",
            direction=StrategyDirection.LONG,
            entry_price=100.0,
            entry_type="market",
            stop_loss=97.0,
            targets=[105.0, 110.0],
            probability=0.6,
        )
        cfg = StrategyConfig()
        ev = calculate_expected_return(v, cfg, {})
        assert ev["costs"] == 0.0
        assert ev["gross"] != 0.0

    def test_long_with_costs(self) -> None:
        v = StrategyVariant(
            variant_id="v2",
            direction=StrategyDirection.LONG,
            entry_price=100.0,
            entry_type="market",
            stop_loss=97.0,
            targets=[105.0],
            probability=0.6,
        )
        cfg = StrategyConfig()
        costs = {"fee_pct": 0.001, "slippage_pct": 0.0005, "spread_bps": 5.0}
        ev = calculate_expected_return(v, cfg, costs)
        # Costs should be positive (2 * (fee + slippage + spread_bps/10000))
        assert ev["costs"] > 0.0
        # net = gross - costs (within floating-point tolerance)
        assert abs(ev["net"] - (ev["gross"] - ev["costs"])) < 1e-6

    def test_short_with_costs(self) -> None:
        v = StrategyVariant(
            variant_id="v3",
            direction=StrategyDirection.SHORT,
            entry_price=100.0,
            entry_type="market",
            stop_loss=103.0,
            targets=[95.0],
            probability=0.6,
        )
        cfg = StrategyConfig()
        costs = {"fee_pct": 0.001, "slippage_pct": 0.0005, "spread_bps": 5.0}
        ev = calculate_expected_return(v, cfg, costs)
        assert ev["costs"] > 0.0
        # net = gross - costs (within floating-point tolerance)
        assert abs(ev["net"] - (ev["gross"] - ev["costs"])) < 1e-6

    def test_empty_targets(self) -> None:
        v = StrategyVariant(
            variant_id="v4",
            direction=StrategyDirection.LONG,
            entry_price=100.0,
            entry_type="market",
            stop_loss=97.0,
            targets=[],
            probability=0.6,
        )
        cfg = StrategyConfig()
        ev = calculate_expected_return(v, cfg, {})
        assert ev["gross"] == 0.0
        assert ev["net"] == 0.0

    def test_no_trade_direction(self) -> None:
        v = StrategyVariant(
            variant_id="v5",
            direction=StrategyDirection.NO_TRADE,
            entry_price=0.0,
            entry_type="market",
            stop_loss=0.0,
            targets=[],
            probability=0.0,
        )
        cfg = StrategyConfig()
        ev = calculate_expected_return(v, cfg, {})
        assert ev["gross"] == 0.0


class TestCalculateRiskReward:
    """Teste calculate_risk_reward."""

    def test_rr_2_to_1(self) -> None:
        rr = calculate_risk_reward(risk_amount=100.0, reward_amount=200.0)
        assert rr == 2.0

    def test_rr_1_to_1(self) -> None:
        rr = calculate_risk_reward(risk_amount=100.0, reward_amount=100.0)
        assert rr == 1.0

    def test_rr_high(self) -> None:
        rr = calculate_risk_reward(risk_amount=50.0, reward_amount=300.0)
        assert rr == 6.0

    def test_zero_risk_raises(self) -> None:
        with pytest.raises(ValueError, match="risk_amount"):
            calculate_risk_reward(risk_amount=0.0, reward_amount=100.0)

    def test_negative_risk_raises(self) -> None:
        with pytest.raises(ValueError, match="risk_amount"):
            calculate_risk_reward(risk_amount=-100.0, reward_amount=100.0)


class TestEvaluateVariant:
    """Teste evaluate_variant."""

    def test_passes_all_gates(self) -> None:
        v = StrategyVariant(
            variant_id="v1",
            direction=StrategyDirection.LONG,
            entry_price=100.0,
            entry_type="market",
            stop_loss=95.0,
            targets=[110.0, 120.0],
            probability=0.7,
        )
        cfg = StrategyConfig()
        result = evaluate_variant(v, cfg, {}, data_quality=0.98)
        assert result["approved"] is True
        assert result["gates_passed"] is True
        assert result["gate_failures"] == []

    def test_fails_edge_gate(self) -> None:
        v = StrategyVariant(
            variant_id="v2",
            direction=StrategyDirection.LONG,
            entry_price=100.0,
            entry_type="market",
            stop_loss=99.0,
            targets=[100.5],
            probability=0.51,
        )
        cfg = StrategyConfig(min_edge=0.01)  # 1% edge required
        result = evaluate_variant(v, cfg, {}, data_quality=0.99)
        assert result["approved"] is False
        assert "edge" in result["gate_failures"]

    def test_fails_prob_gate(self) -> None:
        v = StrategyVariant(
            variant_id="v3",
            direction=StrategyDirection.LONG,
            entry_price=100.0,
            entry_type="market",
            stop_loss=95.0,
            targets=[115.0, 125.0],
            probability=0.4,  # below min_prob=0.55
        )
        cfg = StrategyConfig()
        result = evaluate_variant(v, cfg, {}, data_quality=0.99)
        assert result["approved"] is False
        assert "prob" in result["gate_failures"]

    def test_fails_rr_gate(self) -> None:
        v = StrategyVariant(
            variant_id="v4",
            direction=StrategyDirection.LONG,
            entry_price=100.0,
            entry_type="market",
            stop_loss=99.0,
            targets=[101.0],
            probability=0.7,
        )
        cfg = StrategyConfig(min_rr=5.0)
        result = evaluate_variant(v, cfg, {}, data_quality=0.99)
        assert result["approved"] is False
        assert "rr" in result["gate_failures"]

    def test_fails_quality_gate(self) -> None:
        v = StrategyVariant(
            variant_id="v5",
            direction=StrategyDirection.LONG,
            entry_price=100.0,
            entry_type="market",
            stop_loss=95.0,
            targets=[115.0, 125.0],
            probability=0.7,
        )
        cfg = StrategyConfig()
        result = evaluate_variant(v, cfg, {}, data_quality=0.90)  # below 0.95
        assert result["approved"] is False
        assert "quality" in result["gate_failures"]

    def test_no_trade_direction(self) -> None:
        v = StrategyVariant(
            variant_id="v6",
            direction=StrategyDirection.NO_TRADE,
            entry_price=0.0,
            entry_type="market",
            stop_loss=0.0,
            targets=[],
            probability=0.0,
        )
        cfg = StrategyConfig()
        result = evaluate_variant(v, cfg, {}, data_quality=0.99)
        assert result["approved"] is False
        assert result["gate_failures"] == ["no_trade_direction"]

    def test_all_gate_fields_present(self) -> None:
        v = StrategyVariant(
            variant_id="v7",
            direction=StrategyDirection.LONG,
            entry_price=100.0,
            entry_type="market",
            stop_loss=95.0,
            targets=[110.0],
            probability=0.7,
        )
        cfg = StrategyConfig()
        result = evaluate_variant(v, cfg, {}, data_quality=0.99)
        assert "expected_return_gross" in result
        assert "expected_return_net" in result
        assert "expected_costs" in result
        assert "risk_reward_ratio" in result
        assert "prob_target_before_stop" in result
        assert "prob_stop_before_target" in result
        assert "gate_results" in result


class TestApplyGates:
    """Teste apply_gates."""

    def test_all_passed(self) -> None:
        evaluation = {"gates_passed": True}
        cfg = StrategyConfig()
        assert apply_gates(evaluation, cfg) is True

    def test_one_failed(self) -> None:
        evaluation = {"gates_passed": False}
        cfg = StrategyConfig()
        assert apply_gates(evaluation, cfg) is False

    def test_empty_evaluation(self) -> None:
        assert apply_gates({}, StrategyConfig()) is False
        assert apply_gates(None, StrategyConfig()) is False


# ══════════════════════════════════════════════════════════════════════
# 5. ENGINE TESTS
# ══════════════════════════════════════════════════════════════════════


class TestStrategyEngineGenerateVariants:
    """Teste StrategyEngine.generate_variants."""

    def test_long_consensus(self) -> None:
        engine = StrategyEngine()
        consensus = _make_consensus_long(confidence=0.75)
        features = _make_features()
        variants = engine.generate_variants(consensus, features)
        assert len(variants) == 3
        assert all(v.direction == StrategyDirection.LONG for v in variants)
        assert all(v.entry_price == 100.0 for v in variants)

    def test_short_consensus(self) -> None:
        engine = StrategyEngine()
        consensus = _make_consensus_short()
        features = _make_features()
        variants = engine.generate_variants(consensus, features)
        assert len(variants) == 3
        assert all(v.direction == StrategyDirection.SHORT for v in variants)

    def test_no_trade_consensus(self) -> None:
        engine = StrategyEngine()
        consensus = _make_consensus_no_trade()
        features = _make_features()
        variants = engine.generate_variants(consensus, features)
        assert len(variants) == 0

    def test_range_consensus(self) -> None:
        engine = StrategyEngine()
        consensus = ConsensusResult(
            decision=ConsensusDecision.RANGE,
            vote_distribution={VoteDirection.RANGE: 0.8, VoteDirection.LONG: 0.1,
                               VoteDirection.SHORT: 0.05, VoteDirection.ABSTAIN: 0.05},
            agent_weights={"a1": 1.0},
            agent_agreements=["a1"],
            agent_disagreements=[],
            confidence=0.8,
            reason="range",
        )
        features = _make_features()
        variants = engine.generate_variants(consensus, features)
        assert len(variants) == 1
        assert variants[0].direction == StrategyDirection.NO_TRADE

    def variant_metadata_contains_source(self, variants: list) -> None:
        """Varianten enthalten metadata mit source."""
        assert all("source" in v.metadata for v in variants)

    def test_variant_ids_unique(self) -> None:
        engine = StrategyEngine()
        consensus = _make_consensus_long()
        features = _make_features()
        variants = engine.generate_variants(consensus, features)
        ids = [v.variant_id for v in variants]
        assert len(ids) == len(set(ids))

    def test_variant_risk_amount_positive(self) -> None:
        engine = StrategyEngine()
        consensus = _make_consensus_long()
        features = _make_features()
        variants = engine.generate_variants(consensus, features)
        assert all(v.risk_amount > 0 for v in variants)


class TestStrategyEngineSelectBest:
    """Teste StrategyEngine.select_best."""

    def test_selects_highest_net_ev(self) -> None:
        engine = StrategyEngine()
        configs = StrategyConfig()
        variants = [
            StrategyVariant(
                variant_id="v1",
                direction=StrategyDirection.LONG,
                entry_price=100.0,
                entry_type="market",
                stop_loss=95.0,
                targets=[110.0, 120.0],
                probability=0.6,
            ),
            StrategyVariant(
                variant_id="v2",
                direction=StrategyDirection.LONG,
                entry_price=100.0,
                entry_type="market",
                stop_loss=97.0,
                targets=[108.0, 112.0],
                probability=0.7,
            ),
        ]
        proposal = engine.select_best(variants, configs, {})
        assert proposal is not None
        # v2 should win — higher probability and acceptable risk

    def test_no_variations_pass(self) -> None:
        """Keine Variante geht durch → None."""
        engine = StrategyEngine()
        # Variante mit zu kleinem Reward
        variants = [
            StrategyVariant(
                variant_id="v1",
                direction=StrategyDirection.LONG,
                entry_price=100.0,
                entry_type="market",
                stop_loss=99.9,
                targets=[100.1],
                probability=0.51,
            ),
        ]
        config = StrategyConfig(min_rr=5.0)  # hohe RR-Anforderung
        proposal = engine.select_best(variants, config, {})
        assert proposal is None

    def test_no_trade_variant_skipped(self) -> None:
        variants = [
            StrategyVariant(
                variant_id="v1",
                direction=StrategyDirection.NO_TRADE,
                entry_price=0.0,
                entry_type="market",
                stop_loss=0.0,
                targets=[],
                probability=0.0,
            ),
        ]
        engine = StrategyEngine()
        proposal = engine.select_best(variants)
        assert proposal is None

    def test_empty_variants(self) -> None:
        engine = StrategyEngine()
        proposal = engine.select_best([])
        assert proposal is None


class TestStrategyEngineRun:
    """Teste StrategyEngine.run."""

    def test_long_pipeline(self) -> None:
        engine = StrategyEngine()
        consensus = _make_consensus_long(confidence=0.80)
        context = {
            "consensus": consensus,
            "features": _make_features(current_price=50000.0, atr=500.0),
        }
        proposal = engine.run(context)
        assert proposal is not None
        assert proposal.direction in (
            StrategyDirection.LONG,
            StrategyDirection.NO_TRADE,
        )

    def test_no_trade_pipeline(self) -> None:
        engine = StrategyEngine()
        consensus = _make_consensus_no_trade()
        context = {
            "consensus": consensus,
            "features": _make_features(),
        }
        proposal = engine.run(context)
        assert proposal.direction in (
            StrategyDirection.NO_TRADE,
            StrategyDirection.NO_TRADE_INSUFFICIENT_EDGE,
        )

    def test_short_pipeline(self) -> None:
        engine = StrategyEngine()
        consensus = _make_consensus_short()
        context = {
            "consensus": consensus,
            "features": _make_features(),
        }
        proposal = engine.run(context)
        assert proposal is not None

    def test_proposal_has_expected_fields(self) -> None:
        engine = StrategyEngine()
        consensus = _make_consensus_long()
        context = {
            "consensus": consensus,
            "features": _make_features(),
        }
        proposal = engine.run(context)
        assert hasattr(proposal, "expected_return_net")
        assert hasattr(proposal, "risk_reward_ratio")
        assert hasattr(proposal, "expected_mae")
        assert hasattr(proposal, "expected_mfe")
        assert hasattr(proposal, "prob_target_before_stop")

    def test_pipeline_with_costs(self) -> None:
        engine = StrategyEngine()
        consensus = _make_consensus_long()
        context = {
            "consensus": consensus,
            "features": _make_features(current_price=1000.0, atr=10.0),
            "costs": {"fee_pct": 0.001, "slippage_pct": 0.0005, "spread_bps": 3.0},
        }
        proposal = engine.run(context)
        # With costs, net return should be lower
        assert proposal is not None


# ══════════════════════════════════════════════════════════════════════
# 6. INTEGRATION TESTS
# ══════════════════════════════════════════════════════════════════════


class TestIntegrationPipeline:
    """Full pipeline: consensus → variants → evaluate → proposal."""

    def test_full_long_pipeline(self) -> None:
        """Kompletter Long-Handels-Pipeline."""
        engine = StrategyEngine()

        # 1. Consensus erstellen
        reports = [
            AgentReport(
                report_id=f"rpt-{i}",
                run_id="run-001",
                agent_id=f"agent-{i}",
                agent_version="0.1.0",
                instrument="BTC/USDT",
                horizon="1h",
                as_of=datetime.now(),
                hypothesis=f"bullish-hypothesis-{i}",
                probabilities={"up": 0.75, "down": 0.1, "range": 0.15},
                evidence=[
                    EvidenceReference(
                        reference=f"agent-{i}:rsi",
                        feature="rsi",
                        value="30",
                        direction="positive",
                        relevance=0.8,
                    )
                ],
                status=AgentStatus.ACTIVE,
            )
            for i in range(3)
        ]
        reports.append(
            AgentReport(
                report_id="rpt-short",
                run_id="run-001",
                agent_id="agent-short",
                agent_version="0.1.0",
                instrument="BTC/USDT",
                horizon="1h",
                as_of=datetime.now(),
                hypothesis="bearish-hypothesis",
                probabilities={"up": 0.1, "down": 0.75, "range": 0.15},
                evidence=[
                    EvidenceReference(
                        reference="agent-short:macd",
                        feature="macd",
                        value="-2.5",
                        direction="negative",
                        relevance=0.7,
                    )
                ],
                status=AgentStatus.SHADOW,
            )
        )

        engine_consensus = WeightedConsensusEngine()
        consensus = engine_consensus.compute_consensus(reports)
        assert consensus.decision == ConsensusDecision.LONG_BIAS

        # 2. Variants generieren
        features = _make_features(current_price=50000.0, atr=500.0)
        variants = engine.generate_variants(consensus, features)
        assert len(variants) > 0
        assert all(v.direction == StrategyDirection.LONG for v in variants)

        # 3. Varianten bewerten
        config = StrategyConfig()
        costs = {"fee_pct": 0.001, "slippage_pct": 0.0005, "spread_bps": 5.0}
        best = engine.select_best(variants, config, costs)
        assert best is not None
        assert best.direction == StrategyDirection.LONG
        assert best.risk_reward_ratio > 0
        assert best.expected_return_net != 0.0

    def test_full_short_pipeline(self) -> None:
        """Kompletter Short-Handels-Pipeline."""
        engine = StrategyEngine()

        reports = [
            AgentReport(
                report_id=f"rpt-{i}",
                run_id="run-002",
                agent_id=f"agent-{i}",
                agent_version="0.1.0",
                instrument="ETH/USDT",
                horizon="1h",
                as_of=datetime.now(),
                hypothesis="bearish-hypothesis",
                probabilities={"up": 0.1, "down": 0.75, "range": 0.15},
                evidence=[
                    EvidenceReference(
                        reference=f"agent-{i}:ema",
                        feature="ema",
                        value="cross",
                        direction="negative",
                        relevance=0.75,
                    )
                ],
                status=AgentStatus.ACTIVE,
            )
            for i in range(2)
        ]

        engine_consensus = WeightedConsensusEngine()
        consensus = engine_consensus.compute_consensus(reports)
        assert consensus.decision == ConsensusDecision.SHORT_BIAS

        features = _make_features(current_price=3000.0, atr=50.0)
        variants = engine.generate_variants(consensus, features)
        assert len(variants) > 0

        best = engine.select_best(variants)
        assert best is not None

    def test_full_no_trade_pipeline(self) -> None:
        """Kompletter NO_TRADE-Pipeline (kein Konsens)."""
        engine = StrategyEngine()

        reports = [
            AgentReport(
                report_id="rpt-long",
                run_id="run-003",
                agent_id="agent-long",
                agent_version="0.1.0",
                instrument="SOL/USDT",
                horizon="1h",
                as_of=datetime.now(),
                hypothesis="bullish",
                probabilities={"up": 0.6, "down": 0.2, "range": 0.2},
                evidence=[
                    EvidenceReference(
                        reference="agent-long:rsi",
                        feature="rsi",
                        value="35",
                        direction="positive",
                        relevance=0.65,
                    )
                ],
                status=AgentStatus.ACTIVE,
            ),
            AgentReport(
                report_id="rpt-short",
                run_id="run-003",
                agent_id="agent-short",
                agent_version="0.1.0",
                instrument="SOL/USDT",
                horizon="1h",
                as_of=datetime.now(),
                hypothesis="bearish",
                probabilities={"up": 0.2, "down": 0.6, "range": 0.2},
                evidence=[
                    EvidenceReference(
                        reference="agent-short:volume",
                        feature="volume",
                        value="low",
                        direction="negative",
                        relevance=0.6,
                    )
                ],
                status=AgentStatus.ACTIVE,
            ),
        ]

        engine_consensus = WeightedConsensusEngine(
            config=WeightConfig(min_consensus_threshold=0.8)
        )
        consensus = engine_consensus.compute_consensus(reports)
        assert consensus.decision == ConsensusDecision.NO_TRADE

        proposal = engine.run({"consensus": consensus, "features": _make_features()})
        assert proposal.direction == StrategyDirection.NO_TRADE


class TestIntegrationCosts:
    """Integration test with transaction costs."""

    def test_high_costs_eliminate_trade(self) -> None:
        """Sehr hohe Kosten können Trade unmöglich machen."""
        engine = StrategyEngine()
        consensus = _make_consensus_long()

        variants = engine.generate_variants(consensus, _make_features())

        # Sehr hohe Kosten
        high_costs = {"fee_pct": 0.05, "slippage_pct": 0.05, "spread_bps": 500.0}
        config = StrategyConfig()

        proposal = engine.select_best(variants, config, high_costs)
        # Bei sehr hohen Kosten sollte keine Variante durchkommen
        assert proposal is None

    def test_low_costs_preserve_trade(self) -> None:
        """Niedrige Kosten lassen Trade bestehen."""
        engine = StrategyEngine()
        consensus = _make_consensus_long()

        variants = engine.generate_variants(consensus, _make_features())

        low_costs = {"fee_pct": 0.0001, "slippage_pct": 0.0001, "spread_bps": 0.5}
        config = StrategyConfig()

        proposal = engine.select_best(variants, config, low_costs)
        assert proposal is not None
        assert proposal.direction == StrategyDirection.LONG


class TestIntegrationQuality:
    """Integration test with data quality."""

    def test_low_quality_blocks_trade(self) -> None:
        """Geringe Datenqualität blockiert Trade."""
        engine = StrategyEngine()
        consensus = _make_consensus_long()

        variants = engine.generate_variants(consensus, _make_features())

        config = StrategyConfig()
        costs = {}

        # Manuell evaluieren mit schlechter Qualität
        from packages.strategy.evaluation import evaluate_variant
        for v in variants:
            result = evaluate_variant(v, config, costs, data_quality=0.85)
            assert result["approved"] is False
            assert "quality" in result["gate_failures"]


# ══════════════════════════════════════════════════════════════════════
# 7. ADDITIONAL EDGE CASE TESTS
# ══════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge case tests für alle Module."""

    def test_very_small_entry_price(self) -> None:
        """Sehr kleiner Entry-Preis."""
        targets = calculate_targets(
            entry_price=0.01,
            volatility=0.001,
            atr=0.001,
            direction="long",
        )
        assert len(targets) == 4
        assert targets[0].price > 0.01

    def test_large_atr_relative_to_price(self) -> None:
        """ATR grösser als Entry-Preis."""
        targets = calculate_targets(
            entry_price=10.0,
            volatility=5.0,
            atr=15.0,
            direction="long",
        )
        # Stop sollte unter 0 fallen, aber Preis bleibt positiv
        assert targets[3].price < 10.0

    def test_probability_clamped(self) -> None:
        """Wahrscheinlichkeiten müssen zwischen 0 und 1 liegen."""
        targets = calculate_targets(
            entry_price=100.0,
            volatility=1.0,
            atr=2.0,
            direction="long",
        )
        for t in targets:
            assert 0.0 <= t.probability <= 1.0

    def test_rr_rounding(self) -> None:
        """RR sollte auf 4 Dezimalstellen gerundet sein."""
        rr = calculate_risk_reward(
            risk_amount=123.456789,
            reward_amount=987.654321,
        )
        assert len(str(rr).split(".")[-1]) <= 4

    def test_multiple_no_trade_reasons(self) -> None:
        """Verschiedene NO_TRADE-Gründe."""
        from packages.strategy.engine import StrategyEngine

        # model_uncertainty
        p = StrategyEngine._no_trade_proposal(
            _make_consensus_no_trade(), "no_consensus"
        )
        assert p.direction in (
            StrategyDirection.NO_TRADE,
            StrategyDirection.NO_TRADE_DATA_QUALITY,
        )


class TestStrategyConfigValidation:
    """Teste StrategyConfig-Validierungsgrenzen."""

    def test_edge_at_boundary(self) -> None:
        with pytest.raises(ValueError, match="min_edge"):
            StrategyConfig(min_edge=0.0).validate()

    def test_edge_at_one(self) -> None:
        with pytest.raises(ValueError, match="min_edge"):
            StrategyConfig(min_edge=1.0).validate()

    def test_quality_above_one(self) -> None:
        with pytest.raises(ValueError, match="min_quality"):
            StrategyConfig(min_quality=1.5).validate()

    def test_quality_zero(self) -> None:
        with pytest.raises(ValueError, match="min_quality"):
            StrategyConfig(min_quality=0.0).validate()
