from trading_harness.models import TradeProposal
from trading_harness.services.risk_engine import RiskEngine

POLICY = {
    "allowed_symbols": ["BTCUSDT"],
    "max_risk_per_trade": 0.005,
    "max_daily_loss": 0.02,
    "max_portfolio_risk": 0.04,
    "max_leverage": 2.0,
    "max_positions": 5,
    "minimum_risk_reward": 1.8,
    "max_slippage_bps": 20,
}


def base_proposal(**updates):
    values = {
        "decision_id": "d1",
        "symbol": "BTCUSDT",
        "side": "BUY",
        "equity": 10000,
        "entry_price": 100,
        "stop_price": 98,
        "target_price": 104,
        "requested_leverage": 1,
        "open_positions": 0,
        "current_daily_loss_fraction": 0,
        "current_portfolio_risk_fraction": 0,
        "expected_slippage_bps": 5,
    }
    values.update(updates)
    return TradeProposal(**values)


def test_kill_switch_rejects():
    decision = RiskEngine(POLICY).evaluate(base_proposal(), kill_switch=True)
    assert decision.approved is False
    assert decision.reason == "KILL_SWITCH_ACTIVE"


def test_valid_trade_is_approved():
    decision = RiskEngine(POLICY).evaluate(base_proposal(), kill_switch=False)
    assert decision.approved is True
    assert decision.risk_amount == 50
    assert decision.max_position_size == 25
    assert decision.risk_reward == 2


def test_excess_leverage_rejects():
    decision = RiskEngine(POLICY).evaluate(
        base_proposal(requested_leverage=3),
        kill_switch=False,
    )
    assert decision.approved is False
    assert decision.reason == "MAX_LEVERAGE_EXCEEDED"


def test_low_rr_rejects():
    decision = RiskEngine(POLICY).evaluate(
        base_proposal(target_price=102),
        kill_switch=False,
    )
    assert decision.approved is False
    assert decision.reason == "MINIMUM_RISK_REWARD_NOT_MET"
