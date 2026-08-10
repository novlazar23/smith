from __future__ import annotations

from trading_harness.models import RiskDecision, TradeProposal


class RiskEngine:
    def __init__(self, policy: dict):
        self.policy = policy

    def evaluate(self, proposal: TradeProposal, kill_switch: bool) -> RiskDecision:
        if kill_switch:
            return RiskDecision(approved=False, reason="KILL_SWITCH_ACTIVE")

        if proposal.symbol not in set(self.policy.get("allowed_symbols", [])):
            return RiskDecision(approved=False, reason="SYMBOL_NOT_ALLOWED")

        if proposal.requested_leverage > float(self.policy["max_leverage"]):
            return RiskDecision(approved=False, reason="MAX_LEVERAGE_EXCEEDED")

        if proposal.open_positions >= int(self.policy["max_positions"]):
            return RiskDecision(approved=False, reason="MAX_POSITIONS_REACHED")

        if proposal.current_daily_loss_fraction >= float(self.policy["max_daily_loss"]):
            return RiskDecision(approved=False, reason="MAX_DAILY_LOSS_REACHED")

        if proposal.current_portfolio_risk_fraction >= float(self.policy["max_portfolio_risk"]):
            return RiskDecision(approved=False, reason="MAX_PORTFOLIO_RISK_REACHED")

        if proposal.expected_slippage_bps > float(self.policy["max_slippage_bps"]):
            return RiskDecision(approved=False, reason="MAX_SLIPPAGE_EXCEEDED")

        stop_distance = abs(proposal.entry_price - proposal.stop_price)
        target_distance = abs(proposal.target_price - proposal.entry_price)

        if stop_distance <= 0:
            return RiskDecision(approved=False, reason="INVALID_STOP_DISTANCE")

        rr = target_distance / stop_distance
        if rr < float(self.policy["minimum_risk_reward"]):
            return RiskDecision(
                approved=False,
                reason="MINIMUM_RISK_REWARD_NOT_MET",
                risk_reward=rr,
            )

        max_risk_fraction = float(self.policy["max_risk_per_trade"])
        remaining_portfolio_risk = max(
            0.0,
            float(self.policy["max_portfolio_risk"])
            - proposal.current_portfolio_risk_fraction,
        )
        risk_fraction = min(max_risk_fraction, remaining_portfolio_risk)

        if risk_fraction <= 0:
            return RiskDecision(approved=False, reason="NO_RISK_BUDGET_AVAILABLE")

        risk_amount = proposal.equity * risk_fraction
        max_position_size = risk_amount / stop_distance

        return RiskDecision(
            approved=True,
            reason="APPROVED",
            max_position_size=max_position_size,
            risk_amount=risk_amount,
            risk_fraction=risk_fraction,
            risk_reward=rr,
        )
