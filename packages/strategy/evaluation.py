"""EV nach Kosten — Expected Value Berechnung für StrategyVariants.

Berechnet erwartete Rendite brutto und netto (nach Kosten/Slippage/Spread),
Risk-Reward-Verhältnis, und wendet die strategischen Gates
(min_edge, min_prob, min_rr, min_quality) an.
"""

from __future__ import annotations

from typing import Any

from .models import StrategyConfig, StrategyDirection, StrategyVariant


def calculate_expected_return(
    variant: StrategyVariant,
    config: StrategyConfig,
    costs: dict[str, float] | None = None,
) -> dict[str, float]:
    """Berechnen erwartete Rendite brutto und netto.

    Expected return = weighted average of outcomes:
      P(win) * reward - P(lose) * risk

    Nach Kosten:
      Net = Gross - transaction_costs - slippage - spread_cost

    Args:
        variant: StrategyVariant mit Entry/Stop/Targets.
        config: StrategyConfig mit Schwellwerten.
        costs: Dict mit 'fee_pct', 'slippage_pct', 'spread_bps'.

    Returns:
        Dict mit 'gross', 'costs', 'net'.
    """
    costs = costs or {}
    fee_pct = float(costs.get("fee_pct", 0.0))
    slippage_pct = float(costs.get("slippage_pct", 0.0))
    spread_bps = float(costs.get("spread_bps", 0.0)) / 10000.0

    # Transaction costs (round-trip: entry + exit)
    total_cost_pct = 2.0 * (fee_pct + slippage_pct + spread_bps)

    # Gross expected return (probability-weighted)
    direction = variant.direction
    entry = variant.entry_price
    stop = variant.stop_loss
    targets = variant.targets

    if not targets or entry == 0:
        return {"gross": 0.0, "costs": 0.0, "net": 0.0}

    # Wahrscheinlichkeits-gewichteter Expected Return
    if direction == StrategyDirection.LONG:
        # Long: Reward = target - entry, Risk = entry - stop
        best_target = max(targets)
        reward = best_target - entry
        risk = entry - stop
    else:
        # Short: Reward = entry - target, Risk = stop - entry
        best_target = min(targets)
        reward = entry - best_target
        risk = stop - entry

    if risk <= 0:
        return {"gross": 0.0, "costs": 0.0, "net": 0.0}

    # Wahrscheinlichkeiten aus variant
    prob_win = variant.probability if variant.probability > 0 else 0.55
    prob_lose = 1.0 - prob_win

    # Gross EV
    gross_ev = prob_win * reward - prob_lose * risk

    # Costs als Anteil des Entry-Preises
    gross_ev_pct = gross_ev / abs(entry) if abs(entry) > 0 else 0.0
    net_ev_pct = gross_ev_pct - total_cost_pct

    return {
        "gross": round(gross_ev_pct, 8),
        "costs": round(total_cost_pct, 8),
        "net": round(net_ev_pct, 8),
    }


def calculate_risk_reward(
    risk_amount: float,
    reward_amount: float,
) -> float:
    """Berechnen Risk-Reward-Verhältnis.

    RR = reward_amount / risk_amount

    Args:
        risk_amount: Absoluter Risiko-Betrag.
        reward_amount: Absoluter Belohnungs-Betrag.

    Returns:
        Risk-Reward-Verhältnis (float).

    Raises:
        ValueError: Wenn risk_amount <= 0.
    """
    if risk_amount <= 0:
        raise ValueError("risk_amount must be positive")
    return round(reward_amount / risk_amount, 4)


def evaluate_variant(
    variant: StrategyVariant,
    config: StrategyConfig,
    costs: dict[str, float] | None = None,
    data_quality: float = 1.0,
) -> dict[str, Any]:
    """Bewerten einer einzelnen StrategyVariant.

    Berechnet alle relevanten Metriken: EV, RR, Wahrscheinlichkeiten,
    MFE/MAE und wendet Gates an.

    Args:
        variant: Zu bewertende StrategyVariant.
        config: Strategie-Konfiguration.
        costs: Kosten-Parameter.
        data_quality: Datenqualitäts-Score (0.0-1.0).

    Returns:
        Dict mit allen Metriken und Gate-Status.
    """
    costs = costs or {}
    direction = variant.direction

    # Risk und Reward berechnen
    risk = variant.risk_amount
    reward = variant.total_reward

    if direction in (
        StrategyDirection.NO_TRADE,
        StrategyDirection.NO_TRADE_DATA_QUALITY,
        StrategyDirection.NO_TRADE_INSUFFICIENT_EDGE,
        StrategyDirection.NO_TRADE_RISK,
        StrategyDirection.NO_TRADE_PORTFOLIO,
        StrategyDirection.NO_TRADE_MODEL_UNCERTAINTY,
    ):
        return {
            "variant_id": variant.variant_id,
            "direction": direction,
            "approved": False,
            "expected_return_gross": 0.0,
            "expected_return_net": 0.0,
            "expected_costs": 0.0,
            "risk_reward_ratio": 0.0,
            "prob_target_before_stop": 0.0,
            "prob_stop_before_target": 1.0,
            "data_quality": data_quality,
            "gates_passed": False,
            "gate_failures": ["no_trade_direction"],
        }

    # EV nach Kosten
    ev = calculate_expected_return(variant, config, costs)

    # Risk-Reward
    rr = calculate_risk_reward(risk, reward) if risk > 0 else 0.0

    # Probabilities
    prob_target = variant.probability if variant.probability > 0 else 0.55
    prob_stop = 1.0 - prob_target

    # Gate-Prüfung
    gate_results = {
        "edge": ev["net"] >= config.min_edge,
        "prob": prob_target >= config.min_prob,
        "rr": rr >= config.min_rr,
        "quality": data_quality >= config.min_quality,
    }

    all_passed = all(gate_results.values())

    gate_failures = [
        key for key, passed in gate_results.items() if not passed
    ]

    return {
        "variant_id": variant.variant_id,
        "direction": direction,
        "approved": all_passed,
        "expected_return_gross": ev["gross"],
        "expected_return_net": ev["net"],
        "expected_costs": ev["costs"],
        "risk_reward_ratio": rr,
        "risk_amount": risk,
        "reward_amount": reward,
        "prob_target_before_stop": prob_target,
        "prob_stop_before_target": prob_stop,
        "data_quality": data_quality,
        "gates_passed": all_passed,
        "gate_results": gate_results,
        "gate_failures": gate_failures,
    }


def apply_gates(
    evaluation: dict[str, Any],
    config: StrategyConfig,
) -> bool:
    """Prüfen ob eine Evaluation alle Gates passiert.

    Alle vier Gates müssen bestanden sein:
      - edge: expected_return_net >= min_edge
      - prob: prob_target_before_stop >= min_prob
      - rr: risk_reward_ratio >= min_rr
      - quality: data_quality >= min_quality

    Args:
        evaluation: Ergebnis von evaluate_variant.
        config: Strategie-Konfiguration.

    Returns:
        True wenn alle Gates bestanden.
    """
    if not evaluation:
        return False

    return bool(evaluation.get("gates_passed", False))
