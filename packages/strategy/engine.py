"""StrategyEngine — Haupt-Orchestrator für Entry/Stop/Target/Exit.

Generiert StrategyVariants aus Consensus-Ergebnissen, wertet diese
aus und wählt die beste Variante für den Trade aus. Gibt NO_TRADE
zurück wenn keine Variante die Gates passiert.
"""

from __future__ import annotations

import uuid
from typing import Any

from packages.consensus import ConsensusDecision, ConsensusResult

from .entry import EntryCondition, EntryType
from .evaluation import evaluate_variant
from .models import (
    StrategyConfig,
    StrategyDirection,
    StrategyProposal,
    StrategyVariant,
)
from .targets import (
    calculate_targets,
    estimate_mfe_mae,
)


class StrategyEngine:
    """Generiert, bewertet und wählt StrategyVariants aus Consensus-Ergebnissen."""

    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = config or StrategyConfig()
        self.config.validate()

    def generate_variants(
        self,
        consensus: ConsensusResult,
        features: dict[str, Any],
        portfolio: dict[str, Any] | None = None,
    ) -> list[StrategyVariant]:
        """Generieren alternative Szenarien aus Consensus-Ergebnissen.

        Args:
            consensus: ConsensusResult aus dem Consensus-Service.
            features: Dict mit Merkmalen (ATR, volatility, etc.).
            portfolio: Portfolio-Informationen (optional).

        Returns:
            Liste von StrategyVariant-Objekten.
        """
        direction = consensus.decision

        # NO_TRADE Decisions → keine Varianten
        if direction == ConsensusDecision.NO_TRADE:
            return []

        # Direction mapping
        if direction == ConsensusDecision.LONG_BIAS:
            strategy_dir = StrategyDirection.LONG
        elif direction == ConsensusDecision.SHORT_BIAS:
            strategy_dir = StrategyDirection.SHORT
        elif direction == ConsensusDecision.RANGE:
            # Range → NO_TRADE
            return [
                StrategyVariant(
                    variant_id=str(uuid.uuid4()),
                    direction=StrategyDirection.NO_TRADE,
                    entry_price=0.0,
                    entry_type=EntryType.MARKET,
                    stop_loss=0.0,
                    targets=[],
                    probability=0.0,
                )
            ]
        else:
            return []

        # Feature-Extraktion
        entry_price = features.get("current_price", features.get("price", 100.0))
        atr_value = features.get("atr", features.get("volatility", 1.0))
        _volatility = features.get("volatility", atr_value * 0.6)
        entry_type = features.get("entry_type", EntryType.MARKET)
        entry_condition = features.get("entry_condition", EntryCondition.MOMENTUM)

        # Konsens-Konfidenz als Basis-Wahrscheinlichkeit
        base_prob = max(0.5, consensus.confidence)

        variants: list[StrategyVariant] = []

        # Base variant mit Konsens-Wahrscheinlichkeit
        variants.append(
            StrategyVariant(
                variant_id=f"base-{uuid.uuid4()}",
                direction=strategy_dir,
                entry_price=entry_price,
                entry_type=entry_type,
                stop_loss=self._compute_stop(entry_price, atr_value, strategy_dir),
                targets=self._compute_targets_list(entry_price, atr_value, strategy_dir),
                probability=base_prob,
                entry_condition=entry_condition,
                metadata={
                    "consensus_confidence": consensus.confidence,
                    "source": "consensus_base",
                },
            )
        )

        # Aggressive variant — engerer Stop, näheres Target
        variants.append(
            StrategyVariant(
                variant_id=f"aggressive-{uuid.uuid4()}",
                direction=strategy_dir,
                entry_price=entry_price,
                entry_type=entry_type,
                stop_loss=self._compute_stop(
                    entry_price, atr_value, strategy_dir, multiplier=1.5
                ),
                targets=self._compute_targets_list(
                    entry_price, atr_value, strategy_dir, tp_multiplier=0.8
                ),
                probability=min(0.95, base_prob * 1.1),
                entry_condition=entry_condition,
                metadata={
                    "consensus_confidence": consensus.confidence,
                    "source": "consensus_aggressive",
                },
            )
        )

        # Conservative variant — weiter Stop, fernes Target
        variants.append(
            StrategyVariant(
                variant_id=f"conservative-{uuid.uuid4()}",
                direction=strategy_dir,
                entry_price=entry_price,
                entry_type=entry_type,
                stop_loss=self._compute_stop(
                    entry_price, atr_value, strategy_dir, multiplier=3.0
                ),
                targets=self._compute_targets_list(
                    entry_price, atr_value, strategy_dir, tp_multiplier=1.5
                ),
                probability=min(0.95, base_prob * 0.9),
                entry_condition=entry_condition,
                metadata={
                    "consensus_confidence": consensus.confidence,
                    "source": "consensus_conservative",
                },
            )
        )

        return variants

    def select_best(
        self,
        variants: list[StrategyVariant],
        config: StrategyConfig | None = None,
        costs: dict[str, float] | None = None,
    ) -> StrategyProposal | None:
        """Auswählen der besten Variante nach Gate-Bewertung.

        Bewertet jede Variante mit evaluate_variant und wählt
        diejenige mit der höchsten erwarteten Netto-Rendite aus,
        die alle Gates passiert.

        Args:
            variants: Liste von StrategyVariant-Objekten.
            config: Override-Config (optional).
            costs: Kosten-Parameter.

        Returns:
            Beste StrategyProposal oder None wenn keine Variante durchkommt.
        """
        cfg = config or self.config
        costs = costs or {}

        best_proposal: StrategyProposal | None = None
        best_net_ev: float = float("-inf")

        for variant in variants:
            if variant.direction in (
                StrategyDirection.NO_TRADE,
                StrategyDirection.NO_TRADE_DATA_QUALITY,
                StrategyDirection.NO_TRADE_INSUFFICIENT_EDGE,
                StrategyDirection.NO_TRADE_RISK,
                StrategyDirection.NO_TRADE_PORTFOLIO,
                StrategyDirection.NO_TRADE_MODEL_UNCERTAINTY,
            ):
                continue

            evaluation = evaluate_variant(variant, cfg, costs, data_quality=1.0)

            if not evaluation["approved"]:
                continue

            if evaluation["expected_return_net"] > best_net_ev:
                best_net_ev = evaluation["expected_return_net"]
                best_proposal = self._build_proposal(variant, evaluation)

        return best_proposal

    def run(
        self,
        context: dict[str, Any],
    ) -> StrategyProposal:
        """Haupt-Methode: Konsens → Variants → Evaluate → Select → Proposal.

        Args:
            context: Dict mit 'consensus', 'features', 'portfolio', 'costs'.

        Returns:
            StrategyProposal (ggf. NO_TRADE).
        """
        consensus: ConsensusResult = context["consensus"]
        features: dict[str, Any] = context.get("features", {})
        portfolio = context.get("portfolio")
        costs = context.get("costs", {})

        # Variants generieren
        variants = self.generate_variants(consensus, features, portfolio)

        # Beste Variante auswählen
        proposal = self.select_best(variants, self.config, costs)

        if proposal is not None:
            return proposal

        # Keine Variante — NO_TRADE begründen
        reason = self._determine_no_trade_reason(consensus, variants)
        return self._no_trade_proposal(consensus, reason)

    # --- Private helpers ---

    @staticmethod
    def _compute_stop(
        entry_price: float,
        atr: float,
        direction: StrategyDirection,
        multiplier: float = 2.0,
    ) -> float:
        """Berechnen Stopp-Loss-Preis."""
        stop_distance = atr * multiplier
        if direction == StrategyDirection.LONG:
            return round(entry_price - stop_distance, 8)
        return round(entry_price + stop_distance, 8)

    @staticmethod
    def _compute_targets_list(
        entry_price: float,
        atr: float,
        direction: StrategyDirection,
        tp_multiplier: float = 1.0,
    ) -> list[float]:
        """Berechnen Take-Profit-Preise."""
        mult = tp_multiplier
        dist1 = atr * 1.5 * mult
        dist2 = atr * 3.0 * mult
        dist3 = atr * 5.0 * mult

        if direction == StrategyDirection.LONG:
            return [
                round(entry_price + dist1, 8),
                round(entry_price + dist2, 8),
                round(entry_price + dist3, 8),
            ]
        return [
            round(entry_price - dist1, 8),
            round(entry_price - dist2, 8),
            round(entry_price - dist3, 8),
        ]

    def _build_proposal(
        self,
        variant: StrategyVariant,
        evaluation: dict[str, Any],
    ) -> StrategyProposal:
        """Bauen eine StrategyProposal aus einer Variante."""
        entry_price = variant.entry_price
        atr = abs(entry_price * 0.02) if entry_price > 0 else 1.0
        direction_str = (
            variant.direction.value
            if isinstance(variant.direction, StrategyDirection)
            else variant.direction
        )

        target_levels = calculate_targets(
            entry_price=entry_price,
            volatility=atr * 0.6,
            atr=atr,
            direction=direction_str,
        )

        mae, mfe = estimate_mfe_mae(
            entry_price, target_levels, atr * 0.6, direction_str
        )

        prob_target = evaluation.get(
            "prob_target_before_stop", variant.probability
        )
        prob_stop = evaluation.get(
            "prob_stop_before_target", 1.0 - variant.probability
        )

        target_prices = [
            t.price for t in target_levels if t.type.name != "STOP_LIMIT"
        ]

        return StrategyProposal(
            direction=variant.direction,
            entry_type=variant.entry_type,
            entry_price=variant.entry_price,
            entry_condition=variant.entry_condition,
            stop_loss=variant.stop_loss,
            targets=target_prices,
            prob_target_before_stop=round(prob_target, 4),
            prob_stop_before_target=round(prob_stop, 4),
            expected_return_gross=evaluation.get("expected_return_gross", 0.0),
            expected_return_net=evaluation.get("expected_return_net", 0.0),
            expected_costs=evaluation.get("expected_costs", 0.0),
            expected_mae=mae,
            expected_mfe=mfe,
            risk_reward_ratio=evaluation.get("risk_reward_ratio", 0.0),
            confidence=variant.probability,
            metadata=variant.metadata,
        )

    @staticmethod
    def _determine_no_trade_reason(
        consensus: ConsensusResult,
        variants: list[StrategyVariant],
    ) -> str:
        """Ermitteln der Begründung für NO_TRADE."""
        if consensus.decision == ConsensusDecision.NO_TRADE:
            return "no_consensus"
        if not variants:
            return "no_variants"
        # variants exist but none passed gates — check which gate failed
        return "gates_failed"

    @staticmethod
    def _no_trade_proposal(
        consensus: ConsensusResult,
        reason: str,
    ) -> StrategyProposal:
        """Erzeugen ein NO_TRADE Proposal."""
        if consensus.decision == ConsensusDecision.NO_TRADE:
            direction = StrategyDirection.NO_TRADE
        elif reason == "gates_failed":
            # Finde das kritischste Gate-Failure
            direction = StrategyDirection.NO_TRADE_INSUFFICIENT_EDGE
        elif reason == "no_variants":
            direction = StrategyDirection.NO_TRADE_MODEL_UNCERTAINTY
        else:
            direction = StrategyDirection.NO_TRADE

        return StrategyProposal(
            direction=direction,
            entry_type="",
            entry_price=0.0,
            entry_condition="",
            stop_loss=0.0,
            targets=[],
            reason=reason,
        )
