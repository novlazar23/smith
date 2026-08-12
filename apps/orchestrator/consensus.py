"""Konsens und Entscheidungsfindung: Gewichtung, Strategie, Portfolio.

§11.13–11.15
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from apps.orchestrator.graph import StageManager, TradingGraphState
from apps.orchestrator.stages_enum import AnalysisStage
from packages.consensus import (
    ConsensusDecision,
    ConsensusResult,
    WeightedConsensusEngine,
)
from packages.strategy.engine import StrategyEngine
from packages.strategy.models import (
    StrategyDirection,
)


def calculate_consensus(
    state: TradingGraphState,
    manager: StageManager,
    weighted_engine: WeightedConsensusEngine,
) -> tuple[TradingGraphState, StageManager, ConsensusResult]:
    """11.13 calculate_consensus — Kalibrierte Wahrscheinlichkeiten.

    Aggregiert First/Second Round Reports mit gewichteten Abstimmungen,
    berücksichtigt Abhängigkeiten und Kalibrierung.

    Args:
        state: Aktueller Graphzustand.
        manager: StageManager.
        weighted_engine: WeightedConsensusEngine für Gewichtung.

    Returns:
        (TradingGraphState, StageManager, ConsensusResult)
    """
    all_reports = list(state.first_round_reports or [])
    all_reports.extend(state.second_round_reports or [])

    # In AgentReports umwandeln wenn nötig
    from packages.schemas.agent_report import AgentReport

    agent_reports: list[AgentReport] = []
    for r in all_reports:
        if isinstance(r, AgentReport):
            agent_reports.append(r)
        elif isinstance(r, dict):
            # Konvertiere dict zu AgentReport für konsens
            agent_reports.append(AgentReport(report_id=r.get("report_id", "unknown"), run_id=state.run_id, agent_id=r.get("agent_id", "unknown"), agent_version=r.get("agent_version", "0.1.0"), instrument=r.get("instrument", state.instrument), horizon=r.get("horizon", "1h"), as_of=r.get("as_of", datetime.now(UTC)), hypothesis=r.get("hypothesis", ""), probabilities=r.get("probabilities", {"up": 0.33, "down": 0.33, "range": 0.34}), evidence=r.get("evidence", [{"reference": "default", "feature": "default", "value": "default", "direction": "neutral", "relevance": 0.5}]), raw_confidence=r.get("raw_confidence"), calibrated_confidence=r.get("calibrated_confidence"), status=r.get("status", "shadow")))

    # Wenn keine Reports → NO_TRADE
    if not agent_reports:
        consensus = ConsensusResult(
            decision=ConsensusDecision.NO_TRADE,
            vote_distribution={"long": 0.0, "short": 0.0, "range": 0.0, "abstain": 1.0},
            agent_weights={},
            agent_agreements=[],
            agent_disagreements=[],
            confidence=0.0,
            reason="No agent reports available",
        )
    else:
        # Filter out reports with empty evidence — AgentReport requires >= 1
        valid_reports = [r for r in agent_reports if getattr(r, 'evidence', None) and len(r.evidence) > 0]
        if not valid_reports:
            consensus = ConsensusResult(
                decision=ConsensusDecision.NO_TRADE,
                vote_distribution={"long": 0.0, "short": 0.0, "range": 0.0, "abstain": 1.0},
                agent_weights={},
                agent_agreements=[],
                agent_disagreements=[],
                confidence=0.0,
                reason="No valid agent reports available",
            )
        else:
            consensus = weighted_engine.compute_consensus(valid_reports)

    state = state.model_copy(update={
        "current_stage": AnalysisStage.CALCULATE_CONSENSUS.value,
        "consensus_report": {
            "decision": consensus.decision.value,
            "confidence": consensus.confidence,
            "agent_weights": consensus.agent_weights,
            "agreements": consensus.agent_agreements,
            "disagreements": consensus.agent_disagreements,
            "vote_distribution": consensus.vote_distribution,
        },
    })

    manager.transition(
        AnalysisStage.CALCULATE_CONSENSUS,
        inputs={"agent_count": len(agent_reports)},
        outputs={"decision": consensus.decision.value, "confidence": consensus.confidence},
    )

    return state, manager, consensus


def generate_strategy_step(
    state: TradingGraphState,
    manager: StageManager,
    strategy_engine: StrategyEngine,
    consensus: ConsensusResult | None = None,
) -> tuple[TradingGraphState, StageManager]:
    """11.14 generate_strategy — Varianten, Target-before-Stop, EV.

    Generiert Strategie-Varianten (Base/Aggressive/Conservative),
    wendet Target-before-Stop an, berechnet EV nach Kosten.

    Args:
        state: Aktueller Graphzustand.
        manager: StageManager.
        strategy_engine: StrategyEngine für Variante-Generierung.
        consensus: Optionaler Konsens (wird aus State gelesen wenn None).

    Returns:
        (TradingGraphState, StageManager) — mit strategy_report.
    """
    if consensus is None and state.consensus_report:
        # Versuche Konsens aus State zu rekonstruieren
        consensus_data = state.consensus_report
        consensus = ConsensusResult(
            decision=ConsensusDecision(consensus_data.get("decision", "NO_TRADE")),
            vote_distribution=consensus_data.get("vote_distribution", {}),
            agent_weights=consensus_data.get("agent_weights", {}),
            agent_agreements=consensus_data.get("agreements", []),
            agent_disagreements=consensus_data.get("disagreements", []),
            confidence=consensus_data.get("confidence", 0.0),
            reason=consensus_data.get("reason", ""),
        )
    elif consensus is None:
        consensus = ConsensusResult(
            decision=ConsensusDecision.NO_TRADE,
            vote_distribution={"long": 0.0, "short": 0.0, "range": 0.0, "abstain": 0.0},
            agent_weights={},
            agent_agreements=[],
            agent_disagreements=[],
            confidence=0.0,
            reason="No consensus data available",
        )

    features = state.features or {}
    variants = strategy_engine.generate_variants(consensus, features)
    proposal = strategy_engine.select_best(variants) if variants else None

    strategy_report = {
        "variants": [
            {
                "variant_id": v.variant_id,
                "direction": v.direction.value if isinstance(v.direction, StrategyDirection) else str(v.direction),
                "entry_price": v.entry_price,
                "stop_loss": v.stop_loss,
                "targets": v.targets,
                "probability": v.probability,
            }
            for v in variants
        ],
        "best_proposal": {
            "direction": proposal.direction.value if proposal else "NO_TRADE",
            "entry_price": proposal.entry_price if proposal else 0.0,
            "stop_loss": proposal.stop_loss if proposal else 0.0,
            "targets": proposal.targets if proposal else [],
            "expected_return_net": proposal.expected_return_net if proposal else 0.0,
        } if proposal else None,
        "variant_count": len(variants),
    }

    state = state.model_copy(update={
        "current_stage": AnalysisStage.GENERATE_STRATEGY.value,
        "strategy_report": strategy_report,
    })

    manager.transition(
        AnalysisStage.GENERATE_STRATEGY,
        inputs={"variant_count": len(variants)},
        outputs={"has_proposal": proposal is not None},
    )

    return state, manager


def evaluate_portfolio_step(
    state: TradingGraphState,
    manager: StageManager,
    portfolio_manager: Any | None = None,
) -> tuple[TradingGraphState, StageManager]:
    """11.15 evaluate_portfolio — Bestehende Exposures prüfen.

    Bewertet bestehende Portfolio-Exposures, Korrelation und Konzentration.

    Args:
        state: Aktueller Graphzustand.
        manager: StageManager.
        portfolio_manager: Portfolio-Manager für Berechnungen.

    Returns:
        (TradingGraphState, StageManager) — mit portfolio_report.
    """
    existing_exposures: list[dict[str, Any]] = []
    correlation_score = 1.0
    concentration_score = 1.0

    if portfolio_manager is not None:
        existing_exposures = portfolio_manager.get_exposures(state)
        correlation_score = portfolio_manager.correlation_score(state)
        concentration_score = portfolio_manager.concentration_score(state)

    portfolio_report = {
        "existing_exposures": existing_exposures,
        "correlation_score": correlation_score,
        "concentration_score": concentration_score,
        "max_exposure": max((e.get("size", 0) for e in existing_exposures), default=0),
        "new_trade_allowed": correlation_score > 0.3 and concentration_score > 0.3,
    }

    state = state.model_copy(update={
        "current_stage": AnalysisStage.EVALUATE_PORTFOLIO.value,
        "portfolio_report": portfolio_report,
    })

    manager.transition(
        AnalysisStage.EVALUATE_PORTFOLIO,
        inputs={"exposure_count": len(existing_exposures)},
        outputs={
            "correlation_score": correlation_score,
            "new_trade_allowed": portfolio_report["new_trade_allowed"],
        },
    )

    return state, manager
