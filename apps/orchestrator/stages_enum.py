"""AnalysisStage Enum — 18 Graphknoten der Analyse-Pipeline (§11)."""

from __future__ import annotations

from enum import StrEnum


class AnalysisStage(StrEnum):
    """Die 18 Stufen des Analyse-Graphen.

    Reihenfolge: CREATE_RUN → BUILD_MARKET_SNAPSHOT → VALIDATE_DATA →
    COMPUTE_FEATURES → CLASSIFY_REGIME → RUN_FIRST_ROUND → SEAL_FIRST_ROUND →
    HISTORICAL_VALIDATE → MEASURE_DEPENDENCIES → RUN_SECOND_ROUND →
    CONTRARIAN_REVIEW → MULTI_TIMEFRAME → CALCULATE_CONSENSUS →
    GENERATE_STRATEGY → EVALUATE_PORTFOLIO → APPLY_RISK_GATES →
    PUBLISH_DECISION → SCHEDULE_EVALUATION

    Sonderübergänge:
    - RUN_SECOND_ROUND kann nach SEAL_FIRST_ROUND gesprungen werden (klarer Consensus)
    - CONTRARIAN_REVIEW kann bei hohem Dissens gesprungen werden
    """

    CREATE_RUN = "create_run"
    BUILD_MARKET_SNAPSHOT = "build_market_snapshot"
    VALIDATE_DATA = "validate_data"
    COMPUTE_FEATURES = "compute_features"
    CLASSIFY_REGIME = "classify_regime"
    RUN_FIRST_ROUND = "run_first_round"
    SEAL_FIRST_ROUND = "seal_first_round"
    HISTORICAL_VALIDATE = "historical_validate"
    MEASURE_DEPENDENCIES = "measure_dependencies"
    RUN_SECOND_ROUND = "run_second_round"
    CONTRARIAN_REVIEW = "contrarian_review"
    MULTI_TIMEFRAME = "multi_timeframe"
    CALCULATE_CONSENSUS = "calculate_consensus"
    GENERATE_STRATEGY = "generate_strategy"
    EVALUATE_PORTFOLIO = "evaluate_portfolio"
    APPLY_RISK_GATES = "apply_risk_gates"
    PUBLISH_DECISION = "publish_decision"
    SCHEDULE_EVALUATION = "schedule_evaluation"

    @classmethod
    def default_sequence(cls) -> list[AnalysisStage]:
        """Gibt die standard Sequenz aller 18 Stufen zurück."""
        return [
            cls.CREATE_RUN,
            cls.BUILD_MARKET_SNAPSHOT,
            cls.VALIDATE_DATA,
            cls.COMPUTE_FEATURES,
            cls.CLASSIFY_REGIME,
            cls.RUN_FIRST_ROUND,
            cls.SEAL_FIRST_ROUND,
            cls.HISTORICAL_VALIDATE,
            cls.MEASURE_DEPENDENCIES,
            cls.RUN_SECOND_ROUND,
            cls.CONTRARIAN_REVIEW,
            cls.MULTI_TIMEFRAME,
            cls.CALCULATE_CONSENSUS,
            cls.GENERATE_STRATEGY,
            cls.EVALUATE_PORTFOLIO,
            cls.APPLY_RISK_GATES,
            cls.PUBLISH_DECISION,
            cls.SCHEDULE_EVALUATION,
        ]

    @classmethod
    def minimal_sequence(cls) -> list[AnalysisStage]:
        """Gibt die minimale Sequenz zurück (ohne optionalen Round 2)."""
        return [
            cls.CREATE_RUN,
            cls.BUILD_MARKET_SNAPSHOT,
            cls.VALIDATE_DATA,
            cls.COMPUTE_FEATURES,
            cls.CLASSIFY_REGIME,
            cls.RUN_FIRST_ROUND,
            cls.SEAL_FIRST_ROUND,
            cls.HISTORICAL_VALIDATE,
            cls.MEASURE_DEPENDENCIES,
            cls.CONTRARIAN_REVIEW,
            cls.MULTI_TIMEFRAME,
            cls.CALCULATE_CONSENSUS,
            cls.GENERATE_STRATEGY,
            cls.EVALUATE_PORTFOLIO,
            cls.APPLY_RISK_GATES,
            cls.PUBLISH_DECISION,
            cls.SCHEDULE_EVALUATION,
        ]
