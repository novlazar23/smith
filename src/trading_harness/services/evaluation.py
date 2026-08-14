from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, Protocol

from trading_harness.models import (
    EvaluationResult,
    MarketRegime,
    OutcomeRecord,
    WalkForwardResult,
)
from trading_harness.services.evaluation_result_store import (
    PersistedEvaluationResultStore,
)


class OutcomeStore(Protocol):
    """Protocol for outcome stores used by evaluation."""

    def get(self, outcome_id: str) -> OutcomeRecord | None: ...
    def by_agent(self, agent_id: str) -> list[OutcomeRecord]: ...
    def by_run(self, run_id: str) -> list[OutcomeRecord]: ...
    def by_regime(self, regime: MarketRegime) -> list[OutcomeRecord]: ...


# ---------------------------------------------------------------------------
# Metric computations
# ---------------------------------------------------------------------------

@dataclass
class _BinaryClassificationResult:
    """Internal container for binary classification metrics."""

    correct: int = 0
    total: int = 0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    brier_score: float = 0.0
    calibration_error: float = 0.0
    expectancy: float = 0.0
    directional_accuracy: float = 0.0


def _compute_classification_metrics(
    outcomes: list[OutcomeRecord],
) -> _BinaryClassificationResult:
    """Compute binary classification metrics from outcomes."""
    result = _BinaryClassificationResult()
    result.total = len(outcomes)

    if result.total == 0:
        return result

    for o in outcomes:
        predicted_correct = o.direction_predicted.upper() == o.direction_actual.upper()
        if predicted_correct:
            result.correct += 1
        # Directional accuracy
        if predicted_correct:
            result.directional_accuracy = result.correct / result.total

        # Confusion matrix (treating LONG as positive)
        pred_long = o.direction_predicted.upper() in ("LONG", "BUY")
        actual_long = o.direction_actual.upper() in ("LONG", "BUY")

        if pred_long and actual_long:
            result.true_positives += 1
        elif pred_long and not actual_long:
            result.false_positives += 1
        elif not pred_long and actual_long:
            result.false_negatives += 1
        else:
            result.true_negatives += 1

    # Brier Score
    # For binary classification: B = (1/T) * sum((p_i - o_i)^2)
    # where p_i is predicted probability (confidence) and o_i is actual outcome (1 or 0)
    brier_sum = 0.0
    for o in outcomes:
        pred_long = o.direction_predicted.upper() in ("LONG", "BUY")
        actual_long = o.direction_actual.upper() in ("LONG", "BUY")
        actual_binary = 1.0 if actual_long else 0.0
        predicted_prob = o.confidence_predicted if pred_long else (1.0 - o.confidence_predicted)
        predicted_prob = max(0.01, min(0.99, predicted_prob))  # clamp to avoid edge cases
        brier_sum += (predicted_prob - actual_binary) ** 2

    result.brier_score = brier_sum / result.total if result.total > 0 else 0.0

    # Calibration Error (Expected Calibration Error with 5 bins)
    result.calibration_error = _compute_ece(outcomes)

    # Expectancy = (WinRate * AvgWin) - (LossRate * AvgLoss)
    wins = [o for o in outcomes if o.realized_pnl > 0]
    losses = [o for o in outcomes if o.realized_pnl < 0]
    total_trades = len(outcomes)

    if total_trades > 0:
        win_rate = len(wins) / total_trades
        avg_win = sum(o.realized_pnl for o in wins) / len(wins) if wins else 0.0
        avg_loss = abs(sum(o.realized_pnl for o in losses) / len(losses)) if losses else 0.0
        result.expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    return result


def _compute_ece(outcomes: list[OutcomeRecord], num_bins: int = 5) -> float:
    """Compute Expected Calibration Error with equal-width bins."""
    if not outcomes:
        return 0.0

    # Group predictions into confidence bins
    bins: list[list[OutcomeRecord]] = [[] for _ in range(num_bins)]

    for o in outcomes:
        # Map confidence [0, 1] to bin index
        # For directional predictions, use confidence if predicted direction matches "positive"
        pred_long = o.direction_predicted.upper() in ("LONG", "BUY")
        if pred_long:
            prob = o.confidence_predicted
        else:
            prob = 1.0 - o.confidence_predicted

        # Clamp to [0, 1]
        prob = max(0.0, min(1.0, prob))

        # Assign to bin (0-indexed)
        bin_idx = min(int(prob * num_bins), num_bins - 1)
        bins[bin_idx].append(o)

    # Compute ECE
    ece = 0.0
    total = len(outcomes)

    for bin_items in bins:
        if not bin_items:
            continue

        bin_size = len(bin_items)
        avg_confidence = sum(o.confidence_predicted for o in bin_items) / bin_size
        avg_accuracy = sum(
            1.0 if o.direction_predicted.upper() == o.direction_actual.upper() else 0.0
            for o in bin_items
        ) / bin_size

        ece += (bin_size / total) * abs(avg_confidence - avg_accuracy)

    return ece


def _compute_mfe_mae(
    outcomes: list[OutcomeRecord],
) -> dict[str, float]:
    """Compute aggregate MFE/MAE statistics."""
    if not outcomes:
        return {"avg_mfe": 0.0, "avg_mae": 0.0, "max_mfe": 0.0, "max_mae": 0.0}

    mfe_values = [o.mfe for o in outcomes if o.mfe > 0]
    mae_values = [o.mae for o in outcomes if o.mae > 0]

    return {
        "avg_mfe": sum(mfe_values) / len(mfe_values) if mfe_values else 0.0,
        "avg_mae": sum(mae_values) / len(mae_values) if mae_values else 0.0,
        "max_mfe": max(mfe_values) if mfe_values else 0.0,
        "max_mae": max(mae_values) if mae_values else 0.0,
    }


# ---------------------------------------------------------------------------
# EvaluationService — central metric computation engine
# ---------------------------------------------------------------------------

class EvaluationService:
    """Central evaluation service for Phase 2 metrics.

    Computes: Brier Score, Calibration Error, Expectancy, MFE/MAE,
    Directional Accuracy, Regime Performance, Drawdown, OOS, Walk-Forward.
    """

    def __init__(
        self,
        outcome_store: OutcomeStore,
        performance_store: Any = None,  # PerformanceStore
        result_store: PersistedEvaluationResultStore | None = None,
    ) -> None:
        self._outcomes = outcome_store
        self._performance_store = performance_store
        self._result_store = result_store
        self._results: list[EvaluationResult] = []
        self._lock = RLock()

    def evaluate_agent(
        self,
        agent_id: str,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Compute all evaluation metrics for an agent."""
        if run_id:
            outcomes = [o for o in self._outcomes.by_run(run_id) if o.agent_id == agent_id]
        else:
            outcomes = self._outcomes.by_agent(agent_id)

        if not outcomes:
            return {
                "agent_id": agent_id,
                "run_id": run_id,
                "observations": 0,
                "metrics": {},
            }

        metrics = _compute_classification_metrics(outcomes)
        mfe_mae = _compute_mfe_mae(outcomes)

        result = {
            "agent_id": agent_id,
            "run_id": run_id,
            "observations": metrics.total,
            "metrics": {
                "brier_score": round(metrics.brier_score, 6),
                "calibration_error": round(metrics.calibration_error, 6),
                "expectancy": round(metrics.expectancy, 6),
                "directional_accuracy": round(metrics.directional_accuracy, 6),
                **mfe_mae,
                "precision": (
                    metrics.true_positives / (metrics.true_positives + metrics.false_positives)
                    if (metrics.true_positives + metrics.false_positives) > 0
                    else 0.0
                ),
                "recall": (
                    metrics.true_positives / (metrics.true_positives + metrics.false_negatives)
                    if (metrics.true_positives + metrics.false_negatives) > 0
                    else 0.0
                ),
                "f1_score": self._f1_score(
                    metrics.true_positives,
                    metrics.false_positives,
                    metrics.false_negatives,
                ),
            },
            "confusion_matrix": {
                "tp": metrics.true_positives,
                "fp": metrics.false_positives,
                "tn": metrics.true_negatives,
                "fn": metrics.false_negatives,
            },
        }

        # Store as EvaluationResult
        eval_result = EvaluationResult(
            run_id=run_id or "",
            agent_id=agent_id,
            metric_name="aggregate_evaluation",
            metric_value=metrics.brier_score,
            observations=metrics.total,
            details=result["metrics"],  # type: ignore[arg-type]
        )
        with self._lock:
            self._results.append(eval_result)
        if self._result_store:
            self._result_store.add(eval_result)

        return result

    def evaluate_regime_performance(
        self,
        agent_id: str,
        regime: MarketRegime,
    ) -> dict[str, Any]:
        """Compute metrics for an agent within a specific market regime."""
        outcomes = self._outcomes.by_regime(regime)
        agent_outcomes = [o for o in outcomes if o.agent_id == agent_id]

        if not agent_outcomes:
            return {
                "agent_id": agent_id,
                "regime": regime.value,
                "observations": 0,
                "metrics": {},
            }

        metrics = _compute_classification_metrics(agent_outcomes)
        pnl_values = [o.realized_pnl for o in agent_outcomes]

        return {
            "agent_id": agent_id,
            "regime": regime.value,
            "observations": len(agent_outcomes),
            "metrics": {
                "brier_score": round(metrics.brier_score, 6),
                "directional_accuracy": round(metrics.directional_accuracy, 6),
                "expectancy": round(metrics.expectancy, 6),
                "total_pnl": round(sum(pnl_values), 6),
                "avg_pnl": round(sum(pnl_values) / len(pnl_values), 6),
            },
        }

    def evaluate_drawdown(
        self,
        agent_id: str,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Compute drawdown metrics for an agent."""
        if run_id:
            outcomes = [o for o in self._outcomes.by_run(run_id) if o.agent_id == agent_id]
        else:
            outcomes = self._outcomes.by_agent(agent_id)

        if not outcomes:
            return {
                "agent_id": agent_id,
                "drawdown": 0.0,
                "max_drawdown": 0.0,
                "recovery_periods": 0,
            }

        # Sort by timestamp and compute cumulative PnL equity curve
        sorted_outcomes = sorted(outcomes, key=lambda o: o.timestamp)
        equity = [0.0]
        cumulative = 0.0
        for o in sorted_outcomes:
            cumulative += o.realized_pnl
            equity.append(cumulative)

        # Compute drawdown
        peak = equity[0]
        max_drawdown = 0.0
        current_drawdown = 0.0

        for value in equity:
            peak = max(peak, value)
            drawdown = (peak - value) / peak if peak > 0 else 0.0
            current_drawdown = drawdown
            max_drawdown = max(max_drawdown, drawdown)

        # Count recovery periods (drawdown > 10% that recovered)
        recovery_periods = 0
        in_drawdown = False
        for value in equity:
            drawdown = (peak - value) / peak if peak > 0 else 0.0
            if drawdown > 0.10 and not in_drawdown:
                in_drawdown = True
            elif in_drawdown and drawdown <= 0.05:
                in_drawdown = False
                recovery_periods += 1

        return {
            "agent_id": agent_id,
            "run_id": run_id,
            "observations": len(outcomes),
            "max_drawdown": round(max_drawdown, 6),
            "current_drawdown": round(current_drawdown, 6),
            "recovery_periods": recovery_periods,
            "peak_equity": round(peak, 6),
            "final_equity": round(equity[-1], 6),
        }

    def evaluate_out_of_sample(
        self,
        agent_id: str,
        train_outcomes: list[OutcomeRecord],
        test_outcomes: list[OutcomeRecord],
    ) -> dict[str, Any]:
        """Evaluate out-of-sample performance.

        Compares in-sample (train) metrics to out-of-sample (test) metrics.
        A model that is overfit will show significantly better train metrics.
        """
        train_metrics = _compute_classification_metrics(train_outcomes)
        test_metrics = _compute_classification_metrics(test_outcomes)

        # OOS degradation ratio
        if train_metrics.brier_score > 0:
            brier_degradation = test_metrics.brier_score / train_metrics.brier_score
        else:
            brier_degradation = 0.0

        if train_metrics.directional_accuracy > 0:
            accuracy_degradation = (
                test_metrics.directional_accuracy / train_metrics.directional_accuracy
            )
        else:
            accuracy_degradation = 0.0

        # OOS pass: test metrics should not degrade by more than 50%
        oos_pass = (
            brier_degradation <= 2.0  # Brier score can double
            and accuracy_degradation >= 0.5  # Accuracy shouldn't halve
            and test_metrics.total > 0  # Must have test data
        )

        return {
            "agent_id": agent_id,
            "train_observations": train_metrics.total,
            "test_observations": test_metrics.total,
            "train_brier_score": round(train_metrics.brier_score, 6),
            "test_brier_score": round(test_metrics.brier_score, 6),
            "brier_degradation": round(brier_degradation, 6),
            "train_directional_accuracy": round(train_metrics.directional_accuracy, 6),
            "test_directional_accuracy": round(test_metrics.directional_accuracy, 6),
            "accuracy_degradation": round(accuracy_degradation, 6),
            "oos_pass": oos_pass,
        }

    def evaluate_walk_forward(
        self,
        agent_id: str,
        outcomes: list[OutcomeRecord],
        window_size: int = 50,
        step_size: int = 10,
    ) -> dict[str, Any]:
        """Evaluate walk-forward stability.

        Slides a window across outcomes and computes metrics for each window.
        Returns per-window results and overall stability score.
        """
        if len(outcomes) < window_size * 2:
            return {
                "agent_id": agent_id,
                "stable": False,
                "reason": "INSUFFICIENT_DATA",
                "windows": [],
            }

        sorted_outcomes = sorted(outcomes, key=lambda o: o.timestamp)
        windows: list[WalkForwardResult] = []
        window_id = 0

        i = 0
        while i + window_size * 2 <= len(sorted_outcomes):
            train = sorted_outcomes[i : i + window_size]
            test = sorted_outcomes[i + window_size : i + window_size * 2]

            train_metrics = _compute_classification_metrics(train)
            test_metrics = _compute_classification_metrics(test)

            # Stability = 1 - |train - test| / train (normalized)
            if train_metrics.brier_score > 0:
                stability = 1.0 - abs(
                    test_metrics.brier_score - train_metrics.brier_score
                ) / train_metrics.brier_score
            else:
                stability = 0.0

            wf_result = WalkForwardResult(
                window_id=f"wf-{window_id}",
                train_start=train[0].timestamp.isoformat() if train else "",
                train_end=train[-1].timestamp.isoformat() if train else "",
                test_start=test[0].timestamp.isoformat() if test else "",
                test_end=test[-1].timestamp.isoformat() if test else "",
                metric_name="brier_score",
                train_metric=round(train_metrics.brier_score, 6),
                test_metric=round(test_metrics.brier_score, 6),
                stability=round(stability, 6),
            )
            windows.append(wf_result)
            window_id += 1
            i += step_size

        if windows:
            avg_stability = sum(w.stability for w in windows) / len(windows)
            stable = avg_stability >= 0.5  # At least 50% stability
        else:
            avg_stability = 0.0
            stable = False

        return {
            "agent_id": agent_id,
            "stable": stable,
            "avg_stability": round(avg_stability, 6),
            "num_windows": len(windows),
            "windows": [w.model_dump() for w in windows],
        }

    def _f1_score(self, tp: int, fp: int, fn: int) -> float:
        """Compute F1 score."""
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def get_results(self, agent_id: str | None = None) -> list[EvaluationResult]:
        with self._lock:
            if agent_id:
                return [r for r in self._results if r.agent_id == agent_id]
            return list(self._results)