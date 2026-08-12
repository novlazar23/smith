"""Backtest evaluation worker — computes performance metrics from trade lists."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class BacktestEvaluator:
    """Evaluates backtest results and computes performance metrics."""

    def evaluate(
        self,
        trades: list[dict[str, Any]],
        initial_capital: float = 100000.0,
    ) -> dict[str, Any]:
        """Evaluates a list of trades and returns computed performance metrics.

        Each trade dict must contain: symbol, direction, entry_price, exit_price,
        quantity, timestamp, commission, pnl.

        Args:
            trades: List of trade dicts to evaluate.
            initial_capital: Starting capital for return calculations.

        Returns:
            Dict with computed metrics including win_rate, profit_factor,
            sharpe_ratio, max_drawdown, return_pct, and a enriched trade_list.
        """
        if not trades:
            return self._empty_result(initial_capital)

        # Sort trades chronologically
        sorted_trades = sorted(trades, key=lambda t: t["timestamp"])

        # Enrich trades with computed metrics
        enriched_trades: list[dict[str, Any]] = []
        for trade in sorted_trades:
            pnl = trade.get("pnl", 0.0)
            enriched: dict[str, Any] = dict(trade)
            enriched["computed_pnl"] = pnl
            enriched["is_winner"] = pnl > 0.0
            enriched["is_loser"] = pnl < 0.0
            enriched_trades.append(enriched)

        total_trades = len(enriched_trades)
        winning_trades = sum(1 for t in enriched_trades if t["is_winner"])
        losing_trades = sum(1 for t in enriched_trades if t["is_loser"])

        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        wins = [t["computed_pnl"] for t in enriched_trades if t["is_winner"]]
        losses = [abs(t["computed_pnl"]) for t in enriched_trades if t["is_loser"]]

        total_pnl = sum(t["computed_pnl"] for t in enriched_trades)
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0

        total_wins = sum(wins) if wins else 0.0
        total_losses = sum(losses) if losses else 0.0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

        avg_trade_pnl = total_pnl / total_trades if total_trades > 0 else 0.0

        # Build equity curve from trades in chronological order
        equity_curve: list[float] = [initial_capital]
        for t in enriched_trades:
            equity_curve.append(equity_curve[-1] + t["computed_pnl"])

        # Max drawdown from equity curve
        max_drawdown = self._compute_max_drawdown(equity_curve)

        # Sharpe ratio (annualized)
        sharpe_ratio = self._compute_sharpe_ratio(equity_curve)

        # Return percentage
        final_equity = equity_curve[-1]
        return_pct = (final_equity - initial_capital) / initial_capital * 100.0

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": round(win_rate, 6),
            "total_pnl": round(total_pnl, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 6),
            "avg_trade_pnl": round(avg_trade_pnl, 2),
            "max_drawdown": round(max_drawdown, 6),
            "sharpe_ratio": round(sharpe_ratio, 6),
            "return_pct": round(return_pct, 2),
            "trade_list": enriched_trades,
        }

    def generate_report(self, evaluation: dict[str, Any]) -> str:
        """Generates a text report from evaluation results.

        Args:
            evaluation: Dict as returned by evaluate().

        Returns:
            Formatted string with key metrics.
        """
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("BACKTEST EVALUATION REPORT")
        lines.append("=" * 60)
        lines.append("")

        lines.append("--- TRADE SUMMARY ---")
        lines.append(f"  Total Trades:    {evaluation.get('total_trades', 0)}")
        lines.append(f"  Winning Trades:  {evaluation.get('winning_trades', 0)}")
        lines.append(f"  Losing Trades:   {evaluation.get('losing_trades', 0)}")
        lines.append("")

        lines.append("--- PERFORMANCE ---")
        lines.append(f"  Win Rate:        {evaluation.get('win_rate', 0.0):.2%}")
        lines.append(f"  Total PnL:       {evaluation.get('total_pnl', 0.0):.2f}")
        lines.append(f"  Avg Win:         {evaluation.get('avg_win', 0.0):.2f}")
        lines.append(f"  Avg Loss:        {evaluation.get('avg_loss', 0.0):.2f}")
        lines.append(f"  Avg Trade PnL:   {evaluation.get('avg_trade_pnl', 0.0):.2f}")
        lines.append(f"  Return %:        {evaluation.get('return_pct', 0.0):.2f}%")
        lines.append("")

        lines.append("--- RISK METRICS ---")
        lines.append(f"  Profit Factor:   {evaluation.get('profit_factor', 0.0):.4f}")
        lines.append(f"  Max Drawdown:    {evaluation.get('max_drawdown', 0.0):.4%}")
        lines.append(f"  Sharpe Ratio:    {evaluation.get('sharpe_ratio', 0.0):.4f}")
        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    @staticmethod
    def _compute_max_drawdown(equity_curve: list[float]) -> float:
        """Computes peak-to-trough maximum drawdown from an equity curve."""
        if len(equity_curve) < 2:
            return 0.0

        equity_arr = np.array(equity_curve, dtype=np.float64)
        running_max = np.maximum.accumulate(equity_arr)
        drawdowns = (running_max - equity_arr) / running_max
        drawdowns = np.where(running_max == 0, 0.0, drawdowns)
        return float(np.max(drawdowns))

    @staticmethod
    def _compute_sharpe_ratio(equity_curve: list[float]) -> float:
        """Computes annualized Sharpe ratio from an equity curve.

        Uses daily returns: mean(daily_returns) / std(daily_returns) * sqrt(252).
        Returns 0.0 if std is zero or not enough data.
        """
        if len(equity_curve) < 3:
            return 0.0

        equity_arr = np.array(equity_curve, dtype=np.float64)
        daily_returns = np.diff(equity_arr) / equity_arr[:-1]
        daily_returns = np.where(np.isfinite(daily_returns), daily_returns, 0.0)

        if len(daily_returns) < 2:
            return 0.0

        mean_ret = float(np.mean(daily_returns))
        std_ret = float(np.std(daily_returns, ddof=1))

        if std_ret == 0.0:
            return 0.0

        return mean_ret / std_ret * float(np.sqrt(252))

    @staticmethod
    def _empty_result(initial_capital: float) -> dict[str, Any]:
        """Returns default metrics for an empty trade list."""
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "avg_trade_pnl": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "return_pct": 0.0,
            "trade_list": [],
        }


class ScoringEngine:
    """Scores agent predictions using Brier score, Log Loss, and classification metrics."""

    @staticmethod
    def compute_brier_score(
        predictions: list[float],
        actuals: list[int],
    ) -> float:
        """Compute Brier score for probabilistic predictions.

        Brier score = mean((predicted - actual)^2)
        Lower is better (0 = perfect, 1 = worst).

        Args:
            predictions: List of predicted probabilities in [0, 1].
            actuals: List of actual outcomes (0 or 1).

        Returns:
            Brier score.
        """
        if len(predictions) != len(actuals):
            raise ValueError("predictions and actuals must have same length")
        if not predictions:
            return 0.0

        scores = [(p - a) ** 2 for p, a in zip(predictions, actuals, strict=True)]
        return float(np.mean(scores))

    @staticmethod
    def compute_log_loss(
        predictions: list[float],
        actuals: list[int],
        eps: float = 1e-15,
    ) -> float:
        """Compute logarithmic loss for probabilistic predictions.

        Args:
            predictions: List of predicted probabilities in [0, 1].
            actuals: List of actual outcomes (0 or 1).
            eps: Epsilon to clip predictions to avoid log(0).

        Returns:
            Log loss value.
        """
        if len(predictions) != len(actuals):
            raise ValueError("predictions and actuals must have same length")
        if not predictions:
            return 0.0

        preds_arr = np.array(predictions, dtype=np.float64)
        actuals_arr = np.array(actuals, dtype=np.float64)
        clipped = np.clip(preds_arr, eps, 1.0 - eps)
        log_losses = -(actuals_arr * np.log(clipped) + (1 - actuals_arr) * np.log(1 - clipped))
        return float(np.mean(log_losses))

    @staticmethod
    def compute_confusion_matrix(
        predictions: list[int],
        actuals: list[int],
    ) -> dict[str, int]:
        """Compute confusion matrix counts.

        Args:
            predictions: List of predicted class labels (0 or 1).
            actuals: List of actual class labels (0 or 1).

        Returns:
            Dict with tp, fp, tn, fn counts.
        """
        if len(predictions) != len(actuals):
            raise ValueError("predictions and actuals must have same length")

        tp = sum(1 for p, a in zip(predictions, actuals, strict=True) if p == 1 and a == 1)
        fp = sum(1 for p, a in zip(predictions, actuals, strict=True) if p == 1 and a == 0)
        tn = sum(1 for p, a in zip(predictions, actuals, strict=True) if p == 0 and a == 0)
        fn = sum(1 for p, a in zip(predictions, actuals, strict=True) if p == 0 and a == 1)

        return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}

    @staticmethod
    def compute_precision_recall_f1(
        predictions: list[int],
        actuals: list[int],
    ) -> dict[str, float]:
        """Compute precision, recall, and F1 score.

        Args:
            predictions: List of predicted class labels.
            actuals: List of actual class labels.

        Returns:
            Dict with precision, recall, f1 scores.
        """
        cm = ScoringEngine.compute_confusion_matrix(predictions, actuals)
        tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        return {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }

    def score_agent(
        self,
        predictions: list[float],
        actuals: list[int],
        thresholds: list[int] | None = None,
    ) -> dict[str, Any]:
        """Score an agent's predictions comprehensively.

        Args:
            predictions: List of predicted probabilities.
            actuals: List of actual outcomes.
            thresholds: List of threshold values for classification scores.

        Returns:
            Dict with all scoring metrics.
        """
        brier = float(self.compute_brier_score(predictions, actuals))
        logloss = float(self.compute_log_loss(predictions, actuals))

        # Default threshold 0.5 for classification
        if thresholds is None:
            thresholds = [0.5]

        classification_results: dict[str, dict[str, Any]] = {}
        for threshold in thresholds:
            preds_binary = [1 if p >= threshold else 0 for p in predictions]
            metrics = self.compute_precision_recall_f1(preds_binary, actuals)
            cm = self.compute_confusion_matrix(preds_binary, actuals)
            classification_results[f"threshold_{threshold:.1f}"] = {
                **metrics,
                **cm,
                "accuracy": (cm["tp"] + cm["tn"]) / len(actuals) if actuals else 0.0,
            }

        return {
            "brier_score": round(brier, 6),
            "log_loss": round(logloss, 6),
            "n_samples": len(actuals),
            "classification": classification_results,
        }


class AgentEvaluator:
    """Evaluates and compares agent performance over time."""

    def evaluate_agent(
        self,
        agent_id: str,
        predictions: list[float],
        actuals: list[int],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluate a single agent's performance.

        Args:
            agent_id: Agent identifier.
            predictions: Predicted probabilities.
            actuals: Actual outcomes.
            metadata: Optional metadata about the evaluation window.

        Returns:
            Dict with agent evaluation results.
        """
        scoring = ScoringEngine()
        scores = scoring.score_agent(predictions, actuals)

        return {
            "agent_id": agent_id,
            "evaluation": scores,
            "metadata": metadata or {},
        }

    def compare_agents(
        self,
        agent_results: list[dict[str, Any]],
        metric: str = "brier_score",
    ) -> list[dict[str, Any]]:
        """Compare multiple agents by a scoring metric.

        Args:
            agent_results: List of agent evaluation dicts.
            metric: Metric to sort by (lower is better).

        Returns:
            Sorted list of agent results (best first).
        """
        sorted_results = sorted(
            agent_results,
            key=lambda r: r["evaluation"].get(metric, float("inf")),
        )

        for rank, result in enumerate(sorted_results, start=1):
            result["rank"] = rank
            result["metric_value"] = result["evaluation"].get(metric, 0.0)

        return sorted_results

    def champion_challenger(
        self,
        champion_id: str,
        challenger_id: str,
        champion_scores: dict[str, float],
        challenger_scores: dict[str, float],
        required_improvement: float = 0.01,
    ) -> dict[str, Any]:
        """Run a champion-challenger comparison.

        Args:
            champion_id: ID of the current champion agent.
            challenger_id: ID of the challenger agent.
            champion_scores: Dict of metrics for champion.
            challenger_scores: Dict of metrics for challenger.
            required_improvement: Minimum relative improvement needed.

        Returns:
            Dict with comparison result and promotion recommendation.
        """
        improvements: dict[str, bool] = {}
        for key in set(champion_scores) | set(challenger_scores):
            champion_val = champion_scores.get(key, 0.0)
            challenger_val = challenger_scores.get(key, 0.0)
            if champion_val == 0.0:
                improvements[key] = challenger_val < champion_val
            else:
                rel_change = (champion_val - challenger_val) / abs(champion_val)
                improvements[key] = rel_change > required_improvement

        overall_promote = all(improvements.values()) if improvements else False

        return {
            "champion_id": champion_id,
            "challenger_id": challenger_id,
            "champion_scores": champion_scores,
            "challenger_scores": challenger_scores,
            "improvements": improvements,
            "promote_challenger": overall_promote,
            "required_improvement": required_improvement,
        }


class ResolutionEngine:
    """Resolves expired predictions against market outcomes."""

    @staticmethod
    def resolve_prediction(
        prediction: dict[str, Any],
        outcome: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve a single prediction against the actual outcome.

        Args:
            prediction: Dict with prediction details (direction, confidence, expiry).
            outcome: Dict with actual outcome (price, direction, timestamp).

        Returns:
            Resolved prediction with accuracy and score.
        """
        pred_direction = prediction.get("direction", "")
        actual_direction = outcome.get("direction", "")
        confidence = prediction.get("confidence", 0.0)

        correct = pred_direction == actual_direction
        score = confidence if correct else (1.0 - confidence)

        return {
            "prediction_id": prediction.get("id", "unknown"),
            "direction": pred_direction,
            "confidence": confidence,
            "actual_direction": actual_direction,
            "correct": correct,
            "score": round(score, 6),
            "resolved_at": outcome.get("timestamp", ""),
        }

    def resolve_batch(
        self,
        predictions: list[dict[str, Any]],
        outcomes: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Resolve a batch of predictions against outcomes.

        Args:
            predictions: List of prediction dicts.
            outcomes: List of outcome dicts.

        Returns:
            List of resolved prediction dicts.
        """
        results: list[dict[str, Any]] = []
        for pred, outcome in zip(predictions, outcomes, strict=True):
            result = self.resolve_prediction(pred, outcome)
            results.append(result)

        # Aggregate statistics
        total = len(results)
        correct = sum(1 for r in results if r["correct"])
        avg_score = sum(r["score"] for r in results) / total if total > 0 else 0.0

        return {
            "resolved": results,
            "total": total,
            "correct": correct,
            "accuracy": round(correct / total, 6) if total > 0 else 0.0,
            "avg_score": round(avg_score, 6),
        }
