"""Historical weight tracker for agent accuracy-based adjustments.

Tracks prediction accuracy per agent over a sliding lookback window and
computes exponential-decay-weighted accuracy to adjust agent weights.
"""

from __future__ import annotations

from datetime import datetime

from .base import VoteDirection


class HistoricalWeightTracker:
    """Verwahrt historische Vorhersagegenauigkeit pro Agent."""

    def __init__(
        self,
        lookback: int = 50,
        accuracy_decay: float = 0.1,
    ) -> None:
        self.lookback = lookback
        self.accuracy_decay = accuracy_decay
        # agent_id -> list of (datetime, predicted, actual, correct)
        self._history: dict[str, list[tuple[datetime, VoteDirection, VoteDirection, bool]]] = {}

    def record_prediction(
        self,
        agent_id: str,
        direction: VoteDirection,
        outcome: VoteDirection,
    ) -> None:
        """Erstellt einen Eintrag für eine Vorhersage und ihr Ergebnis.

        Args:
            agent_id: ID des Agenten.
            direction: Vom Agenten vorhergesagte Richtung.
            outcome: Tatsächliches Ergebnis.
        """
        correct = direction == outcome
        entry = (datetime.now(), direction, outcome, correct)

        if agent_id not in self._history:
            self._history[agent_id] = []

        self._history[agent_id].append(entry)

        # Enforce lookback limit
        history = self._history[agent_id]
        if len(history) > self.lookback:
            self._history[agent_id] = history[-self.lookback :]

    def get_accuracy(self, agent_id: str) -> float:
        """Berechnet die gewichtete Genauigkeit über den Lookback-Window.

        Jedes vergangen Eintrag trägt bei: 1.0 bei Treffer, 0.0 bei Fehltrager.
        Gewichtung erfolgt exponentiell: weight = accuracy_decay ** (lookback - index).

        Args:
            agent_id: ID des Agenten.

        Returns:
            Gewichtete Genauigkeit im Bereich [0.0, 1.0]. Gibt 1.0 zurück
            wenn keine Historie vorhanden ist.
        """
        history = self._history.get(agent_id, [])
        if not history:
            return 1.0

        decay = self.accuracy_decay
        total_weighted_correct = 0.0
        total_weight = 0.0

        for i, (_, _, _, correct) in enumerate(history):
            weight = decay ** (self.lookback - 1 - i)
            if correct:
                total_weighted_correct += weight
            total_weight += weight

        if total_weight == 0.0:
            return 1.0

        return total_weighted_correct / total_weight

    def get_weight_adjustment(self, agent_id: str) -> float:
        """Berechnet die Gewichtsanpassung basierend auf der historischen Genauigkeit.

        adjustment = 0.5 + accuracy * 0.5 (Bereich 0.5 bis 1.0).

        Args:
            agent_id: ID des Agenten.

        Returns:
            Gewichtsanpassung im Bereich [0.0, 1.0].
        """
        accuracy = self.get_accuracy(agent_id)
        adjustment = 0.5 + accuracy * 0.5
        return max(0.0, min(1.0, adjustment))

    def get_agent_stats(self, agent_id: str) -> dict:
        """Gibt Statistiken für einen bestimmten Agenten zurück.

        Args:
            agent_id: ID des Agenten.

        Returns:
            Dict mit accuracy, total_predictions, correct, incorrect.
        """
        history = self._history.get(agent_id, [])
        total = len(history)
        correct = sum(1 for _, _, _, c in history if c)
        incorrect = total - correct
        accuracy = correct / total if total > 0 else 0.0

        return {
            "accuracy": accuracy,
            "total_predictions": total,
            "correct": correct,
            "incorrect": incorrect,
        }
