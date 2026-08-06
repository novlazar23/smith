"""Drawdown monitoring — peak equity tracking and gate evaluation."""

from __future__ import annotations

from .base import RiskGateResult, RiskGateType


class DrawdownMonitor:
    """Überwacht Drawdowns und evaluiert Risk-Gates."""

    def __init__(
        self,
        max_drawdown_pct: float = 0.15,
        warning_drawdown_pct: float = 0.10,
    ) -> None:
        self._max_drawdown_pct = max_drawdown_pct
        self._warning_drawdown_pct = warning_drawdown_pct
        self._peak_equity: float = 0.0
        self._current_equity: float = 0.0

    def update_equity(self, current_equity: float) -> None:
        """Aktualisiert den aktuellen Equity-Wert und den Peak."""
        self._current_equity = current_equity
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

    def current_drawdown(self) -> float:
        """Berechnet den aktuellen Drawdown als Bruchteil des Peaks."""
        if self._peak_equity == 0:
            return 0.0
        return (self._peak_equity - self._current_equity) / self._peak_equity

    def check_drawdown(self, current_equity: float) -> RiskGateResult:
        """Prüft ob der Drawdown innerhalb der Limits liegt."""
        self.update_equity(current_equity)
        dd = self.current_drawdown()

        if dd >= self._max_drawdown_pct:
            return RiskGateResult(
                gate_type=RiskGateType.DRAWDOWN,
                passed=False,
                severity="hard",
                blocking_reasons=[
                    f"Drawdown {dd:.4%} exceeds hard limit {self._max_drawdown_pct:.4%}"
                ],
            )

        if dd >= self._warning_drawdown_pct:
            return RiskGateResult(
                gate_type=RiskGateType.DRAWDOWN,
                passed=False,
                severity="soft",
                blocking_reasons=[
                    f"Drawdown {dd:.4%} exceeds warning limit {self._warning_drawdown_pct:.4%}"
                ],
                reduction_factor=0.5,
            )

        return RiskGateResult(
            gate_type=RiskGateType.DRAWDOWN,
            passed=True,
            severity="soft",
        )

    def get_state(self) -> dict[str, float]:
        """Gibt den aktuellen Zustand des Monitors zurück."""
        return {
            "peak_equity": self._peak_equity,
            "current_equity": self._current_equity,
            "drawdown_pct": self.current_drawdown(),
        }
