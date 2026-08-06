"""Position sizing — Kelly and ATR-based calculators."""

from __future__ import annotations

from .base import PositionSizerConfig


class KellyPositionSizer:
    """Kelly-Kriterium zur Positionsgrössen-Bestimmung.

    Verwendet die halbe Kelly-Formel (Half-Kelly) für mehr Konservatismus.
    """

    def __init__(self, config: PositionSizerConfig | None = None) -> None:
        self._config = config or PositionSizerConfig()

    def calculate_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        account_size: float,
    ) -> float:
        """Berechnet die absolute Positionsgrösse (in Geld)."""
        fraction = self.calculate_fraction(win_rate, avg_win, avg_loss)
        return fraction * account_size

    def calculate_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """Berechnet die Positionsgrösse als Bruchteil des Kontos.

        Kelly-Formel: f* = (p * b - q) / b
        Dabei: p = win_rate, q = 1 - win_rate, b = avg_win / avg_loss

        Verwendet Half-Kelly für mehr Konservatismus.
        """
        if avg_loss == 0:
            raise ValueError("avg_loss must be non-zero")

        p = win_rate
        q = 1.0 - p
        b = avg_win / avg_loss

        kelly = (p * b - q) / b

        # Half-Kelly für mehr Konservatismus
        kelly /= 2.0

        # Negative Edge → keine Position
        if kelly < 0.0:
            return 0.0

        # Clamp auf Config-Bereiche
        upper = min(self._config.max_position_size, self._config.base_risk_pct)
        return max(self._config.min_position_size, min(kelly, upper))


class ATRPositionSizer:
    """Positionsgrössen-Berechner basierend auf Average True Range."""

    def __init__(
        self,
        config: PositionSizerConfig | None = None,
        atr_multiplier: float = 2.0,
        max_atr_risk: float = 0.02,
    ) -> None:
        self._config = config or PositionSizerConfig()
        self.atr_multiplier = atr_multiplier
        self.max_atr_risk = max_atr_risk

    def calculate_size(
        self,
        atr: float,
        stop_distance_atr: int = 2,
        account_size: float = 10000.0,
    ) -> float:
        """Berechnet die Positionsgrösse aus ATR-Wert und Stop-Distanz.

        stop_distance = atr * stop_distance_atr
        position_size = (account_size * max_atr_risk) / atr

        Raises:
            ValueError: Wenn atr <= 0 ist.
        """
        if atr <= 0:
            raise ValueError("ATR must be positive")

        position_size = (account_size * self.max_atr_risk) / atr
        return position_size
