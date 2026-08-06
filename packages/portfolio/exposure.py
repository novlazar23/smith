"""ExposureManager — Überwacht und validiert Portfolio-Exposure-Limits.

Prüft beim Hinzufügen einer Position:
- Single-Position-Limit (max weight)
- Netto-Lang/Short-Limits
- Gross-Leverage-Limit
"""

from __future__ import annotations

from .base import PortfolioType, Position


class ExposureManager:
    """Verwaltet und prüft Exposure-Limits für ein Portfolio."""

    def __init__(
        self,
        max_single_position_pct: float = 0.25,
        max_sector_exposure_pct: float = 0.40,
        max_net_long_pct: float = 0.80,
        max_net_short_pct: float = -0.20,
        max_gross_leverage: float = 2.0,
        portfolio_type: PortfolioType = PortfolioType.TRADING,
    ) -> None:
        self.max_single_position_pct = max_single_position_pct
        self.max_sector_exposure_pct = max_sector_exposure_pct
        self.max_net_long_pct = max_net_long_pct
        self.max_net_short_pct = max_net_short_pct
        self.max_gross_leverage = max_gross_leverage
        self.portfolio_type = portfolio_type

    def add_position(
        self,
        positions: list[Position],
        total_equity: float,
        new_position: Position,
    ) -> tuple[bool, list[str], float]:
        """Prüft ob das Hinzufügen einer Position Limits verletzen würde.

        Args:
            positions: Aktuelle Positionen im Portfolio.
            total_equity: Gesamtwert des Portfolios.
            new_position: Neue Position die geprüft werden soll.

        Returns:
            Tuple aus (erlaubt, blockierende_gründe, maximal_groesse).
        """
        if total_equity <= 0:
            raise ValueError("total_equity must be positive")

        new_weight = (abs(new_position.quantity) * new_position.current_price) / total_equity
        # Berechne Exposure nach Hinzufügen der Position
        existing_long = sum(
            pos.market_value for pos in positions if pos.quantity >= 0
        )
        existing_short = sum(
            pos.market_value for pos in positions if pos.quantity < 0
        )

        # Bestimme ob die neue Position Long oder Short ist
        if new_position.quantity >= 0:
            projected_long = existing_long + new_position.market_value
            projected_short = existing_short
        else:
            projected_long = existing_long
            projected_short = existing_short + new_position.market_value

        projected_net_long = projected_long
        projected_net_short = projected_short
        projected_gross = projected_long + projected_short
        projected_leverage = projected_gross / total_equity

        blocking_reasons: list[str] = []
        calculated_max: float = total_equity  # default: alles ist erlaubt

        # 1. Single-Position-Limit prüfen
        if new_weight > self.max_single_position_pct:
            max_size = self.max_single_position_pct * total_equity
            return (False, ["single_position_exceeded"], max_size)

        # 2. Netto-Lang-Limit prüfen
        if projected_net_long / total_equity > self.max_net_long_pct:
            max_allowed_long = self.max_net_long_pct * total_equity
            current_other = projected_long - new_position.market_value
            max_additional_long = max_allowed_long - current_other
            calculated_max = max_additional_long if max_additional_long > 0 else 0.0
            blocking_reasons.append("net_exposure_exceeded")

        # 3. Netto-Short-Limit prüfen
        if projected_net_short / total_equity < self.max_net_short_pct:
            max_allowed_short = abs(self.max_net_short_pct) * total_equity
            current_other = projected_short - new_position.market_value
            max_additional_short = max_allowed_short - current_other
            if max_additional_short > 0:
                calculated_max = min(calculated_max, max_additional_short)
            else:
                calculated_max = 0.0
            blocking_reasons.append("net_exposure_exceeded")

        # 4. Gross-Leverage prüfen
        if projected_leverage > self.max_gross_leverage:
            max_gross = self.max_gross_leverage * total_equity
            current_other_gross = projected_gross - new_position.market_value
            max_additional_gross = max_gross - current_other_gross
            if max_additional_gross > 0:
                calculated_max = min(calculated_max, max_additional_gross)
            else:
                calculated_max = 0.0
            if "leverage_exceeded" not in blocking_reasons:
                blocking_reasons.append("leverage_exceeded")

        if blocking_reasons:
            return (False, blocking_reasons, calculated_max)

        return (True, [], calculated_max)

    def get_exposure_report(
        self, positions: list[Position], total_equity: float
    ) -> dict[str, float]:
        """Generiert einen Exposure-Bericht für das aktuelle Portfolio.

        Args:
            positions: Aktuelle Positionen.
            total_equity: Gesamtwert des Portfolios.

        Returns:
            Dict mit Exposure-Kennzahlen.
        """
        if total_equity <= 0:
            raise ValueError("total_equity must be positive")

        long_exposure = sum(pos.market_value for pos in positions if pos.quantity > 0)
        short_exposure = sum(pos.market_value for pos in positions if pos.quantity < 0)
        net_exposure = long_exposure - short_exposure
        gross_exposure = long_exposure + short_exposure
        leverage = gross_exposure / total_equity

        weights = [pos.weight for pos in positions if pos.weight > 0]
        top_weight = max(weights) if weights else 0.0

        return {
            "net_exposure_pct": net_exposure / total_equity,
            "gross_exposure_pct": gross_exposure / total_equity,
            "leverage": leverage,
            "long_exposure_pct": long_exposure / total_equity,
            "short_exposure_pct": short_exposure / total_equity,
            "num_positions": float(len(positions)),
            "top_weight": top_weight,
        }

    def check_all_limits(
        self, positions: list[Position], total_equity: float
    ) -> dict[str, bool]:
        """Prüft alle Limits für das aktuelle Portfolio.

        Args:
            positions: Aktuelle Positionen.
            total_equity: Gesamtwert des Portfolios.

        Returns:
            Dict mit limit_name -> passed (bool).
        """
        if total_equity <= 0:
            raise ValueError("total_equity must be positive")

        long_exposure = sum(pos.market_value for pos in positions if pos.quantity > 0)
        short_exposure = sum(pos.market_value for pos in positions if pos.quantity < 0)
        gross_exposure = long_exposure + short_exposure

        # Single Position Limit
        max_weight = max((pos.weight for pos in positions), default=0.0)
        single_position_ok = max_weight <= self.max_single_position_pct

        # Net Long Limit
        net_long_ok = (long_exposure / total_equity) <= self.max_net_long_pct

        # Net Short Limit
        net_short_ok = (short_exposure / total_equity) >= abs(self.max_net_short_pct)

        # Gross Leverage
        leverage = gross_exposure / total_equity
        gross_leverage_ok = leverage <= self.max_gross_leverage

        # Total Exposure
        total_exposure = (long_exposure + short_exposure) / total_equity
        total_exposure_ok = total_exposure <= self.max_gross_leverage * 1.2

        return {
            "single_position": single_position_ok,
            "net_long": net_long_ok,
            "net_short": net_short_ok,
            "gross_leverage": gross_leverage_ok,
            "total_exposure": total_exposure_ok,
        }
