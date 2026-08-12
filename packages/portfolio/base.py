"""Basis-Typen und Datenklassen für Portfolio Management.

Enthält:
- PortfolioType (TRADING, SAVINGS, HEDGE)
- Position mit PnL-Berechnung
- PortfolioSummary mit Exposure-Metriken
- PortfolioConfig für Konfiguration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


class PortfolioType(StrEnum):
    """Typ des Portfolios mit unterschiedlicher Risikotoleranz."""

    TRADING = "trading"  # aktives Trading, höhere Risikotoleranz
    SAVINGS = "savings"  # konservativ, Kapitalerhaltung
    HEDGE = "hedge"  # gehedgt, marktneutral


@dataclass(frozen=True)
class Position:
    """Eine einzelne Position innerhalb eines Portfolios."""

    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    weight: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0

    @property
    def market_value(self) -> float:
        """Berechnet den Marktwert der Position (absolut)."""
        return abs(self.quantity) * self.current_price

    def calculate_pnl(self) -> tuple[float, float]:
        """Berechnet PnL und PnL-Prozent aus avg vs. current Price.

        Long (quantity > 0): pnl = (current - avg) * quantity
        Short (quantity < 0): pnl = (avg - current) * abs(quantity)
        pnl_pct = (pnl / (abs(quantity) * avg_price)) * 100 if avg_price > 0 else 0.0
        """
        if self.quantity > 0:
            pnl = (self.current_price - self.avg_price) * self.quantity
        else:
            pnl = (self.avg_price - self.current_price) * abs(self.quantity)

        abs_avg_value = abs(self.quantity) * self.avg_price
        pnl_pct = (pnl / abs_avg_value) * 100 if abs_avg_value != 0 else 0.0

        return pnl, pnl_pct


@dataclass
class PortfolioSummary:
    """Zusammenfassung des gesamten Portfolios zu einem Zeitpunkt."""

    portfolio_id: str
    portfolio_type: PortfolioType
    total_equity: float
    total_position_value: float
    cash: float
    net_exposure: float  # long_exposure - short_exposure
    gross_exposure: float  # long_exposure + short_exposure
    leverage: float  # gross_exposure / total_equity
    num_positions: int
    top_position_weight: float
    portfolio_pnl: float
    positions: list[Position]
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def long_exposure(self) -> float:
        """Summe aller Long-Exposures."""
        return sum(
            pos.market_value for pos in self.positions if pos.quantity > 0
        )

    @property
    def short_exposure(self) -> float:
        """Summe aller Short-Exposures."""
        return sum(
            pos.market_value for pos in self.positions if pos.quantity < 0
        )


@dataclass
class PortfolioConfig:
    """Konfiguration für ein Portfolio."""

    portfolio_id: str
    portfolio_type: PortfolioType
    max_single_position_pct: float = 0.25
    max_sector_exposure_pct: float = 0.40
    max_net_long_pct: float = 0.80
    max_net_short_pct: float = -0.20
    max_gross_leverage: float = 2.0
    currency: str = "USD"

    def to_exposure_limits(self) -> ExposureLimits:
        """Konvertiert die Konfiguration in Exposure-Limits."""
        return ExposureLimits(
            max_single_position_pct=self.max_single_position_pct,
            max_sector_exposure_pct=self.max_sector_exposure_pct,
            max_net_long_pct=self.max_net_long_pct,
            max_net_short_pct=self.max_net_short_pct,
            max_gross_leverage=self.max_gross_leverage,
            portfolio_type=self.portfolio_type,
        )


@dataclass(frozen=True)
class ExposureLimits:
    """Immutable Exposure-Limits aus einer PortfolioConfig."""

    max_single_position_pct: float
    max_sector_exposure_pct: float
    max_net_long_pct: float
    max_net_short_pct: float
    max_gross_leverage: float
    portfolio_type: PortfolioType
