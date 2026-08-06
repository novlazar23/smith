"""Trading Configuration - Risiko- und Trading-spezifische Konfiguration.

Enthält Risikolimits, Position Sizing, Handelsmodi und Orchestrator-Einstellungen.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RiskConfig(BaseModel):
    """Risikomanagement-Konfiguration."""

    max_drawdown: float = Field(
        default=0.10, ge=0.0, le=1.0, description="Max. Drawdown (10% = 0.10)."
    )
    max_position_size: float = Field(
        default=0.25, ge=0.0, le=1.0, description="Max. Position als %-Anteil des Portfolios."
    )
    max_total_exposure: float = Field(
        default=1.0, ge=0.0, le=2.0, description="Max. Gesamt-Exposure (1.0 = 100% Capital)."
    )
    max_open_positions: int = Field(
        default=10, ge=1, description="Max. gleichzeitige offene Positionen."
    )
    max_single_trade_risk: float = Field(
        default=0.02, ge=0.0, le=1.0, description="Max. Risiko pro Trade (2% = 0.02)."
    )
    kelly_fraction: float = Field(
        default=0.25, ge=0.0, le=1.0, description="Kelly-Fraktion für Position Sizing."
    )
    min_confidence_threshold: float = Field(
        default=0.55, ge=0.0, le=1.0, description="Min. Agenten-Konfidenz für Trade."
    )

    @property
    def max_drawdown_pct(self) -> float:
        """Drawdown als Prozentwert."""
        return self.max_drawdown * 100

    @property
    def max_single_trade_risk_pct(self) -> float:
        """Risiko pro Trade als Prozentwert."""
        return self.max_single_trade_risk * 100


class TradingConfig(BaseModel):
    """Trading-Konfiguration."""

    mode: str = Field(
        default="paper",
        description="Modus: research, backtest, paper, live.",
    )
    risk: RiskConfig = Field(default_factory=RiskConfig)
    default_timeframes: list[str] = Field(
        default=["15m", "1h", "4h", "1d"],
        description="Standard-Zeitrahmen für Analyse.",
    )
    min_candles: int = Field(
        default=100, ge=10, description="Minimale Kerzen für Indikatoren-Berechnung."
    )
    rebalance_interval: str = Field(
        default="1h", description="Portfolio-Rebalancing-Intervall."
    )

    def validate_mode(self) -> bool:
        """Prüft gültigen Modus."""
        valid_modes = ("research", "backtest", "paper", "shadow", "live")
        return self.mode in valid_modes
