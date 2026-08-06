"""Domain — Core Domain Models des Trading Orchestrators.

Abstrahiert über die Schemas hinaus mit reichhaltigeren Entitäten,
Beziehungen und domänenspezifischen Methoden.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Instrument(BaseModel):
    """Ein handelbares Wertpapier (Asset).

    Beispiel: BTC/USDT, EUR/USD, AAPL.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(..., min_length=1, description="Handels-Symbol.")
    venue: str = Field(..., min_length=1, description="Handelsplatz.")
    asset_type: str = Field(..., description="asset_type: spot, future, option, etc.")
    tick_size: float = Field(gt=0, description="Minimale Preisänderung.")
    lot_size: float = Field(gt=0, description="Minimale Handelsmenge.")
    contract_size: float = Field(gt=0, default=1.0, description="Kontraktgröße.")


class Position(BaseModel):
    """Offene Position in einem Instrument.

    Repräsentiert die aktuelle Exposure eines Portfolios in einem Instrument.
    """

    model_config = ConfigDict(frozen=True)

    instrument: str
    venue: str
    side: str = Field(..., description="long / short / flat.")
    quantity: float = Field(..., description="Aktuelle Positionsgröße.")
    entry_price: float = Field(gt=0, description="Einstiegspreis.")
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    @model_validator(mode='after')
    def validate_position(self) -> Position:
        """Positiv: quantity >= 0, side muss konsistent."""
        if self.quantity < 0:
            raise ValueError("quantity must be >= 0")
        if self.side not in ("long", "short", "flat"):
            raise ValueError("side must be long, short, or flat")
        return self

    def notional(self, current_price: float) -> float:
        """Nominalwert der Position zum aktuellen Preis."""
        return abs(self.quantity) * current_price


class Portfolio(BaseModel):
    """Portfolio mit multipler Positionen.

    Verwaltet die Gesamtexposure, PnL-Tracking und Capital-Reservierung.
    """

    model_config = ConfigDict(frozen=True)

    portfolio_id: str = Field(..., min_length=1)
    total_equity: float = Field(gt=0, description="Gesamtkapital.")
    positions: dict[str, Position] = Field(
        default_factory=dict, description="instrument_key -> Position."
    )
    max_exposure_per_position: float = Field(
        default=0.25, ge=0, le=1, description="Max. Exposure pro Position (Anteil)."
    )
    max_total_exposure: float = Field(
        default=1.0, ge=0, le=1, description="Max. Gesamt-Exposure (Anteil)."
    )

    @property
    def total_unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions.values())

    @property
    def total_realized_pnl(self) -> float:
        return sum(p.realized_pnl for p in self.positions.values())

    @property
    def total_exposure_ratio(self) -> float:
        """Gesamt-Exposure als Anteil am Kapital."""
        if not self.total_equity:
            return 0.0
        total_notional = sum(
            p.quantity * self._current_price(instrument)
            for instrument, p in self.positions.items()
            if p.side != "flat"
        )
        return total_notional / self.total_equity

    @staticmethod
    def _current_price(instrument: str) -> float:
        """Fallback: Preis aus externer Quelle nachschlagen.

        Im echten System wird dies über den MarketDataService erfolgen.
        Für das MVP gibt es keine externe Abhängigkeit — daher 1.0.
        """
        return 1.0

    def has_position(self, instrument_key: str) -> bool:
        return instrument_key in self.positions and self.positions[instrument_key].side != "flat"

    def get_position(self, instrument_key: str) -> Position | None:
        pos = self.positions.get(instrument_key)
        if pos and pos.side != "flat":
            return pos
        return None


class Trade(BaseModel):
    """Ausgeführter Trade (Order execution)."""

    model_config = ConfigDict(frozen=True)

    trade_id: str = Field(..., min_length=1)
    portfolio_id: str
    instrument: str
    venue: str
    price: float = Field(gt=0)
    quantity: float = Field(gt=0)
    side: str = Field(..., description="buy / sell.")
    commission: float = Field(ge=0, default=0.0)
    executed_at: datetime = Field(default_factory=datetime.utcnow)


class Order(BaseModel):
    """Offene Order (nicht ausgeführt)."""

    model_config = ConfigDict(frozen=True)

    order_id: str = Field(..., min_length=1)
    portfolio_id: str
    instrument: str
    venue: str
    price: float = Field(gt=0)
    quantity: float = Field(gt=0)
    side: str = Field(..., description="buy / sell.")
    order_type: str = Field(default="limit", description="limit / market / stop.")
    status: str = Field(default="pending", description="pending / filled / cancelled / rejected.")
    created_at: datetime = Field(default_factory=datetime.utcnow)


__all__ = [
    "Instrument",
    "Order",
    "Portfolio",
    "Position",
    "Trade",
]
