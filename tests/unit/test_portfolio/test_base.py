"""Tests fur packages.portfolio.base — Position, PortfolioSummary, PortfolioConfig."""

from __future__ import annotations

import pytest
from packages.portfolio.base import (
    ExposureLimits,
    PortfolioConfig,
    PortfolioSummary,
    PortfolioType,
    Position,
)


class TestPosition:
    """Testet die Position-Datenklasse."""

    def test_position_long_pnl(self) -> None:
        """Berechnet PnL fur eine Long-Position."""
        pos = Position(
            symbol="AAPL",
            quantity=100.0,
            avg_price=150.0,
            current_price=160.0,
        )
        pnl, pnl_pct = pos.calculate_pnl()
        assert pnl == 1000.0
        assert pnl_pct == pytest.approx(6.67, abs=0.01)

    def test_position_short_pnl(self) -> None:
        """Berechnet PnL fur eine Short-Position."""
        pos = Position(
            symbol="TSLA",
            quantity=-50.0,
            avg_price=200.0,
            current_price=180.0,
        )
        pnl, pnl_pct = pos.calculate_pnl()
        assert pnl == 1000.0
        assert pnl_pct == pytest.approx(10.0, abs=0.01)

    def test_position_market_value(self) -> None:
        """Berechnet den Marktwert einer Position."""
        pos = Position(
            symbol="BTC",
            quantity=2.5,
            avg_price=50000.0,
            current_price=52000.0,
        )
        assert pos.market_value == 130000.0

        pos_short = Position(
            symbol="MSFT",
            quantity=-10.0,
            avg_price=300.0,
            current_price=290.0,
        )
        assert pos_short.market_value == 2900.0

    def test_position_weight_calculation(self) -> None:
        """Gewicht = (abs(quantity) * current_price) / portfolio_value."""
        pos = Position(
            symbol="AAPL",
            quantity=100.0,
            avg_price=150.0,
            current_price=160.0,
            weight=0.25,
        )
        assert pos.weight == pytest.approx(0.25)

    def test_position_pnl_pct(self) -> None:
        """Berechnet PnL-Prozent korrekt."""
        pos = Position(
            symbol="GOOG",
            quantity=20.0,
            avg_price=2800.0,
            current_price=2660.0,
        )
        pnl, pnl_pct = pos.calculate_pnl()
        assert pnl == -2800.0
        assert pnl_pct == pytest.approx(-5.0, abs=0.01)

    def test_position_pnl_pct_zero_avg(self) -> None:
        """PnL bei avg_price = 0 ergibt 0."""
        pos = Position(
            symbol="ZERO",
            quantity=10.0,
            avg_price=0.0,
            current_price=5.0,
        )
        pnl, pnl_pct = pos.calculate_pnl()
        assert pnl == 50.0
        assert pnl_pct == 0.0


class TestPortfolioType:
    """Testet PortfolioType-Enum."""

    def test_all_portfolio_types(self) -> None:
        assert str(PortfolioType.TRADING) == "trading"
        assert str(PortfolioType.SAVINGS) == "savings"
        assert str(PortfolioType.HEDGE) == "hedge"

    def test_portfolio_type_values(self) -> None:
        assert PortfolioType.TRADING == "trading"
        assert PortfolioType.SAVINGS == "savings"
        assert PortfolioType.HEDGE == "hedge"


class TestPortfolioSummary:
    """Testet PortfolioSummary."""

    def test_summary_fields(self) -> None:
        """Pruft dass alle Felder existieren."""
        positions = [
            Position(
                symbol="AAPL",
                quantity=100.0,
                avg_price=150.0,
                current_price=160.0,
                weight=0.25,
                pnl=1000.0,
                pnl_pct=6.67,
            )
        ]
        summary = PortfolioSummary(
            portfolio_id="pf-001",
            portfolio_type=PortfolioType.TRADING,
            total_equity=40000.0,
            total_position_value=25600.0,
            cash=14400.0,
            net_exposure=25600.0,
            gross_exposure=25600.0,
            leverage=0.64,
            num_positions=1,
            top_position_weight=0.25,
            portfolio_pnl=1000.0,
            positions=positions,
        )
        assert summary.portfolio_id == "pf-001"
        assert summary.portfolio_type == PortfolioType.TRADING
        assert summary.total_equity == 40000.0
        assert summary.num_positions == 1
        assert summary.timestamp is not None

    def test_summary_long_short_exposure(self) -> None:
        """Pruft long/short Exposure Berechnung."""
        positions = [
            Position(symbol="AAPL", quantity=100.0, avg_price=150.0, current_price=160.0),
            Position(symbol="MSFT", quantity=-50.0, avg_price=300.0, current_price=290.0),
        ]
        summary = PortfolioSummary(
            portfolio_id="pf-002",
            portfolio_type=PortfolioType.TRADING,
            total_equity=50000.0,
            total_position_value=30100.0,
            cash=19900.0,
            net_exposure=0.0,
            gross_exposure=30100.0,
            leverage=0.602,
            num_positions=2,
            top_position_weight=0.40,
            portfolio_pnl=500.0,
            positions=positions,
        )
        assert summary.long_exposure == 16000.0
        assert summary.short_exposure == 14500.0
        assert summary.net_exposure == 0.0
        assert summary.long_exposure - summary.short_exposure == pytest.approx(1500.0)

    def test_summary_only_long_positions(self) -> None:
        """Nur Long-Positionen -> short_exposure = 0."""
        positions = [
            Position(symbol="AAPL", quantity=100.0, avg_price=150.0, current_price=160.0),
            Position(symbol="GOOG", quantity=10.0, avg_price=3000.0, current_price=3100.0),
        ]
        summary = PortfolioSummary(
            portfolio_id="pf-003",
            portfolio_type=PortfolioType.SAVINGS,
            total_equity=60000.0,
            total_position_value=47000.0,
            cash=13000.0,
            net_exposure=47000.0,
            gross_exposure=47000.0,
            leverage=0.783,
            num_positions=2,
            top_position_weight=0.4,
            portfolio_pnl=1000.0,
            positions=positions,
        )
        assert summary.long_exposure == pytest.approx(47000.0)
        assert summary.short_exposure == 0.0

    def test_summary_frozen_position(self) -> None:
        """Position ist gefroren (frozen=True)."""
        pos = Position(symbol="AAPL", quantity=1.0, avg_price=100.0, current_price=101.0)
        # frozen dataclass sollte nicht verandert werden konnen
        with pytest.raises(Exception):
            pos.quantity = 999.0  # type: ignore[attr-defined]


class TestPortfolioConfig:
    """Testet PortfolioConfig."""

    def test_defaults(self) -> None:
        cfg = PortfolioConfig(portfolio_id="pf-001", portfolio_type=PortfolioType.TRADING)
        assert cfg.portfolio_id == "pf-001"
        assert cfg.portfolio_type == PortfolioType.TRADING
        assert cfg.max_single_position_pct == 0.25
        assert cfg.max_sector_exposure_pct == 0.40
        assert cfg.max_net_long_pct == 0.80
        assert cfg.max_gross_leverage == 2.0
        assert cfg.currency == "USD"

    def test_custom_values(self) -> None:
        cfg = PortfolioConfig(
            portfolio_id="pf-002",
            portfolio_type=PortfolioType.SAVINGS,
            max_single_position_pct=0.10,
            max_gross_leverage=1.0,
            currency="EUR",
        )
        assert cfg.max_single_position_pct == 0.10
        assert cfg.max_gross_leverage == 1.0
        assert cfg.currency == "EUR"

    def test_to_exposure_limits(self) -> None:
        cfg = PortfolioConfig(
            portfolio_id="pf-001",
            portfolio_type=PortfolioType.TRADING,
            max_single_position_pct=0.20,
        )
        limits = cfg.to_exposure_limits()
        assert limits.max_single_position_pct == 0.20
        assert limits.max_net_long_pct == 0.80
        assert limits.portfolio_type == PortfolioType.TRADING
        assert isinstance(limits, ExposureLimits)

    def test_exposure_limits_are_frozen(self) -> None:
        cfg = PortfolioConfig(portfolio_id="pf-001", portfolio_type=PortfolioType.TRADING)
        limits = cfg.to_exposure_limits()
        with pytest.raises(Exception):
            limits.max_gross_leverage = 999.0  # type: ignore[attr-defined]


class TestExposureLimits:
    """Testet ExposureLimits."""

    def test_all_fields_present(self) -> None:
        limits = ExposureLimits(
            max_single_position_pct=0.25,
            max_sector_exposure_pct=0.40,
            max_net_long_pct=0.80,
            max_net_short_pct=-0.20,
            max_gross_leverage=2.0,
            portfolio_type=PortfolioType.TRADING,
        )
        assert limits.max_single_position_pct == 0.25
        assert limits.max_sector_exposure_pct == 0.40
        assert limits.max_gross_leverage == 2.0

    def test_limits_immutability(self) -> None:
        limits = ExposureLimits(
            max_single_position_pct=0.25,
            max_sector_exposure_pct=0.40,
            max_net_long_pct=0.80,
            max_net_short_pct=-0.20,
            max_gross_leverage=2.0,
            portfolio_type=PortfolioType.TRADING,
        )
        with pytest.raises(Exception):
            limits.max_single_position_pct = 0.99  # type: ignore[attr-defined]

    def test_portfolio_type_held(self) -> None:
        limits = ExposureLimits(
            max_single_position_pct=0.25,
            max_sector_exposure_pct=0.40,
            max_net_long_pct=0.80,
            max_net_short_pct=-0.20,
            max_gross_leverage=2.0,
            portfolio_type=PortfolioType.HEDGE,
        )
        assert limits.portfolio_type == PortfolioType.HEDGE
