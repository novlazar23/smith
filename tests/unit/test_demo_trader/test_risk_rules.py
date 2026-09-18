"""Tests für die vier Portfolio-Risikoregeln des Demo-Traders (BUY-Seite).

Heat-Cap, Drawdown-Circuit-Breaker, Volatility-Scaling und
Cost-Margin-Gate — harte Grenzen für neue Positionen; SELLs,
Glattstellungen und Exit-Backstops bleiben davon unberührt.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from apps.demo_trader.service import (
    DEFAULT_MAX_DRAWDOWN_PCT,
    DEFAULT_MAX_PORTFOLIO_HEAT_PCT,
    DEFAULT_MIN_MOVE_COST_MULTIPLE,
    DemoTraderConfig,
    atr_pct,
    config_from_env,
    cost_margin_ok,
    heat_cap_ok,
    parse_horizon_minutes,
    portfolio_exposure,
    update_drawdown_guard,
    vol_size_factor,
)
from apps.orchestrator_service.service import CandleWindow
from packages.paper import PaperAccount, PaperPosition, TradeDirection

from .conftest import (
    BTC,
    FakeConnection,
    StubCandleSource,
    StubPipeline,
    make_ohlcv,
    make_result,
    make_trader,
)


def _window(close: np.ndarray, half_range: float) -> CandleWindow:
    """Kerzenfenster mit fester Halbbreite ``half_range`` um den Close."""
    n = len(close)
    return CandleWindow(
        open=close,
        high=close + half_range,
        low=close - half_range,
        close=close,
        volume=np.full(n, 1000.0),
        timestamps=np.arange(n, dtype=np.int64) * 300_000_000_000,
    )


def _regime_window(
    quiet_n: int, quiet_range: float, vol_n: int, vol_range: float
) -> CandleWindow:
    """Zweigeteiltes Fenster bei Close 100: erste Phase ruhig, zweite volatil."""
    n = quiet_n + vol_n
    close = np.full(n, 100.0)
    high = np.full(n, 100.0)
    low = np.full(n, 100.0)
    high[:quiet_n] += quiet_range
    low[:quiet_n] -= quiet_range
    high[quiet_n:] += vol_range
    low[quiet_n:] -= vol_range
    return CandleWindow(
        open=close,
        high=high,
        low=low,
        close=close,
        volume=np.full(n, 1000.0),
        timestamps=np.arange(n, dtype=np.int64) * 300_000_000_000,
    )


class TestAtrPct:
    """aktuelle ATR(14) als Anteil des letzten Closes (pure Funktion)."""

    def test_uniform_range(self) -> None:
        """Konstante Halbbreite 0,5 bei Close 100 → ATR% = 1,0 %."""
        window = _window(np.full(50, 100.0), 0.5)
        assert atr_pct(window) == pytest.approx(0.01)

    def test_short_window_returns_zero(self) -> None:
        """< period + 1 Kerzen → 0,0 (keine Daten)."""
        window = _window(np.full(14, 100.0), 0.5)
        assert atr_pct(window) == 0.0


class TestVolSizeFactor:
    """factor = clamp(Median-ATR% / aktuelle ATR%, 0.5, 1.0) (pure Funktion)."""

    def test_high_current_atr_scales_to_floor(self) -> None:
        """Ruhiges Fenster mit volatilen Schlusskerzen → Faktor 0,5 (Boden)."""
        window = _regime_window(180, 0.2, 20, 2.0)
        assert vol_size_factor(window) == pytest.approx(0.5)

    def test_low_current_atr_no_upscaling(self) -> None:
        """Volatiles Fenster mit ruhigen Schlusskerzen → Faktor 1,0 (nie vergrößern)."""
        window = _regime_window(180, 2.0, 20, 0.2)
        assert vol_size_factor(window) == pytest.approx(1.0)

    def test_uniform_window_factor_one(self) -> None:
        """Gleichbleibende Volatilität (current = Median) → Faktor 1,0."""
        window = _window(np.full(200, 100.0), 0.5)
        assert vol_size_factor(window) == pytest.approx(1.0)

    def test_boundary_ratio_half(self) -> None:
        """Current = 2x Median → exakt am Boden 0,5."""
        window = _regime_window(180, 0.2, 14, 0.4)
        assert vol_size_factor(window) == pytest.approx(0.5)

    def test_short_window_fail_open(self) -> None:
        """< period + 1 Kerzen → Faktor 1,0 (fail-open beim Sizing)."""
        window = _window(np.full(14, 100.0), 0.5)
        assert vol_size_factor(window) == 1.0


class TestCostMargin:
    """erwartete Bewegung ≥ multiple x 2 x trade_cost_pct (pure Funktion)."""

    def test_at_threshold_passes(self) -> None:
        """Exakt an der Schwelle (3,0 x 2 x 0,1 % = 0,6 %) → durch."""
        assert cost_margin_ok(0.006, 0.001, 3.0)

    def test_below_threshold_blocks(self) -> None:
        """Unter der Schwelle → blockiert (fail-closed)."""
        assert not cost_margin_ok(0.005, 0.001, 3.0)

    def test_above_threshold_passes(self) -> None:
        assert cost_margin_ok(0.01, 0.001, 3.0)

    def test_multiple_zero_always_ok(self) -> None:
        """multiple = 0 → Gate aus, auch bei null Bewegung."""
        assert cost_margin_ok(0.0, 0.001, 0.0)


class TestParseHorizonMinutes:
    """Horizon-Notation 'Nm' → Minuten; Fallback 15 mit Warning."""

    def test_minutes(self) -> None:
        assert parse_horizon_minutes("15m") == 15
        assert parse_horizon_minutes(" 30M ") == 30

    def test_unsupported_falls_back_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            assert parse_horizon_minutes("1h") == 15
        assert "Fallback 15" in caplog.text


def _account(cash: float, positions: dict[str, tuple[float, float]]) -> PaperAccount:
    """PaperAccount mit vorgegebenen Positionen (quantity, avg_price)."""
    account = PaperAccount(account_id="t", cash=cash, initial_cash=100000.0)
    for symbol, (quantity, avg_price) in positions.items():
        account.positions[symbol] = PaperPosition(
            symbol=symbol, quantity=quantity, avg_price=avg_price
        )
    return account


class TestHeatCap:
    """Exposure + Order-Nominal ≤ max_heat_pct x Equity (pure Funktion)."""

    def test_exposure_proxy(self) -> None:
        """Exposure = Σ quantity x avg_price (Kostengrundlage)."""
        account = _account(50000.0, {"A": (10.0, 100.0), "B": (2.0, 250.0)})
        assert portfolio_exposure(account) == pytest.approx(1500.0)

    def test_block_on_exceed(self) -> None:
        """Exposure 24000 + Notional 1001 > 0,25 x Equity 100000 → blockiert."""
        account = _account(76000.0, {"A": (240.0, 100.0)})
        assert not heat_cap_ok(account, 1001.0, 0.25)

    def test_at_threshold_passes(self) -> None:
        """Exakt an der Cap (24000 + 1000 = 25000) → durch."""
        account = _account(76000.0, {"A": (240.0, 100.0)})
        assert heat_cap_ok(account, 1000.0, 0.25)

    def test_zero_disables(self) -> None:
        """0 = Cap aus."""
        account = _account(1000.0, {"A": (100000.0, 1.0)})
        assert heat_cap_ok(account, 1.0, 0.0)


class TestDrawdownGuard:
    """State-Logik des Circuit-Breakers (pure Funktion, Hysterese)."""

    def test_peak_follows_new_highs(self) -> None:
        """Equity über dem Peak → Peak wird angehoben, kein Halt."""
        assert update_drawdown_guard(110.0, 100.0, halted=False, max_dd=0.15) == (110.0, False)

    def test_triggers_at_threshold(self) -> None:
        """Exakt am Trigger (85 % vom Peak bei 15 % max DD) → Halt."""
        assert update_drawdown_guard(85.0, 100.0, halted=False, max_dd=0.15) == (100.0, True)

    def test_below_threshold_halts(self) -> None:
        assert update_drawdown_guard(84.0, 100.0, halted=False, max_dd=0.15) == (100.0, True)

    def test_just_above_trigger_no_halt(self) -> None:
        assert update_drawdown_guard(85.5, 100.0, halted=False, max_dd=0.15) == (100.0, False)

    def test_no_rearm_in_middle(self) -> None:
        """Zwischen Trigger (85) und Re-Arm-Schwelle (92,5) → weiterhin Halt."""
        assert update_drawdown_guard(90.0, 100.0, halted=True, max_dd=0.15) == (100.0, True)

    def test_rearm_at_half_threshold(self) -> None:
        """Re-Arm exakt bei halber Schwelle (92,5 % vom Peak)."""
        assert update_drawdown_guard(92.5, 100.0, halted=True, max_dd=0.15) == (100.0, False)

    def test_still_halted_below_rearm(self) -> None:
        assert update_drawdown_guard(92.0, 100.0, halted=True, max_dd=0.15) == (100.0, True)

    def test_zero_disables(self) -> None:
        """0 = Breaker aus, auch aus dem Halt-Status."""
        assert update_drawdown_guard(10.0, 100.0, halted=True, max_dd=0.0) == (100.0, False)


class TestRiskRulesEndToEnd:
    """Integration durch DemoTrader (existierende Stubs, echter PaperExecutor)."""

    def test_cost_margin_blocks_buy_on_quiet_market(
        self, config: DemoTraderConfig, fake_conn: FakeConnection, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Flacher Markt (ATR = 0) → erwartete Bewegung 0 → Cost-Margin blockiert."""
        window = _window(np.full(50, 100.0), 0.0)
        provider = StubCandleSource({BTC: window})
        pipeline = StubPipeline({BTC: make_result(decision="LONG_BIAS", confidence=0.6)})
        trader = make_trader(config, provider, fake_conn, pipeline)

        with caplog.at_level("INFO"):
            executed = trader._run_instrument(BTC)

        assert executed == 0
        assert fake_conn.executed == []
        assert BTC not in trader.account.positions
        assert "Cost-Margin" in caplog.text

    def test_cost_margin_disabled_with_zero_multiple(
        self, config: DemoTraderConfig, fake_conn: FakeConnection
    ) -> None:
        """multiple = 0 → Gate aus, ruhiger Markt kauft trotzdem."""
        config = replace(config, min_move_cost_multiple=0.0)
        window = _window(np.full(50, 100.0), 0.0)
        provider = StubCandleSource({BTC: window})
        pipeline = StubPipeline({BTC: make_result(decision="LONG_BIAS", confidence=0.6)})
        trader = make_trader(config, provider, fake_conn, pipeline)

        assert trader._run_instrument(BTC) == 1
        assert BTC in trader.account.positions

    def test_drawdown_halt_blocks_buy_but_not_sell(
        self, config: DemoTraderConfig, fake_conn: FakeConnection, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Equity-Crash -20 % → BUY wird vom Breaker gestoppt,
        SHORT_BIAS-Glattstellung läuft trotzdem."""
        provider = StubCandleSource({BTC: make_ohlcv(100, start_price=100.0)})
        pipeline = StubPipeline({BTC: make_result(decision="LONG_BIAS", confidence=0.6)})
        trader = make_trader(config, provider, fake_conn, pipeline)
        trader.account.cash = 80000.0  # Equity -20 % gegenüber initial_cash
        trader._executor.submit_order(trader.account, BTC, TradeDirection.BUY, 10.0, 100.0)

        with caplog.at_level("INFO"):
            executed = trader.run_cycle()

        assert executed == 0
        assert "Drawdown-Breaker aktiv" in caplog.text
        assert BTC in trader.account.positions

        pipeline.results = {BTC: make_result(decision="SHORT_BIAS", confidence=0.6)}
        executed = trader.run_cycle()

        assert executed == 1
        assert BTC not in trader.account.positions
        sell_rows = [params for (_stmt, params) in fake_conn.executed if "direction" in params]
        assert len(sell_rows) == 1
        assert sell_rows[0]["direction"] == "SELL"

    def test_vol_scale_reduces_quantity(
        self, config: DemoTraderConfig, fake_conn: FakeConnection
    ) -> None:
        """Volatiles Schlussfenster (ATR ≈ 10x Median) → Menge auf Faktor 0,5."""
        window = _regime_window(180, 0.2, 20, 2.0)
        provider = StubCandleSource({BTC: window})
        pipeline = StubPipeline({BTC: make_result(decision="LONG_BIAS", confidence=0.6)})
        trader = make_trader(config, provider, fake_conn, pipeline)

        executed = trader._run_instrument(BTC)

        assert executed == 1
        base_quantity = 2000.0 / float(window.close[-1])
        assert fake_conn.executed[0][1]["quantity"] == pytest.approx(base_quantity * 0.5)

    def test_heat_cap_blocks_buy(
        self, config: DemoTraderConfig, fake_conn: FakeConnection, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Offene Position (~20 % der Equity) + neuer Buy > Heat-Cap → kein Kauf."""
        config = replace(config, flat_size=False)
        window = _window(np.full(50, 100.0), 0.5)
        provider = StubCandleSource({BTC: window})
        pipeline = StubPipeline({BTC: make_result(decision="LONG_BIAS", confidence=0.6)})
        trader = make_trader(config, provider, fake_conn, pipeline)
        # Zwei Vor-Käufe: das 10 %-Positions-Limit des PaperExecutors deckelt
        # jede Order auf ~10 % der Equity → Exposure ≈ 20 % der Equity.
        trader._executor.submit_order(trader.account, BTC, TradeDirection.BUY, 200.0, 100.0)
        trader._executor.submit_order(trader.account, BTC, TradeDirection.BUY, 200.0, 100.0)
        quantity_before = trader.account.positions[BTC].quantity

        with caplog.at_level("INFO"):
            executed = trader.run_cycle()

        assert executed == 0
        assert "Heat-Cap" in caplog.text
        assert trader.account.positions[BTC].quantity == pytest.approx(quantity_before)


class TestRiskRulesFromEnv:
    """config_from_env liest die vier Risiko-Env-Variablen mit Defaults."""

    @pytest.fixture(autouse=True)
    def _clean_risk_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for key in (
            "DEMO_MAX_PORTFOLIO_HEAT_PCT",
            "DEMO_MAX_DRAWDOWN_PCT",
            "DEMO_VOL_SCALE",
            "DEMO_MIN_MOVE_COST_MULTIPLE",
        ):
            monkeypatch.delenv(key, raising=False)

    def test_defaults(self) -> None:
        """Ohne Env-Variablen gelten die vorgegebenen Defaults."""
        config = config_from_env()
        assert config.max_portfolio_heat_pct == DEFAULT_MAX_PORTFOLIO_HEAT_PCT
        assert config.max_drawdown_pct == DEFAULT_MAX_DRAWDOWN_PCT
        assert config.vol_scale is True
        assert config.min_move_cost_multiple == DEFAULT_MIN_MOVE_COST_MULTIPLE

    def test_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Env-Variablen überschreiben die Defaults (inkl. 0 = aus)."""
        monkeypatch.setenv("DEMO_MAX_PORTFOLIO_HEAT_PCT", "0.10")
        monkeypatch.setenv("DEMO_MAX_DRAWDOWN_PCT", "0.10")
        monkeypatch.setenv("DEMO_VOL_SCALE", "false")
        monkeypatch.setenv("DEMO_MIN_MOVE_COST_MULTIPLE", "0")

        config = config_from_env()

        assert config.max_portfolio_heat_pct == 0.10
        assert config.max_drawdown_pct == 0.10
        assert config.vol_scale is False
        assert config.min_move_cost_multiple == 0.0

    def test_invalid_float_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ungültiges DEMO_MAX_DRAWDOWN_PCT → Default statt Absturz."""
        monkeypatch.setenv("DEMO_MAX_DRAWDOWN_PCT", "viel")

        config = config_from_env()

        assert config.max_drawdown_pct == DEFAULT_MAX_DRAWDOWN_PCT
