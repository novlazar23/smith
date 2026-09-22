"""Tests für den CPCV/DSR/PBO-Wrapper (apps.backtest.cv).

Deterministisch, ohne ClickHouse: MemoryDataFeed + synthetische Kerzen
aus ``conftest.make_candles``.
"""

from __future__ import annotations

import math
from datetime import timedelta

import numpy as np
import pytest
from apps.backtest import __main__ as cli
from apps.backtest.cv import _compress_indices, equity_returns, run_cpcv, run_dsr, run_pbo
from packages.backtesting.core import BacktestConfig, BacktestResult, Candle
from packages.backtesting.strategies import BaseStrategy, SignalAction, StrategySignal
from tests.unit.test_backtest.conftest import BTC, make_candles


class OnceLongStrategy(BaseStrategy):
    """Kauft einmalig (Flatsize) nach Warmup + Delay und hält bis zum Ende.

    Deterministische Test-Strategie: ein BUY, dann Ruhe — jede Instanz
    ist unabhängig (Strategien sind zustandsbehaftet).
    """

    def __init__(
        self,
        symbol: str = BTC,
        delay: int = 0,
        size: float = 0.02,
        candle_limit: int = 20,
    ) -> None:
        super().__init__(name="once_long")
        self.instrument = symbol
        self.candle_limit = candle_limit
        self.initial_capital = 100_000.0
        self.trade_notional = 100_000.0 * size
        self._delay = delay
        self._bars = 0
        self._bought = False

    def on_bar(self, candle: Candle) -> StrategySignal | None:
        self._bars += 1
        if self._bought or self._bars <= self.candle_limit + self._delay:
            return None
        self._bought = True
        return StrategySignal(
            action=SignalAction.BUY,
            symbol=candle.symbol,
            confidence=0.9,
            reason="once-long",
            position_size=self.trade_notional / self.initial_capital,
            timestamp=candle.timestamp,
        )


def _test_blocks(indices: list[int]) -> list[list[int]]:
    """Zerlegt sortierte Test-Indizes in zusammenhängende Blöcke."""
    sorted_idx = sorted(indices)
    blocks: list[list[int]] = []
    run = [sorted_idx[0]]
    for idx in sorted_idx[1:]:
        if idx == run[-1] + 1:
            run.append(idx)
        else:
            blocks.append(run)
            run = [idx]
    blocks.append(run)
    return blocks


def _decompress(intervals: list[list[int]]) -> list[int]:
    """Entpackt ``[start, stop)``-Intervalle zu einer flachen Indexliste."""
    out: list[int] = []
    for start, stop in intervals:
        out.extend(range(start, stop))
    return out


# ── equity_returns ──────────────────────────────────────────────────────────


def test_equity_returns_maps_curve_to_bars() -> None:
    """Equity-Kurve (Länge 1+n-warmup) → Per-Bar-Returns an den End-Bars."""
    config = BacktestConfig(warmup_bars=2)
    candles = make_candles(5)
    result = BacktestResult(
        config=config,
        candles=candles,
        snapshots=[],
        trades=[],
        metrics={},
        metadata={"equity_curve": [100.0, 110.0, 105.0, 107.0]},
    )
    out = equity_returns(result)
    assert [ts for ts, _ in out] == [
        candles[2].timestamp,
        candles[3].timestamp,
        candles[4].timestamp,
    ]
    assert out[0][1] == pytest.approx(0.10)
    assert out[1][1] == pytest.approx(-5.0 / 110.0)
    assert out[2][1] == pytest.approx(2.0 / 105.0)


def test_equity_returns_skips_bars_beyond_candles() -> None:
    """Kurven-Einträge jenseits der Kerzen (bar >= len(candles)) werden übersprungen."""
    config = BacktestConfig(warmup_bars=0)
    candles = make_candles(3)
    result = BacktestResult(
        config=config,
        candles=candles,
        snapshots=[],
        trades=[],
        metrics={},
        # 5 Einträge → Returns für Bar 0..3; nur Bar 0..2 existieren.
        metadata={"equity_curve": [100.0, 110.0, 120.0, 130.0, 140.0]},
    )
    out = equity_returns(result)
    assert len(out) == 3  # i=4 → Bar 3 existiert nicht (len(candles) = 3)
    assert [ts for ts, _ in out] == [
        candles[0].timestamp,
        candles[1].timestamp,
        candles[2].timestamp,
    ]
    assert out[0][1] == pytest.approx(0.10)
    assert out[1][1] == pytest.approx(10.0 / 110.0)
    assert out[2][1] == pytest.approx(10.0 / 120.0)


# ── _compress_indices ───────────────────────────────────────────────────────


def test_compress_indices_roundtrip_sorted() -> None:
    idx = list(range(50)) + list(range(100, 150))
    assert _compress_indices(idx) == [[0, 50], [100, 150]]
    assert _decompress(_compress_indices(idx)) == idx


def test_compress_indices_roundtrip_unsorted() -> None:
    idx = [3, 0, 1, 9, 7, 2, 8, 4, 5, 6]
    assert _decompress(_compress_indices(idx)) == sorted(idx)
    assert _compress_indices(idx) == [[0, 10]]


def test_compress_indices_single_and_empty() -> None:
    assert _compress_indices([7]) == [[7, 8]]
    assert _compress_indices([]) == []


# ── run_cpcv ────────────────────────────────────────────────────────────────


def test_run_cpcv_folds_disjoint_and_purged() -> None:
    """800 Kerzen, N=4, K=2 → 6 Folds; Train/Test disjunkt; Purge eingehalten."""
    candles = make_candles(800, step=0.05)
    config = BacktestConfig(max_holding_bars=12)
    out = run_cpcv(
        candles, lambda: OnceLongStrategy(), config, {}, n_splits=4, n_test_groups=2, bar_seconds=60
    )
    assert out["expected_folds"] == 6  # C(4,2)
    assert out["n_folds"] == 6
    assert out["skipped"] == []
    times = [c.timestamp for c in candles]
    horizon = timedelta(minutes=12)
    for fold in out["folds"]:
        train = set(_decompress(fold["train_indices"]))
        test = _decompress(fold["test_indices"])
        assert not train & set(test)
        assert fold["n_train"] == len(train)
        assert fold["n_test"] == len(test)
        for key in ("is_sharpe", "is_total_return_pct", "oos_sharpe", "oos_total_return_pct"):
            value = fold[key]
            assert isinstance(value, float)
            assert math.isfinite(value)
        assert isinstance(fold["oos_final_equity"], float)
        assert math.isfinite(fold["oos_final_equity"])
        for block in _test_blocks(test):
            b_start, b_end = times[block[0]], times[block[-1]]
            for idx in train:
                ts = times[idx]
                assert not (b_start - horizon <= ts < b_start), f"Train zu nah vor Test-Block {block}"
                assert not (b_end < ts <= b_end + horizon), f"Train zu nah nach Test-Block {block}"


def test_run_cpcv_skips_too_short_slices() -> None:
    """Purge-Horizont > Fenster → alle Folds zu kurz → skipped mit Reason."""
    candles = make_candles(400, step=0.05)
    config = BacktestConfig(max_holding_bars=300)  # H = 300 1m-Bars ≈ ganzes Fenster
    out = run_cpcv(
        candles, lambda: OnceLongStrategy(), config, {}, n_splits=4, n_test_groups=2, bar_seconds=60
    )
    assert out["n_folds"] == 0
    assert len(out["skipped"]) == 6
    assert all("reason" in entry for entry in out["skipped"])


# ── run_pbo ─────────────────────────────────────────────────────────────────


def test_run_pbo_zoo() -> None:
    candles = make_candles(400, step=0.1)
    zoo = [
        ("a", lambda: OnceLongStrategy(delay=0, size=0.02)),
        ("b", lambda: OnceLongStrategy(delay=25, size=0.03)),
        ("c", lambda: OnceLongStrategy(delay=50, size=0.01)),
    ]
    out = run_pbo(candles, zoo, BacktestConfig(), {}, n_splits=16)
    assert out["pbo"] is not None
    assert isinstance(out["pbo"], float)
    assert 0.0 <= out["pbo"] <= 1.0
    assert out["n_configs"] == 3
    assert out["n_splits"] == 16
    assert set(out["per_config"]) == {"a", "b", "c"}
    for entry in out["per_config"].values():
        assert entry["n_obs"] == 400 - 20  # n_candles - candle_limit
        assert math.isfinite(entry["sharpe"])


def test_run_pbo_single_config() -> None:
    candles = make_candles(200, step=0.1)
    out = run_pbo(candles, [("a", lambda: OnceLongStrategy())], BacktestConfig(), {})
    assert out["pbo"] is None
    assert out["reason"] == "need >=2 configs"


# ── run_dsr ─────────────────────────────────────────────────────────────────


def test_run_dsr_bounds_and_ordering() -> None:
    rng = np.random.default_rng(42)
    returns = rng.normal(0.005, 0.01, 300)
    trial_sharpes = [float(s) for s in np.linspace(-0.3, 0.4, 10)]
    out = run_dsr(returns, 10, trial_sharpes)
    assert out["dsr"] is not None
    assert 0.0 <= out["dsr"] <= 1.0
    assert out["n_trials"] == 10
    assert out["sr_hat"] == pytest.approx(float(returns.mean() / returns.std(ddof=1)))
    assert out["var_sharpe"] == pytest.approx(float(np.var(trial_sharpes, ddof=1)))
    out_100 = run_dsr(returns, 100, trial_sharpes)
    # Mehr Trials → höherer Deflations-Benchmark → niedrigere DSR
    assert out_100["dsr"] < out["dsr"]


def test_run_dsr_needs_two_trial_sharpes() -> None:
    rng = np.random.default_rng(7)
    returns = rng.normal(0.001, 0.01, 100)
    out = run_dsr(returns, 1, [0.5])
    assert out["dsr"] is None
    assert out["reason"] == "need >=2 trial sharpes"
    assert out["var_sharpe"] is None


# ── argparse / CLI-Integration ──────────────────────────────────────────────


def test_parser_accepts_cpcv_flags() -> None:
    args = cli.build_parser().parse_args(
        [
            "--cpcv",
            "--strategy",
            "ema_cross",
            "--cpcv-splits",
            "6",
            "--cpcv-test-groups",
            "2",
            "--cpcv-zoo",
            "--n-trials",
            "12",
        ]
    )
    assert args.cpcv is True
    assert args.cpcv_splits == 6
    assert args.cpcv_test_groups == 2
    assert args.cpcv_zoo is True
    assert args.n_trials == 12


def test_validate_cpcv_requires_strategy() -> None:
    args = cli.build_parser().parse_args(["--cpcv"])
    with pytest.raises(SystemExit):
        cli.validate_cpcv_args(cli.build_parser(), args)


def test_validate_cpcv_rejects_sweep_gates() -> None:
    args = cli.build_parser().parse_args(
        ["--cpcv", "--strategy", "ema_cross", "--sweep-gates", "0.3,0.5"]
    )
    with pytest.raises(SystemExit):
        cli.validate_cpcv_args(cli.build_parser(), args)


def test_validate_cpcv_rejects_sweep_library() -> None:
    args = cli.build_parser().parse_args(["--cpcv", "--strategy", "ema_cross", "--sweep-library"])
    with pytest.raises(SystemExit):
        cli.validate_cpcv_args(cli.build_parser(), args)


def test_validate_cpcv_accepts_champion_with_params() -> None:
    args = cli.build_parser().parse_args(
        [
            "--cpcv",
            "--strategy",
            "rsi_mean_reversion",
            "--params",
            "period=30,buy_below=20,sell_above=80",
            "--resample",
            "5m",
            "--from",
            "2021-01-01",
            "--to",
            "2026-01-01",
        ]
    )
    cli.validate_cpcv_args(cli.build_parser(), args)  # darf nicht werfen
