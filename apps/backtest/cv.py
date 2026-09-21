"""Überfittungs-Validierung: CPCV, PBO und DSR für Backtests (purgedcv 0.1.6).

Der ``--cpcv``-Modus des Backtest-CLI prüft einen Champion-Kandidaten
(Bibliotheks-Strategie) vor einer Deployment-Entscheidung auf
Backtest-Overfitting:

- **CPCV** (Combinatorial Purged Cross-Validation, AIFML Kap. 12.4):
  Das Fenster wird in ``n_splits`` zusammenhängende Blöcke aufgeteilt;
  jede Kombination von ``n_test_groups`` Blöcken ist ein Fold
  (C(N,K) Folds). Purge-Horizont = maximale Haltedauer
  (``max_holding_bars`` x Bar-Dauer, Default 2016 Bars), zusätzlich als
  Embargo nach jedem Test-Block. Als ``evaluation_times`` dient
  Entscheidungszeitpunkt + maximale Haltedauer (Label-Horizont), damit
  Purge Trainingszeilen entfernt, deren Positionshorizont mit dem
  Test-Fenster überlappen würde.
- **PBO** (Probability of Backtest Overfitting, Bailey et al. 2017,
  J. Comput. Finance): Anteil der CSCV-Kombinationen, bei denen die
  in-sample beste Konfiguration unter dem OOS-Median landet;
  berechnet auf der Per-Bar-Return-Matrix des Strategie-Zoos
  (Champion + Bibliotheks-Strategien).
- **DSR** (Deflated Sharpe Ratio, Bailey & López de Prado 2014,
  AIFML Kap. 11): PSR der Per-Bar-Returns des Champions gegen den
  Benchmark SR*_n, der von der Trial-Zahl (``n_trials``) und der
  Varianz der Trial-Sharpes abhängt.

Splits und Statistik kommen aus dem PyPI-Paket ``purgedcv`` (0.1.6,
eslazarev, MIT) — keine Reimplementierung der Formeln. Alle hier
berichteten Sharpes sind **per-Bar** (mean/std, ddof=1), nicht
annualisiert.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from math import comb
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from packages.backtesting.core import BacktestConfig, Candle
from packages.backtesting.datafeed import MemoryDataFeed
from packages.backtesting.engine import BacktestEngine
from packages.backtesting.strategies import BaseStrategy
from purgedcv import (
    CombinatorialPurgedCV,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)

from .runner import default_config

if TYPE_CHECKING:
    from packages.backtesting.core import BacktestResult

logger = logging.getLogger(__name__)

#: Default-Purge-Horizont in Bars, wenn ``config.max_holding_bars`` unset
#: ist: 2016 = 7 Tage bei 5m-Auflösung.
DEFAULT_HOLDING_BARS: int = 2016


def equity_returns(result: BacktestResult) -> list[tuple[datetime, float]]:
    """Per-Bar-Simple-Returns aus ``result.metadata["equity_curve"]``.

    Die Equity-Kurve hat Länge ``1 + n_candles - warmup_bars``: Index 0
    ist das Startkapital, Index i ≥ 1 die Marktbewertung bei
    ``candles[warmup_bars + i - 1]`` — der Return gehört also zur
    Bar, in der er endet.
    """
    curve = result.metadata.get("equity_curve") or []
    warmup = result.config.warmup_bars
    returns: list[tuple[datetime, float]] = []
    for i in range(1, len(curve)):
        bar = warmup + i - 1
        if bar < len(result.candles):
            returns.append(
                (result.candles[bar].timestamp, (curve[i] - curve[i - 1]) / curve[i - 1])
            )
    return returns


def _per_bar_sharpe(returns: np.ndarray) -> float:
    """Per-Bar-Sharpe (mean/std, ddof=1), nicht annualisiert; degeneriert → 0.0."""
    if returns.size < 2:
        return 0.0
    std = float(returns.std(ddof=1))
    if std == 0.0 or not np.isfinite(std):
        return 0.0
    return float(returns.mean() / std)


def _run_slice(
    slice_: Sequence[Candle],
    strategy: BaseStrategy,
    config: BacktestConfig | None,
    funding_rates: Mapping[datetime, float] | None,
) -> BacktestResult:
    """Voller Engine-Run über eine Kerzen-Slice (wie ``gate_sweep``)."""
    engine = BacktestEngine(default_config(strategy, config))
    return engine.run(
        MemoryDataFeed(list(slice_)),
        strategy,
        warmup_bars=strategy.candle_limit,
        funding_rates=funding_rates,
    )


def run_cpcv(
    candles: Sequence[Candle],
    strategy_factory: Callable[[], BaseStrategy],
    config: BacktestConfig | None,
    funding_rates: Mapping[datetime, float] | None,
    n_splits: int = 8,
    n_test_groups: int = 2,
    bar_seconds: int = 300,
) -> dict[str, Any]:
    """Führt CPCV über ein Kerzen-Fenster aus (AIFML Kap. 12.4).

    Pro Fold (jeweils C-Block-Kombination): IS-Run auf der
    Train-Slice, OOS-Run auf der Test-Slice — jeweils mit einer
    **frischen** Strategie-Instanz (Strategien sind zustandsbehaftet).
    Folds, deren Slice zu kurz ist (< candle_limit + 30 Bars), werden
    übersprungen und in ``"skipped"`` protokolliert.

    Returns:
        ``{"n_folds", "expected_folds", "folds", "skipped"}``; je Fold
        Fold-Index, Slice-Größen, IS/OOS-Metriken (``sharpe_ratio``/
        ``total_return_pct``/``total_trades`` aus ``result.metrics``,
        ``final_equity`` aus ``result.metadata``) sowie die
        Trainings-/Test-Indizes (Positionen in ``candles``).
    """
    times = pd.DatetimeIndex([c.timestamp for c in candles])
    hold_bars = (config.max_holding_bars if config is not None else None) or DEFAULT_HOLDING_BARS
    horizon = timedelta(seconds=hold_bars * bar_seconds)
    cv = CombinatorialPurgedCV(
        n_splits,
        n_test_groups,
        prediction_times=times,
        evaluation_times=times + horizon,
        purge_horizon=horizon,
        embargo=horizon,
    )
    expected_folds = comb(n_splits, n_test_groups)
    folds: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for fold_index, (train_idx, test_idx) in enumerate(cv.split(np.zeros(len(candles)))):
        train_slice = [candles[i] for i in train_idx]
        test_slice = [candles[i] for i in test_idx]
        is_strategy = strategy_factory()
        min_bars = is_strategy.candle_limit + 30
        if len(train_slice) < min_bars or len(test_slice) < min_bars:
            reason = f"Slice zu kurz (train={len(train_slice)}, test={len(test_slice)}; min. {min_bars} Bars)"
            skipped.append(
                {
                    "fold": fold_index,
                    "n_train": len(train_slice),
                    "n_test": len(test_slice),
                    "reason": reason,
                }
            )
            logger.warning("CPCV-Fold %d übersprungen: %s", fold_index, reason)
            continue
        is_result = _run_slice(train_slice, is_strategy, config, funding_rates)
        oos_result = _run_slice(test_slice, strategy_factory(), config, funding_rates)
        folds.append(
            {
                "fold": fold_index,
                "n_train": len(train_slice),
                "n_test": len(test_slice),
                "is_sharpe": is_result.metrics.get("sharpe_ratio"),
                "is_total_return_pct": is_result.metrics.get("total_return_pct"),
                "oos_sharpe": oos_result.metrics.get("sharpe_ratio"),
                "oos_total_return_pct": oos_result.metrics.get("total_return_pct"),
                "oos_total_trades": oos_result.metrics.get("total_trades"),
                "oos_final_equity": oos_result.metadata.get("final_equity"),
                "train_indices": [int(i) for i in train_idx],
                "test_indices": [int(i) for i in test_idx],
            }
        )
        logger.info(
            "CPCV-Fold %d/%s: train=%d, test=%d, IS-Sharpe=%s, OOS-Sharpe=%s, OOS-Return=%s",
            fold_index + 1,
            expected_folds,
            len(train_slice),
            len(test_slice),
            folds[-1]["is_sharpe"],
            folds[-1]["oos_sharpe"],
            folds[-1]["oos_total_return_pct"],
        )
    return {
        "n_folds": len(folds),
        "expected_folds": expected_folds,
        "folds": folds,
        "skipped": skipped,
    }


def run_pbo(
    candles: Sequence[Candle],
    strategies: Sequence[tuple[str, Callable[[], BaseStrategy]]],
    config: BacktestConfig | None,
    funding_rates: Mapping[datetime, float] | None,
    n_splits: int = 16,
) -> dict[str, Any]:
    """Führt den PBO über ein Strategie-Zoo aus (CSCV, purgedcv).

    Jede Strategie (Name, Factory) wird frisch über das **volle**
    Fenster gerechnet; die Per-Bar-Returns bilden eine
    ``(n_configs, n_obs)``-Matrix. Alle Strategien sehen dasselbe
    Fenster — unterschiedliche Längen sind ein Konstruktionsfehler
    (ValueError).

    Returns:
        ``{"pbo", "n_configs", "n_splits", "n_combos", "slope",
        "per_config": {name: {"sharpe", "total_return_pct", "n_obs"}}}``
        oder bei < 2 Konfigurationen ``{"pbo": None, "reason":
        "need >=2 configs", ...}``.
    """
    if len(strategies) < 2:
        return {
            "pbo": None,
            "reason": "need >=2 configs",
            "n_configs": len(strategies),
            "n_splits": n_splits,
            "per_config": {},
        }
    per_config: dict[str, dict[str, Any]] = {}
    rows: list[np.ndarray] = []
    for name, factory in strategies:
        strategy = factory()
        result = _run_slice(candles, strategy, config, funding_rates)
        bar_returns = np.array([ret for _ts, ret in equity_returns(result)], dtype=float)
        rows.append(bar_returns)
        per_config[name] = {
            "sharpe": _per_bar_sharpe(bar_returns),
            "total_return_pct": result.metrics.get("total_return_pct"),
            "n_obs": int(bar_returns.size),
        }
    n_obs = rows[0].size
    if any(row.size != n_obs for row in rows):
        raise ValueError(
            f"Per-Bar-Returns haben unterschiedliche Längen ({[row.size for row in rows]}) — "
            "alle Strategien müssen auf demselben Kerzen-Fenster laufen"
        )
    pbo_result = probability_of_backtest_overfitting(np.vstack(rows), n_splits)
    return {
        "pbo": float(pbo_result.pbo),
        "n_configs": len(strategies),
        "n_splits": n_splits,
        "n_combos": int(pbo_result.n_combos),
        "slope": float(pbo_result.slope),
        "per_config": per_config,
    }


def run_dsr(returns: np.ndarray, n_trials: int, trial_sharpes: list[float]) -> dict[str, Any]:
    """Führt den Deflated Sharpe Ratio aus (per-Bar-Einheiten).

    ``sr_hat`` = mean(r)/std(r, ddof=1) (per-Bar, nicht annualisiert);
    ``var_sharpe`` = Varianz der Trial-Sharpes (ddof=1). ``bars_per_year``
    wird bewusst nicht übergeben, damit var_sharpe in denselben
    per-Bar-Einheiten bleibt wie die Trial-Sharpes.

    Returns:
        ``{"dsr", "sr_hat", "n_trials", "var_sharpe"}`` oder bei < 2
        Trial-Sharpes ``{"dsr": None, "reason": "need >=2 trial sharpes", ...}``.
    """
    series = np.asarray(returns, dtype=float)
    sr_hat = _per_bar_sharpe(series)
    if len(trial_sharpes) < 2:
        return {
            "dsr": None,
            "reason": "need >=2 trial sharpes",
            "sr_hat": sr_hat,
            "n_trials": n_trials,
            "var_sharpe": None,
        }
    var_sharpe = float(np.var(trial_sharpes, ddof=1))
    dsr = float(deflated_sharpe_ratio(series, n_trials=n_trials, var_sharpe=var_sharpe))
    return {"dsr": dsr, "sr_hat": sr_hat, "n_trials": n_trials, "var_sharpe": var_sharpe}
