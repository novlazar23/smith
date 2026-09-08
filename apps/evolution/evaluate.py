"""Testen & Beurteilen: Backtest-Matrix, deterministischer Judge, Bootstrap-CI.

Die Test-Matrix ist fixiert durch den Preregistrierungs-``TestPlan``:
pro Fenster (Kalibrierung / OOS) x Asset ein Backtest der Variante **und**
der Baseline (gleiche Kerzen, gleiche Kosten, Flatsize, ``allow_pyramiding
=False`` - identisch mit den Kalibrierungsläufen 6-12). Der Judge wendet
ausschließlich die preregistrierte ``DecisionRule`` an — kein LLM, keine
Nachverhandlung.

``block_bootstrap_ci`` reproduziert die Methodik des 12. Laufs
(Block-Bootstrap, Blöcke = 1 Tag, 2000 Wiederholungen, Seed 42) auf die
OOS-Portfolio-Mittelrendite.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from datetime import date
from typing import Any

import numpy as np
from packages.backtesting.core import BacktestConfig, Candle
from packages.backtesting.datafeed import MemoryDataFeed
from packages.backtesting.engine import BacktestEngine
from packages.strategies import create_strategy

from .models import Hypothesis, TestPlan, Variant, utcnow_iso

logger = logging.getLogger(__name__)

#: Feed-Lader: (instrument, start_iso, end_iso|None, resample|None) → Kerzen.
FeedFactory = Callable[[str, str | None, str | None, str | None], list[Candle]]

#: Referenz-Setup (Kalibrierungsläufe 6-12): Startkapital, 10-%-Flatsize.
INITIAL_CAPITAL = 100_000.0
TRADE_NOTIONAL = 10_000.0
#: Deflations-Schwelle: ab dieser Anzahl Hypothesen in einer Familie wird
#: die geforderte OOS-Marge verdoppelt (grobe Multi-Testing-Korrektur).
DEFLATION_FAMILY_THRESHOLD = 10

WINDOWS: tuple[str, ...] = ("calibration", "oos")


def run_window(
    candles: Sequence[Candle],
    strategy_name: str,
    params: dict[str, float],
    *,
    initial_capital: float = INITIAL_CAPITAL,
    trade_notional: float = TRADE_NOTIONAL,
    timeframe: str = "5m",
) -> dict[str, Any]:
    """Ein Backtest-Fenster einer Strategie-Variante.

    Kosten = ``BacktestConfig``-Defaults (0,1 % Commission + 5 bps Slippage
    = 0,15 %/Seite, de-fakti-Kosten der Läufe 6-12).

    Returns:
        ``{"return_pct", "max_dd_pct", "legs", "win_rate", "n_candles",
        "sharpe_ratio", "profit_factor"}``.
    """
    if not candles:
        raise ValueError("keine Kerzen für Fenster")
    instrument = candles[0].symbol
    strategy = create_strategy(
        strategy_name,
        instrument,
        dict(params),
        initial_capital=initial_capital,
        trade_notional=trade_notional,
    )
    config = BacktestConfig(
        symbol=instrument,
        timeframe=timeframe,
        initial_capital=initial_capital,
        allow_pyramiding=False,
        warmup_bars=strategy.candle_limit,
    )
    engine = BacktestEngine(config)
    result = engine.run(MemoryDataFeed(list(candles)), strategy, warmup_bars=strategy.candle_limit)
    round_trips = result.metadata.get("round_trips", [])
    wins = sum(1 for trip in round_trips if float(trip.get("pnl", 0.0)) > 0.0)
    metrics = result.metrics
    return {
        "return_pct": float(metrics.get("total_return_pct", 0.0) or 0.0),
        "max_dd_pct": float(metrics.get("max_drawdown_pct", 0.0) or 0.0),
        "legs": len(round_trips),
        "win_rate": round(wins / len(round_trips), 4) if round_trips else None,
        "n_candles": len(candles),
        "sharpe_ratio": metrics.get("sharpe_ratio"),
        "profit_factor": metrics.get("profit_factor"),
    }


def daily_returns(candles: Sequence[Candle]) -> list[float]:
    """Tagesrenditen (Close-to-Close pro UTC-Tag) aus Kerzen."""
    closes_by_day: dict[date, float] = {}
    for candle in candles:
        closes_by_day[candle.timestamp.date()] = candle.close
    closes = [closes_by_day[day] for day in sorted(closes_by_day)]
    return [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes)) if closes[i - 1] > 0]


def block_bootstrap_ci(
    per_asset_daily: list[Sequence[float]],
    *,
    n_iterations: int = 2000,
    seed: int = 42,
) -> dict[str, float] | None:
    """Block-Bootstrap-CI der OOS-Portfolio-Mittelrendite (12.-Lauf-Methodik).

    Pro Wiederholung wird pro Asset die Tagesrenditen-Folge blockweise
    (Block = 1 Tag) resampled, die Asset-Gesamtrendite als Produkt
    berechnet und über die Assets gemittelt. Liefert
    ``{"mean_pct", "ci_low_pct", "ci_high_pct", "n_iterations", "seed"}``
    (nur die Verteilung der Portfolio-Mittel; der Baseline-Vergleich
    erfolgt im Digest, nicht im Judge).
    """
    per_asset = [np.asarray(list(daily), dtype=np.float64) for daily in per_asset_daily]
    per_asset = [arr for arr in per_asset if len(arr) >= 5]
    if not per_asset:
        return None
    rng = np.random.default_rng(seed)
    means = np.empty(n_iterations, dtype=np.float64)
    for i in range(n_iterations):
        totals: list[float] = []
        for arr in per_asset:
            idx = rng.integers(0, len(arr), size=len(arr))
            totals.append(float(np.prod(1.0 + arr[idx]) - 1.0))
        means[i] = float(np.mean(totals)) * 100.0
    return {
        "mean_pct": round(float(np.mean(means)), 4),
        "ci_low_pct": round(float(np.percentile(means, 2.5)), 4),
        "ci_high_pct": round(float(np.percentile(means, 97.5)), 4),
        "n_iterations": n_iterations,
        "seed": seed,
    }


def run_variant_matrix(
    hypothesis: Hypothesis,
    feed_factory: FeedFactory,
    baseline: Variant,
) -> tuple[dict[str, Any] | None, str | None]:
    """Führt die preregistrierte Test-Matrix aus.

    Returns:
        ``(matrix, None)`` bei Erfolg oder ``(None, Grund)`` wenn kein
    einziges OOS-Asset-Fenster Daten lieferte (Hard-Fail).
    """
    plan: TestPlan = hypothesis.test_plan
    resample = "5m" if plan.timeframe == "5m" else None
    windows: dict[str, tuple[str | None, str | None]] = {
        "calibration": (plan.calibration_start, plan.calibration_end),
        "oos": (plan.oos_start, plan.oos_end),
    }
    matrix: dict[str, Any] = {"windows": {}, "portfolio": {}, "bootstrap": None, "data_gaps": []}
    oos_candles: dict[str, Sequence[Candle]] = {}

    for window_name in WINDOWS:
        start, end = windows[window_name]
        per_asset: dict[str, Any] = {}
        for instrument in plan.instruments:
            candles = feed_factory(instrument, start, end, resample)
            if len(candles) < plan.min_candles:
                matrix["data_gaps"].append(
                    f"{instrument}:{window_name}: {len(candles)} Kerzen < {plan.min_candles}"
                )
                continue
            per_asset[instrument] = {
                "variant": run_window(
                    candles, hypothesis.variant.strategy, hypothesis.variant.params, timeframe=plan.timeframe
                ),
                "baseline": run_window(
                    candles, baseline.strategy, baseline.params, timeframe=plan.timeframe
                ),
                "start": candles[0].timestamp.isoformat(),
                "end": candles[-1].timestamp.isoformat(),
            }
            if window_name == "oos":
                oos_candles[instrument] = candles
        matrix["windows"][window_name] = per_asset
        if per_asset:
            variant_returns = [entry["variant"]["return_pct"] for entry in per_asset.values()]
            baseline_returns = [entry["baseline"]["return_pct"] for entry in per_asset.values()]
            matrix["portfolio"][window_name] = {
                "variant_mean_pct": round(float(np.mean(variant_returns)), 4),
                "baseline_mean_pct": round(float(np.mean(baseline_returns)), 4),
                "legs": sum(entry["variant"]["legs"] for entry in per_asset.values()),
                "positive_windows": sum(1 for value in variant_returns if value > 0.0),
                "max_dd_max_pct": max(entry["variant"]["max_dd_pct"] for entry in per_asset.values()),
                "n": len(per_asset),
            }

    oos_portfolio = matrix["portfolio"].get("oos")
    if oos_portfolio is None:
        return None, "kein OOS-Asset-Fenster mit ausreichenden Daten"

    matrix["bootstrap"] = block_bootstrap_ci([daily_returns(candles) for candles in oos_candles.values()])
    return matrix, None


def judge(hypothesis: Hypothesis, matrix: dict[str, Any], n_family: int) -> dict[str, Any]:
    """Deterministischer Judge: wendet die preregistrierte DecisionRule an.

    Multi-Testing-Deflation: ``n_family >= DEFLECTION_FAMILY_THRESHOLD``
    verdoppelt die geforderte OOS-Marge. Die Kalibrations-Metrik wird
    nur mitgeführt (Deskription), nicht als Gate — die Baseline ist
    selbst auf dem Kalibrierungszeitraum ermittelt.
    """
    rule = hypothesis.decision_rule
    oos = matrix["portfolio"]["oos"]
    margin = rule.oos_margin_min_pct
    deflated = n_family >= DEFLATION_FAMILY_THRESHOLD
    if deflated:
        margin *= 2.0

    reasons: list[str] = []
    delta = float(oos["variant_mean_pct"]) - float(oos["baseline_mean_pct"])
    if delta < margin:
        reasons.append(
            f"OOS-Marge {delta:+.2f} pp < gefordert {margin:+.2f} pp"
            + (" (deflatiert)" if deflated else "")
        )
    if int(oos["legs"]) < rule.min_legs:
        reasons.append(f"Legs {int(oos['legs'])} < {rule.min_legs}")
    if float(oos["max_dd_max_pct"]) > rule.max_dd_pct:
        reasons.append(f"Max-DD {float(oos['max_dd_max_pct']):.2f} pp > {rule.max_dd_pct:.2f} pp")
    if rule.majority_positive and int(oos["positive_windows"]) * 2 < int(oos["n"]):
        reasons.append(
            f"nur {int(oos['positive_windows'])}/{int(oos['n'])} OOS-Asset-Fenster positiv (Mehrheit gefordert)"
        )

    return {
        "decision": "promoted" if not reasons else "rejected",
        "reasons": reasons,
        "delta_pp": round(delta, 4),
        "n_family": n_family,
        "deflated": deflated,
        "judged_at": utcnow_iso(),
    }
