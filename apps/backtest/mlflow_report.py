"""MLflow-Logging für Backtest-Runs (opt-in, nie fatal).

Spiegelt das Pattern von ``demo_trader.log_cycle_to_mlflow``: aktiviert über
die Env ``MLFLOW_ENABLED=true`` (Default ``false``), Tracking-URI über
``MLFLOW_TRACKING_URI`` (Default ``http://mlflow:5000``), Experiment
``backtest`` (Override: ``MLFLOW_EXPERIMENT_NAME``). Ein unerreichbarer
MLflow-Server erzeugt nur eine Warnung — der Backtest wird **nie** dadurch
abgebrochen. Bei teilweisem Logging-Fehler wird der Run mit Status
``FAILED`` beendet.

Geloggert wird pro Szenario ein Run (Params/Metriken); für Gate-Sweeps ein
eigener Run (Tag ``sweep=true``, Metriken pro Gate). Artefakt-Dateien
(report.json, equity_curve.csv, evaluations.json, sweep.json) werden
bewusst NICHT als MLflow-Artefakte geloggert: die Datei-Artefakte liegen
im Backtest-Artefakt-Verzeichnis (Docker-Volume), und ein Upload über die
Tracking-API scheitert bei hier genutzten Client/Server-Versionen an
lokalen artifact_uris (Cross-Container). Params/Metriken tragen die
Kalibrierungs-Evidenz; die Dateien bleiben im Volume.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from packages.observability.mlflow_client import MLflowClient

if TYPE_CHECKING:
    from packages.backtesting.core import BacktestConfig, BacktestResult

logger = logging.getLogger(__name__)

DEFAULT_TRACKING_URI = "http://mlflow:5000"
DEFAULT_EXPERIMENT_NAME = "backtest"

METRIC_SOURCES: tuple[tuple[str, str], ...] = (
    ("total_return", "total_return_pct"),
    ("sharpe_ratio", "sharpe_ratio"),
    ("sortino_ratio", "sortino_ratio"),
    ("max_drawdown", "max_drawdown_pct"),
    ("win_rate", "win_rate_pct"),
    ("profit_factor", "profit_factor"),
    ("total_trades", "total_trades"),
)

SWEEP_METRIC_SOURCES: tuple[tuple[str, str], ...] = (
    ("trades", "trades"),
    ("win_rate", "win_rate"),
    ("total_return", "total_return_pct"),
    ("sharpe", "sharpe_ratio"),
    ("max_drawdown", "max_drawdown_pct"),
    ("final_equity", "final_equity"),
    ("gate_pass_rate", "gate_pass_rate"),
)


def _is_enabled() -> bool:
    """True, wenn MLflow-Logging über die Env aktiviert ist."""
    return os.environ.get("MLFLOW_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")


def _collect_metrics(
    result: BacktestResult | None, extra: dict[str, Any]
) -> dict[str, float]:
    """Sammelt die zu loggenden Metriken (nur vorhandene, numerische Werte)."""
    metrics: dict[str, float] = {}
    if result is not None:
        for name, source in METRIC_SOURCES:
            value = result.metrics.get(source)
            if isinstance(value, (int, float)):
                metrics[name] = float(value)
        final_equity = result.metadata.get("final_equity")
        if isinstance(final_equity, (int, float)):
            metrics["final_equity"] = float(final_equity)
    gate_pass_rate = extra.get("gate_pass_rate")
    if isinstance(gate_pass_rate, (int, float)):
        metrics["gate_pass_rate"] = float(gate_pass_rate)
    for row in extra.get("sweep", []) or []:
        gate = row.get("gate")
        if gate is None:
            continue
        for name, source in SWEEP_METRIC_SOURCES:
            value = row.get(source)
            if isinstance(value, (int, float)):
                metrics[f"gate_{gate}_{name}"] = float(value)
    return metrics


def log_backtest_to_mlflow(
    label: str,
    result: BacktestResult | None,
    extra: dict[str, Any],
    config: BacktestConfig,
    client_factory: Callable[[], Any] | None = None,
    extra_tags: dict[str, str] | None = None,
) -> None:
    """Recordet einen Backtest-Run in MLflow (optional, nie fatal).

    Args:
        label: Szenario-Label (Tag ``scenario``, Teil des Run-Namens).
        result: BacktestResult (None für reine Gate-Sweep-Runs).
        extra: Analytics-Dict; ``extra["params"]`` wird als MLflow-Params
            geloggert, ``extra["sweep"]`` (Zeilen-Liste) als pro-Gate-Metriken.
        config: BacktestConfig (Symbol/Kapital für Run-Name und Params).
        client_factory: Erzeugt den MLflow-Client (Testbarkeit; Default:
            ``MLflowClient`` aus den Env-Defaults).
        extra_tags: Zusätzliche Tags (z.B. ``{"sweep": "true"}``).
    """
    if not _is_enabled():
        return
    factory = client_factory
    if factory is None:
        factory = lambda: MLflowClient(  # noqa: E731
            tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI),
            experiment_name=os.environ.get("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME),
        )
    params = dict(extra.get("params", {}) or {})
    params.setdefault("instrument", config.symbol)
    params.setdefault("scenario", label)
    params.setdefault("initial_capital", config.initial_capital)
    tags = {"component": "backtest", "scenario": str(label)}
    if extra_tags:
        tags.update(extra_tags)
    run_name = f"{config.symbol}-{label}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    try:
        client = factory()
        run_id = client.start_run(run_name=run_name, tags=tags)
        try:
            client.log_parameters(run_id, {key: str(value) for key, value in params.items()})
            metrics = _collect_metrics(result, extra)
            if metrics:
                client.log_metrics(run_id, metrics)
            client.end_run(run_id, status="FINISHED")
        except Exception:
            with contextlib.suppress(Exception):
                client.end_run(run_id, status="FAILED")
            raise
    except Exception as exc:
        logger.warning(
            "MLflow-Run-Aufzeichnung für Backtest %s fehlgeschlagen (nicht fatal): %s", label, exc
        )
