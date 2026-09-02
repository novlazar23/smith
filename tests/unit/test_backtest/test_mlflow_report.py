"""Tests für das MLflow-Logging (opt-in, nie fatal)."""

from __future__ import annotations

from typing import Any

import pytest
from apps.backtest.mlflow_report import log_backtest_to_mlflow
from packages.backtesting.core import BacktestConfig, BacktestResult
from tests.unit.test_backtest.conftest import BTC, FakeMLflowClient


def make_result() -> BacktestResult:
    """Minimaler BacktestResult mit Metriken/Metadata."""
    metrics = {
        "total_return_pct": 1.5,
        "sharpe_ratio": 0.75,
        "sortino_ratio": 0.6,
        "max_drawdown_pct": 2.0,
        "win_rate_pct": 50.0,
        "profit_factor": 1.25,
        "total_trades": 4,
    }
    return BacktestResult(
        config=BacktestConfig(symbol=BTC),
        candles=[],
        snapshots=[],
        trades=[],
        metrics=metrics,
        metadata={"final_equity": 101_500.0},
    )


def make_extra() -> dict[str, Any]:
    return {
        "gate_pass_rate": 0.6,
        "params": {
            "instrument": BTC,
            "venue": "BINANCE_FUTURES",
            "scenario": "crash-2021-05",
            "timeframe": "1m",
            "candle_limit": 200,
            "evaluate_every": 5,
            "min_confidence": 0.3,
            "trade_notional": 2000.0,
            "initial_capital": 100_000.0,
            "data_start": "2021-05-15T00:00:00+00:00",
            "data_end": "2021-05-25T23:59:59+00:00",
        },
    }


class TestMlflowBacktestLogging:
    """log_backtest_to_mlflow recordet einen Run und bricht nie ab."""

    def test_run_recorded_with_params_metrics(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MLFLOW_ENABLED", "true")
        fake = FakeMLflowClient()
        log_backtest_to_mlflow(
            "crash-2021-05",
            make_result(),
            make_extra(),
            BacktestConfig(symbol=BTC),
            client_factory=lambda: fake,
        )
        assert fake.started == 1
        assert fake.ended == ["FINISHED"]
        assert fake.tags["component"] == "backtest"
        assert fake.tags["scenario"] == "crash-2021-05"
        assert fake.run_name is not None
        assert fake.run_name.startswith("BTC/USDT-crash-2021-05-")
        params = fake.params[0]
        for key in (
            "instrument",
            "venue",
            "scenario",
            "timeframe",
            "candle_limit",
            "evaluate_every",
            "min_confidence",
            "trade_notional",
            "initial_capital",
            "data_start",
            "data_end",
        ):
            assert key in params
        assert params["min_confidence"] == "0.3"
        metrics = fake.metrics[0]
        assert metrics["total_return"] == 1.5
        assert metrics["sharpe_ratio"] == 0.75
        assert metrics["sortino_ratio"] == 0.6
        assert metrics["max_drawdown"] == 2.0
        assert metrics["win_rate"] == 50.0
        assert metrics["profit_factor"] == 1.25
        assert metrics["total_trades"] == 4.0
        assert metrics["final_equity"] == 101_500.0
        assert metrics["gate_pass_rate"] == 0.6

    def test_no_artifacts_logged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Datei-Artefakte bleiben im Volume — kein Upload via Tracking-API."""
        monkeypatch.setenv("MLFLOW_ENABLED", "true")
        fake = FakeMLflowClient()
        log_backtest_to_mlflow(
            "full", make_result(), make_extra(), BacktestConfig(symbol=BTC),
            client_factory=lambda: fake,
        )
        assert fake.artifacts == []

    def test_disabled_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MLFLOW_ENABLED", "false")
        fake = FakeMLflowClient()
        log_backtest_to_mlflow(
            "full", make_result(), make_extra(), BacktestConfig(symbol=BTC),
            client_factory=lambda: fake,
        )
        assert fake.started == 0

    def test_disabled_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MLFLOW_ENABLED", raising=False)
        fake = FakeMLflowClient()
        log_backtest_to_mlflow(
            "full", make_result(), make_extra(), BacktestConfig(symbol=BTC),
            client_factory=lambda: fake,
        )
        assert fake.started == 0

    def test_client_failure_not_fatal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MLFLOW_ENABLED", "true")

        def boom() -> Any:
            raise RuntimeError("MLflow-Server unerreichbar")

        log_backtest_to_mlflow(
            "full", make_result(), make_extra(), BacktestConfig(symbol=BTC),
            client_factory=boom,
        )  # darf nicht werfen

    def test_partial_logging_failure_ends_run_failed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MLFLOW_ENABLED", "true")
        fake = FakeMLflowClient()

        def broken_metrics(run_id: str, metrics: dict[str, float]) -> None:
            raise RuntimeError("Metrik-Schema defekt")

        fake.log_metrics = broken_metrics  # type: ignore[method-assign]
        log_backtest_to_mlflow(
            "full", make_result(), make_extra(), BacktestConfig(symbol=BTC),
            client_factory=lambda: fake,
        )
        assert fake.ended == ["FAILED"]

    def test_sweep_run_with_tags_and_per_gate_metrics(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MLFLOW_ENABLED", "true")
        sweep_rows = [
            {"gate": 0.3, "trades": 10, "win_rate": 0.5, "total_return_pct": 1.0,
             "sharpe_ratio": 0.5, "max_drawdown_pct": 2.0, "final_equity": 101_000.0,
             "gate_pass_rate": 0.8},
            {"gate": 0.5, "trades": 4, "win_rate": None, "total_return_pct": 0.5,
             "sharpe_ratio": 0.2, "max_drawdown_pct": 1.0, "final_equity": 100_500.0,
             "gate_pass_rate": 0.4},
        ]
        extra = make_extra()
        extra["sweep"] = sweep_rows
        fake = FakeMLflowClient()
        log_backtest_to_mlflow(
            "crash-2021-05-sweep",
            None,
            extra,
            BacktestConfig(symbol=BTC),
            client_factory=lambda: fake,
            extra_tags={"sweep": "true"},
        )
        assert fake.started == 1
        assert fake.ended == ["FINISHED"]
        assert fake.tags["sweep"] == "true"
        assert fake.tags["scenario"] == "crash-2021-05-sweep"
        metrics = fake.metrics[0]
        assert metrics["gate_0.3_trades"] == 10.0
        assert metrics["gate_0.3_win_rate"] == 0.5
        assert metrics["gate_0.3_total_return"] == 1.0
        assert metrics["gate_0.5_trades"] == 4.0
        assert "gate_0.5_win_rate" not in metrics  # None → wird übersprungen
        assert metrics["gate_pass_rate"] == 0.6

    def test_missing_metric_keys_are_guarde(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MLFLOW_ENABLED", "true")
        result = BacktestResult(
            config=BacktestConfig(symbol=BTC),
            candles=[],
            snapshots=[],
            trades=[],
            metrics={},
            metadata={},
        )
        fake = FakeMLflowClient()
        log_backtest_to_mlflow(
            "full", result, {}, BacktestConfig(symbol=BTC),
            client_factory=lambda: fake,
        )
        assert fake.started == 1
        assert fake.ended == ["FINISHED"]
        assert fake.metrics == []  # keine Metriken vorhanden → kein log_metrics-Call
