"""Tests für die MLflow-Aufzeichnung des Demo-Zyklus (optional, nie fatal)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from apps.demo_trader.service import DemoTraderConfig

from apps.demo_trader.service import (
    ACTION_BUY,
    ACTION_NONE,
    DemoTradePlan,
    log_cycle_to_mlflow,
)
from packages.paper import PaperAccount


class FakeMLflowClient:
    """In-Memory-Ersatz für MLflowClient (start/log/end)."""

    def __init__(self) -> None:
        self.started = 0
        self.ended: list[str] = []
        self.params: list[dict[str, Any]] = []
        self.metrics: list[dict[str, float]] = []
        self.tags: dict[str, str] = {}

    def start_run(
        self, run_name: str | None = None, tags: dict[str, str] | None = None
    ) -> str:
        self.started += 1
        self.tags = dict(tags or {})
        return "run-1"

    def log_parameters(self, run_id: str, params: dict[str, Any]) -> None:
        self.params.append(dict(params))

    def log_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        self.metrics.append(dict(metrics))

    def end_run(self, run_id: str, status: str = "FINISHED") -> None:
        self.ended.append(status)


class TestMlflowCycleLogging:
    """log_cycle_to_mlflow recordet einen Run und bricht den Zyklus nie ab."""

    @pytest.fixture
    def account(self) -> PaperAccount:
        return PaperAccount(account_id="demo", cash=100000.0, initial_cash=100000.0)

    @staticmethod
    def _buy_plan() -> DemoTradePlan:
        return DemoTradePlan(
            action=ACTION_BUY, quantity=0.025, price=80000.0, reason="LONG_BIAS → Market-Buy"
        )

    def test_run_recorded_with_params_and_metrics(
        self,
        config: DemoTraderConfig,
        account: PaperAccount,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MLFLOW_ENABLED", "true")
        fake = FakeMLflowClient()
        log_cycle_to_mlflow(
            config,
            "BTC/USDT",
            "LONG_BIAS",
            0.5,
            self._buy_plan(),
            None,
            account,
            80000.0,
            client_factory=lambda: fake,
        )
        assert fake.started == 1
        assert fake.ended == ["FINISHED"]
        assert fake.tags["instrument"] == "BTC/USDT"
        assert fake.params[0]["decision"] == "LONG_BIAS"
        assert fake.params[0]["min_confidence"] == config.min_confidence
        assert fake.params[0]["latest_close"] == 80000.0
        assert fake.metrics[0]["confidence"] == 0.5
        assert fake.metrics[0]["trade_executed"] == 0.0
        assert fake.metrics[0]["equity"] == account.equity

    def test_trade_executed_flag_set(
        self,
        config: DemoTraderConfig,
        account: PaperAccount,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MLFLOW_ENABLED", "true")
        fake = FakeMLflowClient()
        log_cycle_to_mlflow(
            config,
            "BTC/USDT",
            "LONG_BIAS",
            0.5,
            self._buy_plan(),
            object(),  # ausgeführter Trade (Identitätsprüfung)
            account,
            80000.0,
            client_factory=lambda: fake,
        )
        assert fake.metrics[0]["trade_executed"] == 1.0

    def test_disabled_via_env(
        self,
        config: DemoTraderConfig,
        account: PaperAccount,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MLFLOW_ENABLED", "false")
        fake = FakeMLflowClient()
        log_cycle_to_mlflow(
            config,
            "ETH/USDT",
            "NO_TRADE",
            0.1,
            DemoTradePlan(action=ACTION_NONE, quantity=0.0, price=0.0, reason="—"),
            None,
            account,
            3000.0,
            client_factory=lambda: fake,
        )
        assert fake.started == 0

    def test_client_failure_not_fatal(
        self,
        config: DemoTraderConfig,
        account: PaperAccount,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MLFLOW_ENABLED", "true")

        def boom() -> Any:
            raise RuntimeError("MLflow-Server unerreichbar")

        log_cycle_to_mlflow(
            config,
            "BTC/USDT",
            "LONG_BIAS",
            0.5,
            self._buy_plan(),
            None,
            account,
            80000.0,
            client_factory=boom,
        )  # darf nicht werfen

    def test_partial_logging_failure_ends_run_failed(
        self,
        config: DemoTraderConfig,
        account: PaperAccount,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MLFLOW_ENABLED", "true")
        fake = FakeMLflowClient()

        def broken_metrics(run_id: str, metrics: dict[str, float]) -> None:
            raise RuntimeError("Metrik-Schema defekt")

        fake.log_metrics = broken_metrics  # type: ignore[method-assign]
        log_cycle_to_mlflow(
            config,
            "BTC/USDT",
            "LONG_BIAS",
            0.5,
            self._buy_plan(),
            None,
            account,
            80000.0,
            client_factory=lambda: fake,
        )
        assert fake.ended == ["FAILED"]
