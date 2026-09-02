"""Gemeinsame Fakes und Helfer für die Backtest-Tests.

Alle Abhängigkeiten (ClickHouse-Engine, Orchestrator-Pipeline, MLflow-Client)
werden injiziert — es gibt keine Netzwerk- oder Datenbank-Zugriffe.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from apps.backtest.agent_strategy import AgentEnsembleStrategy
from packages.backtesting.core import Candle
from packages.consensus import ConsensusDecision, ConsensusResult
from packages.orchestrator.pipeline import OrchestratorPipelineResult

BTC = "BTC/USDT"
BASE_TIME = datetime(2021, 5, 15, 0, 0, 0, tzinfo=UTC)


def make_candles(
    n: int,
    start: datetime = BASE_TIME,
    price0: float = 100.0,
    step: float = 0.0,
    symbol: str = BTC,
) -> list[Candle]:
    """Erzeugt n deterministische 1m-Kerzen (Close = price0 + i*step)."""
    candles: list[Candle] = []
    price = price0
    for i in range(n):
        close = price + step
        candles.append(
            Candle(
                timestamp=start + timedelta(minutes=i),
                symbol=symbol,
                open=price,
                high=max(price, close) + 0.5,
                low=min(price, close) - 0.5,
                close=close,
                volume=1000.0,
            )
        )
        price = close
    return candles


def trending_up(n: int, **kwargs: Any) -> list[Candle]:
    """Aufwärts-Trend (Close steigt linear)."""
    kwargs.setdefault("step", 1.0)
    return make_candles(n, **kwargs)


def ranging(n: int, **kwargs: Any) -> list[Candle]:
    """Range-Phase (Close oszilliert ±1 um den Startpreis)."""
    candles: list[Candle] = []
    price = kwargs.pop("price0", 100.0)
    for i in range(n):
        delta = 1.0 if i % 2 == 0 else -1.0
        close = price + delta
        candles.append(
            Candle(
                timestamp=BASE_TIME + timedelta(minutes=i),
                symbol=kwargs.get("symbol", BTC),
                open=price,
                high=max(price, close) + 0.5,
                low=min(price, close) - 0.5,
                close=close,
                volume=1000.0,
            )
        )
        price = close
    return candles


def crash(n: int, **kwargs: Any) -> list[Candle]:
    """Crash (Close fällt linear)."""
    kwargs.setdefault("step", -2.0)
    return make_candles(n, **kwargs)


def rows_from_candles(candles: Sequence[Candle]) -> list[list[str]]:
    """Kerzen → ClickHouse-Zeilen (Strings, TabSeparatedWithNames-Kompatibel)."""
    return [
        [
            candle.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            f"{candle.open:.4f}",
            f"{candle.high:.4f}",
            f"{candle.low:.4f}",
            f"{candle.close:.4f}",
            f"{candle.volume:.4f}",
        ]
        for candle in candles
    ]


class FakeEngine:
    """Duck-Typ-Ersatz für die ClickHouse-Engine (rekordiert SQL, feste Zeilen)."""

    def __init__(self, rows: list[list[str]] | None = None, names: list[str] | None = None) -> None:
        self.names = names or ["open_time", "open", "high", "low", "close", "volume"]
        self.rows = rows or []
        self.queries: list[str] = []

    def query(self, sql: str) -> tuple[list[str], list[list[str]]]:
        self.queries.append(sql)
        return self.names, self.rows


class StubPipeline:
    """Stellvertreter für OrchestratorPipeline (vorgefertigte Results).

    Akzeptiert ein einzelnes Result (wird immer zurückgeliefert) oder eine
    Liste (wird pro Aufruf zyklisch abgearbeitet). Alle Aufrufe werden in
    ``calls`` rekordiert.
    """

    def __init__(self, results: OrchestratorPipelineResult | list[OrchestratorPipelineResult]) -> None:
        self.results = list(results) if isinstance(results, list) else [results]
        self.calls: list[dict[str, Any]] = []

    def run(
        self,
        run_id: str,
        instrument: str,
        agents: list,
        market_data: dict,
    ) -> OrchestratorPipelineResult:
        self.calls.append(
            {"run_id": run_id, "instrument": instrument, "agents": agents, "market_data": market_data}
        )
        index = (len(self.calls) - 1) % len(self.results)
        return self.results[index]


def make_pipeline_result(
    decision: str = "LONG_BIAS",
    confidence: float = 0.9,
    reports: list[Any] | None = None,
) -> OrchestratorPipelineResult:
    """Baut ein OrchestratorPipelineResult mit Konsens."""
    consensus = ConsensusResult(
        decision=ConsensusDecision(decision),
        vote_distribution={},
        agent_weights={},
        agent_agreements=[],
        agent_disagreements=[],
        confidence=confidence,
        reason="stub",
    )
    return OrchestratorPipelineResult(
        decision=decision,
        consensus=consensus,
        first_round_reports=reports if reports is not None else [],
        seal_records=[],
        second_round_reports=[],
    )


class FakeReport:
    """Duck-Typ-Ersatz für AgentReport (für die Per-Agent-Extraktion)."""

    def __init__(
        self,
        agent_id: str,
        up: float = 0.5,
        down: float = 0.2,
        range_prob: float = 0.3,
        raw_confidence: float = 0.8,
        status: str = "active",
    ) -> None:
        self.agent_id = agent_id
        self.probabilities = {"up": up, "down": down, "range": range_prob}
        self.raw_confidence = raw_confidence
        self.status = status


class FakeMLflowClient:
    """In-Memory-Ersatz für MLflowClient (start/log/end, rekordiert alles)."""

    def __init__(self) -> None:
        self.started = 0
        self.run_name: str | None = None
        self.tags: dict[str, str] = {}
        self.ended: list[str] = []
        self.params: list[dict[str, Any]] = []
        self.metrics: list[dict[str, float]] = []
        self.artifacts: list[tuple[str, str]] = []

    def start_run(self, run_name: str | None = None, tags: dict[str, str] | None = None) -> str:
        self.started += 1
        self.run_name = run_name
        self.tags = dict(tags or {})
        return "run-1"

    def log_parameters(self, run_id: str, params: dict[str, Any]) -> None:
        self.params.append(dict(params))

    def log_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        self.metrics.append(dict(metrics))

    def log_artifact(self, run_id: str, local_path: str, artifact_path: str | None = None) -> None:
        self.artifacts.append((run_id, local_path))

    def end_run(self, run_id: str, status: str = "FINISHED") -> None:
        self.ended.append(status)


def make_strategy(
    results: OrchestratorPipelineResult | list[OrchestratorPipelineResult],
    **overrides: Any,
) -> AgentEnsembleStrategy:
    """Baut eine AgentEnsembleStrategy mit StubPipeline (injiziert)."""
    defaults: dict[str, Any] = {
        "instrument": BTC,
        "horizon": "15m",
        "candle_limit": 50,
        "min_candles": 30,
        "evaluate_every": 5,
        "min_confidence": 0.3,
        "trade_notional": 2000.0,
        "initial_capital": 100_000.0,
    }
    defaults.update(overrides)
    stub = results if isinstance(results, StubPipeline) else StubPipeline(results)
    return AgentEnsembleStrategy(
        pipeline_factory=lambda: stub,
        ensemble_factory=lambda instrument, horizon: [],
        **defaults,
    )


def drive(strategy: AgentEnsembleStrategy, candles: Sequence[Candle]) -> list:
    """Treibt die Strategie barweise und sammelt die emittierten Signale."""
    signals: list = []
    for candle in candles:
        signal = strategy.on_bar(candle)
        if signal is not None:
            signals.append(signal)
    return signals
