"""Backtest-Runner: Engine-Wrapper, Analytics und Gate-Sweep.

``run_backtest`` ist ein dünner Wrapper um ``BacktestEngine`` mit
sinnvollen Defaults (Startkapital der Strategie, Commission 0,1 %,
Slippage 5 bps, Warmup = candle_limit, Symbol = Instrument).

Der Gate-Sweep rechnet die Pipeline **nicht** je Gate neu, sondern in zwei
Phasen:

1. Phase 1: ein voller Engine-Run mit der echten ``AgentEnsembleStrategy``
   (oder einer übergebenen, bereits gelaufenen ``warm_strategy``) füllt
   ``consensus_cache``.
2. Phase 2: pro Gate ein frischer Engine-Run mit einer
   ``ReplayStrategy``, die pro Bar nur einen O(1)-Cache-Lookup macht und
   die Signale mit dem neuen Gate ableitet — keine Agenten, keine Pipeline.

Laufzeit: O(gates * candles) statt O(gates * candles * agents).

Einschränkungen der zugrunde liegenden Engine (bestehendes Verhalten,
nicht modifizierbar):

- Die Engine-Trade-Liste und die Positionen am Ende tragen keine
  Candle-Zeitstempel, und geschlossene Positionen werden aus dem
  Paper-Account gelöscht. Daher werden ``trades``/``win_rate`` im Sweep
  und in den Confidence-Buckets aus den Signal-Events der Strategie
   rekonstruiert (BUY = Entry, SELL = Exit; PnL = Equity-vor-Entry *
   Positionsanteil * Kursänderung Close→Close, ohne Slippage/Kommission —
   dokumentierte Approximation).
- ``PaperExecutor.close_position`` schließt immer zum ``avg_price``
  (Dummy-Marktpreis) und ``PaperPosition.market_value`` verwendet den
  Einkaufskurs — die Engine-Equity bewertet Longs also nicht zum
  Marktpreis. Für ein einzelnes long-only Instrument bleibt die
  End-Equity daher ≈ Startkapital minus Kosten; die Kursentwicklung zeigt
  sich erst über die close-preis-basierte Rekonstruktion (Buckets/Sweep).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from packages.backtesting.core import BacktestConfig
from packages.backtesting.engine import BacktestEngine
from packages.backtesting.strategies import BaseStrategy, SignalAction, StrategySignal

from .agent_strategy import (
    SignalEvent,
    derive_action,
)

if TYPE_CHECKING:
    from packages.backtesting.core import BacktestResult, Candle
    from packages.backtesting.datafeed import DataFeed

    from .agent_strategy import AgentEnsembleStrategy

logger = logging.getLogger(__name__)

DEFAULT_COMMISSION_RATE = 0.001
DEFAULT_SLIPPAGE_BPS = 5.0

CONFIDENCE_BUCKETS: tuple[tuple[float, float], ...] = (
    (0.3, 0.4),
    (0.4, 0.5),
    (0.5, 0.6),
    (0.6, 0.7),
    (0.7, 0.8),
    (0.8, 1.01),  # oberes Ende inklusive 1.0
)


def bucket_label(low: float, high: float) -> str:
    """Formatiert einen Bucket als ``[low,high)`` (oberes Ende inklusiv bei 1.0)."""
    if high >= 1.0:
        return f"[{low:.1f},1.0]"
    return f"[{low:.1f},{high:.1f})"


def default_config(
    strategy: AgentEnsembleStrategy, config: BacktestConfig | None = None
) -> BacktestConfig:
    """BacktestConfig mit den Runner-Defaults (Symbol, Kapital, Kosten, Warmup)."""
    base = config if config is not None else BacktestConfig()
    return base.model_copy(
        update={
            "symbol": strategy.instrument,
            "initial_capital": strategy.initial_capital,
            "commission_rate": DEFAULT_COMMISSION_RATE,
            "slippage_bps": DEFAULT_SLIPPAGE_BPS,
            "warmup_bars": strategy.candle_limit,
        }
    )


def run_backtest(
    feed: DataFeed,
    strategy_factory: Callable[[], AgentEnsembleStrategy],
    config: BacktestConfig | None = None,
    label: str = "backtest",
) -> BacktestResult:
    """Führt einen einzelnen Backtest über den ``BacktestEngine`` aus.

    Args:
        feed: Datenfeed (z.B. ``ClickHouseDataFeed`` oder ``MemoryDataFeed``).
        strategy_factory: Erzeugt die Strategie (z.B. ``AgentEnsembleStrategy``).
        config: Optionale BacktestConfig (sonst Defaults aus ``default_config``).
        label: Kennung für Logging.

    Returns:
        ``BacktestResult``; die verwendete Strategie-Instanz liegt zusätzlich
        unter ``result.metadata["strategy"]`` (für ``extra_metrics``/Sweeps).
    """
    strategy = strategy_factory()
    engine = BacktestEngine(default_config(strategy, config))
    logger.info("Backtest '%s' gestartet (%s)", label, feed.symbol)
    result = engine.run(feed, strategy, warmup_bars=strategy.candle_limit)
    result.metadata["strategy"] = strategy
    return result


def extra_metrics(result: BacktestResult, strategy: AgentEnsembleStrategy) -> dict[str, Any]:
    """Ergänzende Analytics jenseits der Engine-Metriken (Gate/Entscheidungen/Agenten).

    Returns:
        Dict mit Gate, gate_pass_rate (Anteil Evaluations ≥ Gate),
        decision_distribution, mean_confidence, per_agent-Stats sowie den
        wichtigsten Engine-Metriken (final_equity, total_return, ...).
    """
    evaluations = strategy.evaluations
    n_evaluations = len(evaluations)
    gate = strategy.min_confidence
    passed = sum(1 for evaluation in evaluations if evaluation.confidence >= gate)
    decision_counts: dict[str, int] = {}
    per_agent: dict[str, dict[str, Any]] = {}
    confidence_sum = 0.0
    for evaluation in evaluations:
        decision_counts[evaluation.decision] = decision_counts.get(evaluation.decision, 0) + 1
        confidence_sum += evaluation.confidence
        for agent_id, info in evaluation.per_agent.items():
            entry = per_agent.setdefault(agent_id, {"evaluations": 0, "confidence_sum": 0.0, "directions": {}})
            entry["evaluations"] += 1
            entry["confidence_sum"] += float(info.get("confidence", 0.0))
            direction = str(info.get("direction", "UNKNOWN"))
            entry["directions"][direction] = entry["directions"].get(direction, 0) + 1
    for entry in per_agent.values():
        entry["mean_confidence"] = round(entry.pop("confidence_sum") / entry["evaluations"], 4)
    metrics = result.metrics
    return {
        "gate": gate,
        "gate_pass_rate": round(passed / n_evaluations, 4) if n_evaluations else 0.0,
        "n_evaluations": n_evaluations,
        "decision_distribution": decision_counts,
        "mean_confidence": round(confidence_sum / n_evaluations, 4) if n_evaluations else 0.0,
        "per_agent": per_agent,
        "final_equity": result.metadata.get("final_equity"),
        "total_return_pct": metrics.get("total_return_pct"),
        "sharpe_ratio": metrics.get("sharpe_ratio"),
        "sortino_ratio": metrics.get("sortino_ratio"),
        "max_drawdown_pct": metrics.get("max_drawdown_pct"),
        "win_rate_pct": metrics.get("win_rate_pct"),
        "total_trades": metrics.get("total_trades"),
        "profit_factor": metrics.get("profit_factor"),
    }


def _reconstruct_round_trips(
    signal_events: list[SignalEvent],
    candles: Sequence[Candle],
    equity_curve: list[float],
    warmup: int,
    position_fraction: float,
) -> list[dict[str, Any]]:
    """Rekonstruiert Round-Trips aus Signal-Events (BUY = Entry, SELL = Exit).

    Nur Signale ab ``warmup`` zählen (die Engine ignoriert frühere). Ein BUY
    in Long-Position (Nachkauf) und ein SELL ohne Position werden ignoriert.
    Offene Longs werden zum letzten Close ausgewiesen (``open=True``).
    """
    trips: list[dict[str, Any]] = []
    open_entry: dict[str, Any] | None = None
    last_bar = len(candles) - 1
    for event in signal_events:
        if event.bar_index < warmup:
            continue
        if event.action == "BUY" and open_entry is None:
            equity_index = event.bar_index - warmup
            equity_before = (
                equity_curve[equity_index]
                if equity_index < len(equity_curve)
                else (equity_curve[-1] if equity_curve else 0.0)
            )
            open_entry = {
                "entry_bar": event.bar_index,
                "entry_confidence": event.confidence,
                "equity_before": equity_before,
            }
        elif event.action == "SELL" and open_entry is not None:
            trip = _close_trip(open_entry, event.bar_index, candles, position_fraction)
            trip["open"] = False
            trips.append(trip)
            open_entry = None
    if open_entry is not None and last_bar >= 0:
        trip = _close_trip(open_entry, last_bar, candles, position_fraction)
        trip["open"] = True
        trips.append(trip)
    return trips


def _close_trip(
    entry: dict[str, Any], exit_bar: int, candles: Sequence[Candle], fraction: float
) -> dict[str, Any]:
    """Rechnet den PnL eines Trips aus Close-Preisen (open-Flag setzt der Caller)."""
    entry_price = float(candles[entry["entry_bar"]].close)
    exit_price = float(candles[exit_bar].close)
    change = (exit_price / entry_price - 1.0) if entry_price > 0 else 0.0
    pnl = float(entry["equity_before"]) * fraction * change
    return {
        "entry_bar": entry["entry_bar"],
        "exit_bar": exit_bar,
        "entry_confidence": entry["entry_confidence"],
        "pnl": round(pnl, 2),
        "win": pnl > 0,
    }


def confidence_buckets(result: BacktestResult, strategy: AgentEnsembleStrategy) -> list[dict[str, Any]]:
    """Konfidenz-Buckets: Evaluations pro Bucket + Trade-Stats der Bucket-Einträge.

    Buckets: [0.3,0.4), [0.4,0.5), [0.5,0.6), [0.6,0.7), [0.7,0.8), [0.8,1.0].
    Trade-Zuordnung: Ein Round-Trip wird dem Bucket der Konfidenz seiner
    ENTRY-Evaluation zugeordnet (Signal-Event ↔ Evaluation 1:1). PnL ist eine
    Close-basierte Approximation (ohne Slippage/Kommission, siehe Modul-Doku).
    """
    warmup = strategy.candle_limit
    equity_curve = result.metadata.get("equity_curve", [])
    fraction = strategy.trade_notional / strategy.initial_capital
    trips = _reconstruct_round_trips(
        strategy.signal_events, result.candles, equity_curve, warmup, fraction
    )
    rows: list[dict[str, Any]] = []
    for low, high in CONFIDENCE_BUCKETS:
        bucket_trips = [t for t in trips if low <= t["entry_confidence"] < high]
        wins = sum(1 for trip in bucket_trips if trip["win"])
        rows.append(
            {
                "bucket": bucket_label(low, high),
                "low": low,
                "high": high,
                "n_evaluations": sum(
                    1
                    for evaluation in strategy.evaluations
                    if low <= evaluation.confidence < high
                ),
                "n_trades": len(bucket_trips),
                "win_rate": round(wins / len(bucket_trips), 4) if bucket_trips else None,
                "avg_pnl": (
                    round(sum(t["pnl"] for t in bucket_trips) / len(bucket_trips), 2)
                    if bucket_trips
                    else None
                ),
            }
        )
    return rows


class ReplayStrategy(BaseStrategy):
    """Gate-Sweep-Strategie: bedient gecachte Konsens-Signale mit neuem Gate.

    Führt **keine** Pipeline/Agenten aus — jede Bar macht nur einen Lookup in
    dem in Phase 1 gefüllten ``consensus_cache`` (gleiche Bar-Indizes, da
    dieselbe Candle-Sequenz). Die Cache-Keys kodieren bereits die
    Evaluate-every-/Min-Candles-Logik, daher reicht der Lookup.
    """

    def __init__(
        self,
        cache: Mapping[int, tuple[str, float, dict[str, dict[str, Any]]]],
        instrument: str,
        gate: float,
        trade_notional: float,
        initial_capital: float,
        name: str = "replay",
    ) -> None:
        """Initialisiert die Replay-Strategie.

        Args:
            cache: Konsens-Cache (Bar-Index → (decision, confidence, per_agent)).
            instrument: Handelspaar (Symbol der Signale).
            gate: Neues Konfidenz-Gate für die Signal-Ableitung.
            trade_notional: Nominal pro BUY (für position_size).
            initial_capital: Startkapital (für position_size).
            name: Strategie-Name (Logging).
        """
        super().__init__(name=name)
        self.instrument = instrument
        self.gate = gate
        self.trade_notional = trade_notional
        self.initial_capital = initial_capital
        self._cache = cache
        self._bar_index = 0
        self.signal_events: list[SignalEvent] = []

    def on_bar(self, candle: Candle) -> StrategySignal | None:
        """Gecachte Konsens der Bar mit dem Replay-Gate auf ein Signal mappen."""
        bar_index = self._bar_index
        self._bar_index += 1
        entry = self._cache.get(bar_index)
        if entry is None:
            return None
        decision, confidence, _ = entry
        action = derive_action(decision, confidence, self.gate)
        if action == "NONE":
            return None
        self.signal_events.append(
            SignalEvent(
                bar_index=bar_index,
                timestamp=candle.timestamp,
                action=action,
                confidence=confidence,
                decision=decision,
            )
        )
        return StrategySignal(
            action=SignalAction.BUY if action == "BUY" else SignalAction.SELL,
            symbol=candle.symbol,
            confidence=confidence,
            reason=f"replay: {decision} @ gate {self.gate:.2f}",
            position_size=(
                self.trade_notional / self.initial_capital if action == "BUY" else 0.0
            ),
            timestamp=candle.timestamp,
            metadata={"decision": decision, "confidence": confidence, "gate": self.gate},
        )


def _sweep_row(
    gate: float,
    result: BacktestResult,
    replay: ReplayStrategy,
    cached_confidences: list[float],
    warmup: int,
    position_fraction: float,
) -> dict[str, Any]:
    """Baut eine Gate-Sweep-Tabelle-Zeile aus einem Replay-Run."""
    trips = _reconstruct_round_trips(
        replay.signal_events,
        result.candles,
        result.metadata.get("equity_curve", []),
        warmup,
        position_fraction,
    )
    wins = sum(1 for trip in trips if trip["win"])
    n_cached = len(cached_confidences)
    return {
        "gate": gate,
        "trades": sum(1 for event in replay.signal_events if event.action == "BUY"),
        "round_trips": len(trips),
        "win_rate": round(wins / len(trips), 4) if trips else None,
        "total_return_pct": result.metrics.get("total_return_pct"),
        "sharpe_ratio": result.metrics.get("sharpe_ratio"),
        "max_drawdown_pct": result.metrics.get("max_drawdown_pct"),
        "final_equity": result.metadata.get("final_equity"),
        "gate_pass_rate": (
            round(sum(1 for c in cached_confidences if c >= gate) / n_cached, 4)
            if n_cached
            else 0.0
        ),
    }


def gate_sweep(
    feed: DataFeed,
    strategy_factory: Callable[[], AgentEnsembleStrategy],
    gates: list[float],
    config: BacktestConfig | None = None,
    warm_strategy: AgentEnsembleStrategy | None = None,
) -> list[dict[str, Any]]:
    """Führt den Gate-Sweep aus (keine Pipeline-Rekomputation pro Gate).

    Phase 1: falls ``warm_strategy`` (bereits gelaufen, Cache gefüllt) nicht
    übergeben wird, ein voller Engine-Run mit einer frischen Strategie aus
    ``strategy_factory``. Phase 2: pro Gate ein frischer Engine-Run mit
    ``ReplayStrategy`` (Cache-Lookups statt Agenten).

    Returns:
        Eine Zeile pro Gate: gate, trades (BUY-Entries), round_trips,
        win_rate, total_return_pct, sharpe_ratio, max_drawdown_pct,
        final_equity, gate_pass_rate (Anteil gecachter Konsense ≥ gate).
    """
    base = warm_strategy if warm_strategy is not None else strategy_factory()
    if warm_strategy is None:
        engine = BacktestEngine(default_config(base, config))
        engine.run(feed, base, warmup_bars=base.candle_limit)
    cached_confidences = [confidence for _, confidence, _ in base.consensus_cache.values()]
    fraction = base.trade_notional / base.initial_capital
    rows: list[dict[str, Any]] = []
    for gate in gates:
        replay = ReplayStrategy(
            cache=base.consensus_cache,
            instrument=base.instrument,
            gate=gate,
            trade_notional=base.trade_notional,
            initial_capital=base.initial_capital,
            name=f"replay-gate-{gate}",
        )
        engine = BacktestEngine(default_config(base, config))
        result = engine.run(feed, replay, warmup_bars=base.candle_limit)
        rows.append(
            _sweep_row(gate, result, replay, cached_confidences, base.candle_limit, fraction)
        )
        logger.info(
            "Gate-Sweep gate=%.2f: trades=%s, win_rate=%s, return=%s",
            gate,
            rows[-1]["trades"],
            rows[-1]["win_rate"],
            rows[-1]["total_return_pct"],
        )
    return rows
