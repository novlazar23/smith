"""AgentEnsembleStrategy — die PRODUKTIONS-Entscheidungslogik als Backtest-Strategie.

Jede ``evaluate_every``-te Kerze wird exakt so analysiert wie im Live-Betrieb
des ``demo-trader``: Die letzten ``candle_limit`` Kerzen (älteste → neueste)
werden als ``CandleWindow`` gebaut, aus ihnen ``market_data`` erzeugt, ein
frisches ACTIVE-Ensemble (``build_active_ensemble``: das kanonische
4-Agenten-Ensemble aus Trend, Mean-Reversion, Volatilitäts-Regime und
Volumen-Konviktions) erstellt und die ``OrchestratorPipeline`` (mit der
kalibrierten Ensemble-WeightConfig) ausgeführt. Die
Konsens-Entscheidung wird mit derselben long-only-Semantik wie
``plan_trade`` auf ein ``StrategySignal`` gemappt:

  - ``confidence < gate``        → kein Signal (Konfidenz-Gate)
  - ``LONG_BIAS``                → BUY (position_size = trade_notional / initial_capital)
  - ``SHORT_BIAS``               → SELL (Position glattstellen)
  - sonst (``NO_TRADE``/``RANGE``) → kein Signal

Optional filtert die **Entry-Selektion** nur die BUY-Seite weiter (SELL/Exit
bleibt unverändert): ``entry_gate`` hebt die Konfidenz-Schwelle für BUY an
(überschneidet nicht unter das Basis-Gate), und ``entry_required_agents``
verlangt, dass alle benannten Agenten in derselben Evaluation selbst ``LONG``
voten. Ein Agent, der dafür keine ``LONG``-Report-Lieferung hat, blockiert
den Entry (fail-closed).

Die rohen Konsensergebnisse (decision, confidence, Per-Agent-Details) werden
pro Bar in ``consensus_cache`` gecacht. Ein Gate-Sweep kann daraus über
``replay_with_gate`` bzw. die ``ReplayStrategy`` (``runner.py``) Signale für
andere Gates ableiten, **ohne** Pipeline/Agenten erneut auszuführen.

Einschränkung: Per-Agent-Details stammen aus ``result.first_round_reports``
(``AgentReport``-Felder ``agent_id``/``probabilities``/``raw_confidence``/
``status``). Reports, die diese Felder nicht exposeen (z.B. Platzhalter in
Stub-Pipelines), werden still übersprungen.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from apps.demo_trader.service import build_active_ensemble
from apps.orchestrator_service.service import (
    CandleWindow,
    ContextualAgent,
    build_calibrated_pipeline,
    build_market_data,
)
from numpy.typing import NDArray
from packages.backtesting.strategies import (
    BaseStrategy,
    SignalAction,
    StrategySignal,
)
from packages.consensus import ConsensusDecision

if TYPE_CHECKING:
    from packages.backtesting.core import Candle
    from packages.orchestrator.pipeline import OrchestratorPipelineResult

logger = logging.getLogger(__name__)

PerAgentInfo = dict[str, dict[str, Any]]


class PipelineRunner(Protocol):
    """Duck-Typ-Schnittstelle der ``OrchestratorPipeline`` (Testbarkeit)."""

    def run(
        self,
        run_id: str,
        instrument: str,
        agents: Sequence[object],
        market_data: Mapping[str, NDArray[np.float64]],
    ) -> OrchestratorPipelineResult:
        """Führt eine Pipeline-Zyklen-Iteration aus und liefert das Ergebnis."""
        ...


PipelineFactory = Callable[[], PipelineRunner]
EnsembleFactory = Callable[[str, str], list[ContextualAgent]]


@dataclass(frozen=True)
class Evaluation:
    """Ein einzelner Ensemble-Evaluations-Schritt."""

    timestamp: datetime
    decision: str
    confidence: float
    per_agent: PerAgentInfo = field(default_factory=dict)
    signal_emitted: bool = False


@dataclass(frozen=True)
class SignalEvent:
    """Ein ausgegebenes Signal (für Trade-Rekonstruktion und Sweeps)."""

    bar_index: int
    timestamp: datetime
    action: str  # "BUY" | "SELL"
    confidence: float
    decision: str


def derive_action(decision: str, confidence: float, gate: float) -> str:
    """Mappt (Konsens-Entscheidung, Konfidenz) auf eine Signal-Aktion.

    Returns:
        ``"BUY"`` (LONG_BIAS ≥ gate), ``"SELL"`` (SHORT_BIAS ≥ gate) oder
        ``"NONE"`` (Gate verfehlt oder keine handelbare Entscheidung).
    """
    if confidence < gate:
        return "NONE"
    if decision == ConsensusDecision.LONG_BIAS.value:
        return "BUY"
    if decision == ConsensusDecision.SHORT_BIAS.value:
        return "SELL"
    return "NONE"


def entry_allowed(
    confidence: float,
    per_agent: PerAgentInfo,
    buy_gate: float,
    required_agents: Sequence[str],
) -> bool:
    """Prüft die Entry-Selektion für ein BUY-Signal.

    True, wenn die Konfidenz die Buy-Schwelle ``buy_gate`` erreicht und alle
    in ``required_agents`` benannten Agenten in dieser Evaluation selbst
    ``LONG`` voten. Ein Agent, der in ``per_agent`` fehlt oder nicht ``LONG``
    votet, blockiert den Entry (Defensive: fehlende/UNKNOWN-Details zählen
    als Verneinung, damit eine unvollständige Report-Lieferung nicht versehentlich
    einen Entry durchlässt). Die Exit-Seite (SELL) ist davon nicht betroffen.
    """
    if confidence < buy_gate:
        return False
    for agent_id in required_agents:
        info = per_agent.get(agent_id)
        if info is None or info.get("direction") != "LONG":
            return False
    return True


def dominant_direction(probabilities: Mapping[str, Any]) -> str:
    """Baut die dominante Richtung (LONG/SHORT/RANGE) aus Wahrscheinlichkeiten."""
    try:
        up = float(probabilities.get("up", 0.0))
        down = float(probabilities.get("down", 0.0))
        range_prob = float(probabilities.get("range", 0.0))
    except (TypeError, ValueError):
        return "UNKNOWN"
    if up + down + range_prob <= 0:
        return "UNKNOWN"
    candidates = (("LONG", up), ("SHORT", down), ("RANGE", range_prob))
    direction, _ = max(candidates, key=lambda item: item[1])
    return direction


def extract_per_agent(reports: Sequence[Any]) -> PerAgentInfo:
    """Extrahiert Per-Agent-Details aus First-Round-Reports (defensiv).

    Erwartet ``AgentReport``-artige Objekte (``agent_id``, ``probabilities``,
    ``raw_confidence``, ``status``); Platzhalter-Objekte ohne diese Felder
    werden still übersprungen.

    Returns:
        Mapping ``agent_id → {direction, confidence, status}``.
    """
    per_agent: PerAgentInfo = {}
    for report in reports:
        agent_id = getattr(report, "agent_id", None)
        if agent_id is None:
            continue
        probabilities = getattr(report, "probabilities", None) or {}
        confidence = getattr(report, "raw_confidence", None)
        per_agent[str(agent_id)] = {
            "direction": dominant_direction(probabilities) if isinstance(probabilities, Mapping) else "UNKNOWN",
            "confidence": float(confidence) if confidence is not None else 0.0,
            "status": str(getattr(report, "status", "") or "unknown"),
        }
    return per_agent


class AgentEnsembleStrategy(BaseStrategy):
    """Backtest-Strategie, die die Live-Ensemble-Pipeline pro Bar-Fenster ausführt."""

    def __init__(
        self,
        instrument: str,
        horizon: str = "15m",
        candle_limit: int = 200,
        min_candles: int = 30,
        evaluate_every: int = 5,
        min_confidence: float = 0.3,
        entry_gate: float | None = None,
        entry_required_agents: tuple[str, ...] = (),
        trade_notional: float = 2000.0,
        initial_capital: float = 100_000.0,
        pipeline_factory: PipelineFactory | None = None,
        ensemble_factory: EnsembleFactory = build_active_ensemble,
    ) -> None:
        """Initialisiert die Strategie (Parameter spiegeln den Demo-Trader).

        Args:
            instrument: Handelspaar, z.B. ``"BTC/USDT"``.
            horizon: Analyse-Horizont des Ensembles (z.B. ``"15m"``).
            candle_limit: Fenstergröße in Kerzen (wie ``DEMO_CANDLE_LIMIT``).
            min_candles: Mindestanzahl Kerzen vor der ersten Evaluation.
            evaluate_every: Evaluation alle N Bars (Live-Rhythmus 5 Min.).
            min_confidence: Konfidenz-Gate (wie ``DEMO_MIN_CONFIDENCE``).
            entry_gate: Optional höhere Konfidenz-Schwelle **nur für BUY**
                (SELL nutzt weiter ``min_confidence``); ``None`` = kein
                zusätzlicher Entry-Filter.
            entry_required_agents: Agent-IDs, die für einen BUY **alle
                selbst LONG voten** müssen (z.B. ``("trend",)``); leere
                Sequenz = aus.
            trade_notional: Ziel-Nominal pro BUY in USDT (wie
                ``DEMO_TRADE_NOTIONAL``).
            initial_capital: Startkapital (für position_size = notional/capital).
            pipeline_factory: Erzeugt die Pipeline (Default: echte
                ``OrchestratorPipeline`` mit der kalibrierten
                Ensemble-WeightConfig; in Tests injizierbar).
            ensemble_factory: Erzeugt das ACTIVE-Ensemble (Default:
                ``build_active_ensemble``; in Tests injizierbar).
        """
        super().__init__(name=f"agent-ensemble-{instrument}")
        if candle_limit < 1:
            raise ValueError("candle_limit muss >= 1 sein")
        if min_candles < 1:
            raise ValueError("min_candles muss >= 1 sein")
        if evaluate_every < 1:
            raise ValueError("evaluate_every muss >= 1 sein")
        if initial_capital <= 0:
            raise ValueError("initial_capital muss > 0 sein")
        if entry_gate is not None and not 0.0 <= entry_gate <= 1.0:
            raise ValueError("entry_gate muss in [0, 1] liegen oder None sein")
        self.instrument = instrument
        self.horizon = horizon
        self.candle_limit = candle_limit
        self.min_candles = min_candles
        self.evaluate_every = evaluate_every
        self.min_confidence = min_confidence
        self.entry_gate = entry_gate
        self.entry_required_agents = tuple(entry_required_agents)
        self.trade_notional = trade_notional
        self.initial_capital = initial_capital
        self._pipeline_factory = pipeline_factory or build_calibrated_pipeline
        self._ensemble_factory = ensemble_factory
        self._window: deque[Candle] = deque(maxlen=candle_limit)
        self._bar_index = 0
        self.evaluations: list[Evaluation] = []
        self.signal_events: list[SignalEvent] = []
        self.consensus_cache: dict[int, tuple[str, float, PerAgentInfo]] = {}

    def on_bar(self, candle: Candle) -> StrategySignal | None:
        """Neue Kerze verarbeiten; alle ``evaluate_every`` Bars Ensemble-Evaluation.

        Der Cache-Key ist die Gesamtzahl der ``on_bar``-Aufrufe (Bar-Index im
        Feed), damit eine ReplayStrategy dieselben Indizes wiederverwenden kann.
        """
        self._window.append(candle)
        bars_seen = self._bar_index + 1
        self._bar_index += 1
        if bars_seen % self.evaluate_every != 0:
            return None
        if len(self._window) < self.min_candles:
            return None
        return self._evaluate(candle, self._bar_index - 1)

    def _effective_buy_gate(self, base_gate: float) -> float:
        """Buy-Schwelle: ``entry_gate`` erhöht das Basis-Gate (nie unter es)."""
        if self.entry_gate is None:
            return base_gate
        return max(base_gate, self.entry_gate)

    def _evaluate(self, candle: Candle, bar_index: int) -> StrategySignal | None:
        """Führt die Pipeline aus, cached den Konsens und mappt auf ein Signal."""
        window = CandleWindow(
            open=np.array([c.open for c in self._window], dtype=np.float64),
            high=np.array([c.high for c in self._window], dtype=np.float64),
            low=np.array([c.low for c in self._window], dtype=np.float64),
            close=np.array([c.close for c in self._window], dtype=np.float64),
            volume=np.array([c.volume for c in self._window], dtype=np.float64),
        )
        market_data = build_market_data(window)
        agents = self._ensemble_factory(self.instrument, self.horizon)
        result = self._pipeline_factory().run(
            run_id=f"backtest-{self.instrument}-{candle.timestamp.isoformat()}",
            instrument=self.instrument,
            agents=agents,
            market_data=market_data,
        )
        decision = result.decision
        consensus = result.consensus
        confidence = float(consensus.confidence) if consensus is not None else 0.0
        per_agent = extract_per_agent(result.first_round_reports)
        self.consensus_cache[bar_index] = (decision, confidence, per_agent)

        action = derive_action(decision, confidence, self.min_confidence)
        if action == "BUY" and not entry_allowed(
            confidence, per_agent, self._effective_buy_gate(self.min_confidence), self.entry_required_agents
        ):
            action = "NONE"
        signal = self._build_signal(action, decision, confidence, candle)
        self.evaluations.append(
            Evaluation(
                timestamp=candle.timestamp,
                decision=decision,
                confidence=confidence,
                per_agent=per_agent,
                signal_emitted=signal is not None,
            )
        )
        if signal is not None:
            self.signal_events.append(
                SignalEvent(
                    bar_index=bar_index,
                    timestamp=candle.timestamp,
                    action=action,
                    confidence=confidence,
                    decision=decision,
                )
            )
        return signal

    def _build_signal(
        self,
        action: str,
        decision: str,
        confidence: float,
        candle: Candle,
    ) -> StrategySignal | None:
        """Baute das StrategySignal (long-only-Semantik wie plan_trade)."""
        if action == "NONE":
            return None
        metadata = {
            "decision": decision,
            "confidence": confidence,
            "instrument": self.instrument,
        }
        if action == "BUY":
            return StrategySignal(
                action=SignalAction.BUY,
                symbol=candle.symbol,
                confidence=confidence,
                reason=f"LONG_BIAS → BUY (confidence={confidence:.3f} ≥ gate {self.min_confidence:.2f})",
                position_size=self.trade_notional / self.initial_capital,
                timestamp=candle.timestamp,
                metadata=metadata,
            )
        return StrategySignal(
            action=SignalAction.SELL,
            symbol=candle.symbol,
            confidence=confidence,
            reason=f"SHORT_BIAS → Position glattstellen (confidence={confidence:.3f})",
            position_size=0.0,
            timestamp=candle.timestamp,
            metadata=metadata,
        )

    def replay_with_gate(self, gate: float) -> list[tuple[datetime, str, float]]:
        """Leitet die Signal-Liste für ein anderes Gate aus dem Cache ab.

        Führt **keine** Pipeline/Ausführung aus — nur die Signal-Ableitung aus
        ``consensus_cache``. Hinweis: Die Ausführungseffekte (Equity-Kurve,
        Positionszustand) unterscheiden sich pro Gate, weil die Engine die
        neu abgeleiteten Signale gegen ein frisches Konto läuft; diese Methode
        liefert ausschließlich die Signal-Liste (Zeitstempel, Aktion, Konfidenz).
        """
        replayed: list[tuple[datetime, str, float]] = []
        bar_indices = sorted(self.consensus_cache)
        timestamps = [evaluation.timestamp for evaluation in self.evaluations]
        for bar_index, timestamp in zip(bar_indices, timestamps, strict=True):
            decision, confidence, per_agent = self.consensus_cache[bar_index]
            action = derive_action(decision, confidence, gate)
            if action == "BUY" and not entry_allowed(
                confidence, per_agent, self._effective_buy_gate(gate), self.entry_required_agents
            ):
                action = "NONE"
            if action != "NONE":
                replayed.append((timestamp, action, confidence))
        return replayed

    def evaluations_to_dicts(self) -> list[dict[str, Any]]:
        """Liefert die Evaluations als JSON-serialisierbare Dictionaries."""
        return [
            {
                "timestamp": evaluation.timestamp.isoformat(),
                "decision": evaluation.decision,
                "confidence": evaluation.confidence,
                "per_agent": evaluation.per_agent,
                "signal_emitted": evaluation.signal_emitted,
            }
            for evaluation in self.evaluations
        ]

    def to_dict(self) -> dict[str, Any]:
        """Zusammenfassung der Strategie für Report/MLflow."""
        decision_counts: dict[str, int] = {}
        per_agent: dict[str, dict[str, Any]] = {}
        confidence_sum = 0.0
        for evaluation in self.evaluations:
            decision_counts[evaluation.decision] = decision_counts.get(evaluation.decision, 0) + 1
            confidence_sum += evaluation.confidence
            for agent_id, info in evaluation.per_agent.items():
                entry = per_agent.setdefault(
                    agent_id, {"evaluations": 0, "confidence_sum": 0.0, "directions": {}}
                )
                entry["evaluations"] += 1
                entry["confidence_sum"] += float(info.get("confidence", 0.0))
                direction = str(info.get("direction", "UNKNOWN"))
                entry["directions"][direction] = entry["directions"].get(direction, 0) + 1
        for entry in per_agent.values():
            entry["mean_confidence"] = round(entry.pop("confidence_sum") / entry["evaluations"], 4)
        n_evaluations = len(self.evaluations)
        return {
            "instrument": self.instrument,
            "horizon": self.horizon,
            "candle_limit": self.candle_limit,
            "min_candles": self.min_candles,
            "evaluate_every": self.evaluate_every,
            "min_confidence": self.min_confidence,
            "entry_gate": self.entry_gate,
            "entry_required_agents": list(self.entry_required_agents),
            "trade_notional": self.trade_notional,
            "initial_capital": self.initial_capital,
            "n_evaluations": n_evaluations,
            "decision_distribution": decision_counts,
            "mean_confidence": round(confidence_sum / n_evaluations, 4) if n_evaluations else 0.0,
            "per_agent": per_agent,
            "n_buy_signals": sum(1 for event in self.signal_events if event.action == "BUY"),
            "n_sell_signals": sum(1 for event in self.signal_events if event.action == "SELL"),
        }
