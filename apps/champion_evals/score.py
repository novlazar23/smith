"""Champion-Evaluations-Job: pro-Agent OOS-Scoring über Zeit.

Führt das kanonische ACTIVE-4-Agenten-Ensemble rückwärts auf historischen
Kerzen aus (derselbe Produktions-Pfad wie der Demo-Trader), bewertet pro
Agent die Kalibrierungs- vs. OOS-Leistung (3-Klassen-Brier), Stabilität
(Hit-Rate) und den marginalen Ensemble-Beitrag (Leave-One-Out) und baut das
``champion_evals.json``-Artefakt für den Champion-Feed.

Kein Look-Ahead: Pro Schritt i sieht die Pipeline nur Kerzen bis inkl. i
(Fenster); das Outcome realisiert sich ``horizon_bars`` Kerzen **nach** i.
Die Zeitachse wird temporal in ein früheres (Kalibrierung = champion) und ein
späteres (OOS = challenger) Fenster gesplittet — derselbe Agent, zwei Zeiten.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from packages.backtesting.core import Candle
from packages.validation.ablation.loo import AgentEnsemble, LeaveOneOutAblation
from packages.validation.target_variables import TargetConfig, encode_target

# AgentReport-Probabilities (lowercase) → Brier/Target-Encoding (uppercase)
_PROB_KEY_MAP = {"up": "UP", "down": "DOWN", "range": "RANGE"}
_CLASSES = ("UP", "DOWN", "RANGE")


@dataclass(frozen=True)
class EvalSample:
    """Ein Ensemble-Bewertungsschritt: Pro-Agent-Voraussagen + Real-Outcome."""

    as_of: datetime
    per_agent_probs: dict[str, dict[str, float]]
    actual: str
    realized_return: float


@dataclass(frozen=True)
class AgentMetrics:
    """Pro-Agent-Metriken über das Kalibrierungs- und das OOS-Fenster."""

    agent_id: str
    cal_samples: int
    oos_samples: int
    cal_brier: float
    oos_brier: float
    cal_stability: float
    oos_stability: float
    oos_marginal: float


def normalize_probs(raw: Mapping[str, Any]) -> dict[str, float]:
    """Mappt AgentReport-Probabilities (up/down/range) auf UP/DOWN/RANGE."""
    lowered = {str(key).lower(): float(value) for key, value in raw.items()}
    return {mapped: value for key, value in lowered.items() if (mapped := _PROB_KEY_MAP.get(key))}


def _brier(predictions: Sequence[Mapping[str, float]], actuals: Sequence[str]) -> float:
    """Mittlerer 3-Klassen-Brier-Score (tiefer = besser)."""
    n = len(predictions)
    if n == 0:
        return 0.0
    total = 0.0
    for pred, actual in zip(predictions, actuals, strict=True):
        for cls in _CLASSES:
            total += (float(pred.get(cls, 0.0)) - (1.0 if actual == cls else 0.0)) ** 2
    return total / n


def _hit_rate(predictions: Sequence[Mapping[str, float]], actuals: Sequence[str]) -> float:
    """Anteil der Schritte, deren argmax-Richtung dem Real-Outcome entspricht."""
    n = len(predictions)
    if n == 0:
        return 0.0
    hits = sum(
        1
        for pred, actual in zip(predictions, actuals, strict=True)
        if max(_CLASSES, key=lambda c: float(pred.get(c, 0.0))) == actual
    )
    return hits / n


def _score(brier: float) -> float:
    """Brier (tiefer=besser) → Score (höher=besser) für den Optimizer."""
    return 1.0 - brier


def score_window(samples: Sequence[EvalSample], calibration_ratio: float = 0.5) -> dict[str, AgentMetrics]:
    """SPLITTET die Zeitachse temporal und bewertet pro Agent.

    Das frühere ``calibration_ratio``-Segment ist das Kalibrierungsfenster
    (champion/Bestandsleistung), der Rest das OOS-Fenster (challenger/aktuell).
    Nur Agenten, die in JEDEM Schritt liefern, werden bewertet (sonst sind
    Prognose und Outcome nicht mehr sauber paubar).
    """
    usable = sorted((s for s in samples if s.per_agent_probs), key=lambda s: s.as_of)
    if len(usable) < 2:
        return {}
    common = set.intersection(*(set(s.per_agent_probs) for s in usable))
    if not common:
        return {}
    split = max(1, min(len(usable) - 1, int(len(usable) * calibration_ratio)))
    calibration, oos = usable[:split], usable[split:]

    cal_pairs: dict[str, list[tuple[dict[str, float], str]]] = {a: [] for a in common}
    oos_pairs: dict[str, list[tuple[dict[str, float], str]]] = {a: [] for a in common}
    for sample in calibration:
        for a in common:
            cal_pairs[a].append((sample.per_agent_probs[a], sample.actual))
    for sample in oos:
        for a in common:
            oos_pairs[a].append((sample.per_agent_probs[a], sample.actual))

    marginals: dict[str, float] = {}
    if len(common) >= 2:
        oos_agents = [AgentEnsemble(agent_id=a, predictions=[p for p, _ in oos_pairs[a]]) for a in common]
        for result in LeaveOneOutAblation(higher_is_better=False).run_ensemble(oos_agents, [s.actual for s in oos]):
            marginals[result.agent_id] = result.marginal_contribution

    metrics: dict[str, AgentMetrics] = {}
    for a in common:
        cal_preds = [p for p, _ in cal_pairs[a]]
        cal_actuals = [act for _, act in cal_pairs[a]]
        oos_preds = [p for p, _ in oos_pairs[a]]
        oos_actuals = [act for _, act in oos_pairs[a]]
        metrics[a] = AgentMetrics(
            agent_id=a,
            cal_samples=len(cal_preds),
            oos_samples=len(oos_preds),
            cal_brier=_brier(cal_preds, cal_actuals),
            oos_brier=_brier(oos_preds, oos_actuals),
            cal_stability=_hit_rate(cal_preds, cal_actuals),
            oos_stability=_hit_rate(oos_preds, oos_actuals),
            oos_marginal=marginals.get(a, 0.0),
        )
    return metrics


def build_artifact(metrics: Mapping[str, AgentMetrics], version: str = "current") -> dict[str, Any]:
    """Baut das ``champion_evals.json``-Artefakt (Schema: ``champion_feed``).

    Pro Agent: ``champion`` = Kalibrierungsfenster (Bestandsleistung),
    ``challenger`` = OOS-Fenster (aktuelle Leistung). ``new_risks`` bleibt
    leer (keine Risiko-Detektion in diesem Job) und ``shadow_success`` True.
    """
    artifact: dict[str, Any] = {}
    for agent_id, m in metrics.items():
        artifact[agent_id] = {
            "champion": {
                "version": version,
                "oos_score": _score(m.cal_brier),
                "calibration_score": _score(m.cal_brier),
                "stability_score": m.cal_stability,
                "marginal_contribution": 0.0,
                "shadow_days": 0,
                "samples": m.cal_samples,
            },
            "challenger": {
                "version": version,
                "oos_score": _score(m.oos_brier),
                "calibration_score": _score(m.cal_brier),
                "stability_score": m.oos_stability,
                "marginal_contribution": m.oos_marginal,
                "shadow_days": 0,
                "samples": m.oos_samples,
            },
            "new_risks": [],
            "shadow_success": True,
        }
    return artifact


def write_artifact(path: Path | str, metrics: Mapping[str, AgentMetrics], version: str = "current") -> Path:
    """Schreibt das Artefakt als JSON (lesbar für ``champion_feed``) und liefert den Pfad.

    Atomar (tmp-Datei + ``Path.replace``): Der Orchestrator lädt das
    Artefakt bei Mtime-Änderung neu und sieht nie einen halben Stand.
    """
    out = Path(path)
    payload = json.dumps(build_artifact(metrics, version), indent=2, ensure_ascii=False) + "\n"
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(out)
    return out


def replay_ensemble(
    candles: Sequence[Candle],
    instrument: str,
    horizon: str = "15m",
    *,
    candle_limit: int = 200,
    min_candles: int = 30,
    evaluate_every: int = 5,
    horizon_bars: int = 3,
    target_config: TargetConfig | None = None,
    pipeline_factory: Callable[[], Any] | None = None,
    ensemble_factory: Callable[[str, str], list[Any]] | None = None,
) -> list[EvalSample]:
    """Führt das Ensemble rückwärts auf den Kerzen und sammelt Scoring-Samples.

    Gleicher Produktions-Pfad wie ``AgentEnsembleStrategy``/Demo-Trader
    (Fenster → market_data → Pipeline → erste Runde). ``pipeline_factory`` und
    ``ensemble_factory`` sind injizierbar (Default: echte kalibrierte Pipeline +
    ACTIVE-Ensemble). Nur Schritte, deren Outcome innerhalb der Kerzen
    realisiert werden kann, werden gesammelt (kein Look-Ahead über den Rand).
    """
    # Lazy: Pipeline/Ensemble ziehen das komplette Agent-Subsystem nach; der
    # Scoring-Kern (score_window/build_artifact) bleibt dadurch unabhängig testbar.
    from apps.demo_trader.service import build_active_ensemble
    from apps.orchestrator_service.service import (
        CandleWindow,
        build_calibrated_pipeline,
        build_market_data,
    )

    target = target_config or TargetConfig()
    make_pipeline = pipeline_factory or build_calibrated_pipeline
    make_ensemble = ensemble_factory or (lambda instrument, horizon: build_active_ensemble(instrument, horizon))

    # Einmalig aufbauen und wiederverwenden (Performance): ein Live-Instance, viele Runs.
    pipeline = make_pipeline()
    agents = make_ensemble(instrument, horizon)

    window: deque[Candle] = deque(maxlen=candle_limit)
    samples: list[EvalSample] = []
    n = len(candles)
    for i, candle in enumerate(candles):
        window.append(candle)
        bars_seen = i + 1
        if bars_seen % evaluate_every != 0 or len(window) < min_candles or i + horizon_bars >= n:
            continue
        md = build_market_data(
            CandleWindow(
                open=np.array([c.open for c in window], dtype=np.float64),
                high=np.array([c.high for c in window], dtype=np.float64),
                low=np.array([c.low for c in window], dtype=np.float64),
                close=np.array([c.close for c in window], dtype=np.float64),
                volume=np.array([c.volume for c in window], dtype=np.float64),
            )
        )
        result = pipeline.run(
            run_id=f"champ-eval-{instrument}-{candle.timestamp.isoformat()}",
            instrument=instrument,
            agents=agents,
            market_data=md,
        )
        per_agent: dict[str, dict[str, float]] = {}
        for report in result.first_round_reports:
            agent_id = getattr(report, "agent_id", None)
            raw = getattr(report, "probabilities", None)
            if agent_id is None or not isinstance(raw, Mapping):
                continue
            probs = normalize_probs(raw)
            if probs:
                per_agent[str(agent_id)] = probs
        realized_return = float(candles[i + horizon_bars].close / candle.close - 1.0)
        samples.append(
            EvalSample(
                as_of=candle.timestamp,
                per_agent_probs=per_agent,
                actual=str(encode_target(realized_return, target)),
                realized_return=realized_return,
            )
        )
    return samples
