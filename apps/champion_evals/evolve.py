"""Evolutions-Kern: Mutation, Selektion und Persistenz der Agenten-Parameter.

Pro Familie (``trend``, ``mean_reversion``, ``volatility_regime``,
``volume_conviction``) wetterifert der Champion (aktueller Parametersatz
aus ``champion_configs.json``) gegen k mutierte Varianten auf demselben
Kerzenfenster (``replay_instances`` + ``score_window``). Eine Variante
gewinnt nur, wenn sie den Champion im OOS-Score (1 - Brier) um mindestens
``PROMOTION_MARGIN`` schlägt und ihre OOS-Hit-Rate nicht weiter als
``STABILITY_TOLERANCE`` unter ihrer eigenen Kalibrierungs-Hit-Rate liegt
(kein Overfit auf OOS-Glück). Der Aufrufer kann die Margin per
``select(..., promotion_margin=...)`` anheben — ``trial_ledger``: die
Hurdle steigt mit dem kumulativen Trial-Count gegen die Daten-
wiederverwendung im rollierenden OOS-Fenster. Der Gewinner wird atomar
in ``champion_configs.json`` persistiert; ``champion_evals.json`` bleibt
unverändert.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from apps.champion_evals.agent_params import PARAM_CLASSES
from apps.champion_evals.score import AgentMetrics
from packages.agents.base import BaseParams

logger = logging.getLogger(__name__)

# Minimaler OOS-Score-Vorsprung (1 - Brier) für eine Promotion
PROMOTION_MARGIN = 0.005
# OOS-Hit-Rate darf höchstens so weit unter der Kalibrierungs-Hit-Rate liegen
STABILITY_TOLERANCE = 0.05
# "Nachbar"-Mutationen (kleine Schritte um den Champion), Rest weiter weg
_LOCAL_STRENGTH = 0.05
_GLOBAL_STRENGTH = 0.15


@dataclass(frozen=True)
class SelectionResult:
    """Selektionsentscheid pro Familie: Champion bleibt oder Variante gewinnt."""

    agent_id: str
    selected_id: str
    promoted: bool
    champion_score: float
    best_score: float


def mutate(
    params: BaseParams,
    space: Mapping[str, tuple[str, float, float, float]],
    rng: random.Random,
    strength: float = _GLOBAL_STRENGTH,
) -> BaseParams:
    """Ein mutierter Parametersatz (im Suchraum, deterministisch in ``rng``).

    Jedes Feld wird mit 50 % Wahrscheinlichkeit berührt: int-Felder
    wandern um bis zu ``strength * |Wert|``, float-Felder per Gauß
    mit ``std = strength * (hi - lo)``. Beides wird auf die
    deklarierte Schrittweite gerundet und auf die Suchraum-Grenzen
    geclippt. Berührt kein Feld (alle Skips), wird der Satz
    unverändert zurückgegeben.
    """
    current = params.to_dict()
    touched: dict[str, float | int] = {}
    for name, (kind, lo, hi, step) in space.items():
        if rng.random() >= 0.5:
            continue
        if kind == "int":
            base = float(current[name])
            delta = round(rng.uniform(-strength, strength) * max(1.0, abs(base)))
            touched[name] = int(min(hi, max(lo, round(base) + delta)))
        else:
            value = float(current[name]) + rng.gauss(0.0, strength * (hi - lo))
            if step > 0:
                value = round(round(value / step) * step, 10)
            touched[name] = float(min(hi, max(lo, value)))
    if not touched:
        return params
    return type(params).from_dict({**current, **touched})


def generate_variants(
    agent_id: str,
    champion: Mapping[str, Any],
    k: int,
    seed: int | None = None,
) -> list[tuple[str, BaseParams]]:
    """k mutierte Varianten eines Champion-Satzes (deterministisch pro Seed).

    Die ersten ``max(1, k // 3)`` Varianten liegen eng am Champion
    (strength 0.05), der Rest weiter weg (0.15). IDs:
    ``<agent_id>:v0`` … ``<agent_id>:v{k-1}`` (der Champion selbst ist
    keine Variante).
    """
    cls = PARAM_CLASSES[agent_id]
    base = cls.from_dict(champion)
    space = cls.PARAM_SPACE
    rng = random.Random(seed)
    local_count = max(1, k // 3)
    seen: set[tuple[tuple[str, float | int], ...]] = {tuple(sorted(base.to_dict().items()))}
    variants: list[tuple[str, BaseParams]] = []
    while len(variants) < k:
        strength = _LOCAL_STRENGTH if len(variants) < local_count else _GLOBAL_STRENGTH
        candidate = mutate(base, space, rng, strength=strength)
        key = tuple(sorted(candidate.to_dict().items()))
        if key in seen:
            continue
        seen.add(key)
        variants.append((f"{agent_id}:v{len(variants)}", candidate))
    return variants


def select(
    agent_id: str,
    champion_id: str,
    metrics: Mapping[str, AgentMetrics],
    promotion_margin: float = PROMOTION_MARGIN,
    stability_tolerance: float = STABILITY_TOLERANCE,
) -> SelectionResult:
    """Wählt Champion oder besten Varianten-Gewinner (deterministisch).

    Promotion nur, wenn die beste Variante den Champion im OOS-Score
    (1 - Brier) um mindestens ``promotion_margin`` schlägt und ihre
    OOS-Hit-Rate nicht weiter als ``stability_tolerance`` unter ihrer
    eigenen Kalibrierungs-Hit-Rate liegt. Gleichstand hält den Champion.
    """
    champion = metrics[champion_id]
    champion_score = 1.0 - champion.oos_brier
    best_id, best = champion_id, champion
    for candidate_id, candidate in sorted(metrics.items()):
        if candidate_id != champion_id and 1.0 - candidate.oos_brier > 1.0 - best.oos_brier:
            best_id, best = candidate_id, candidate
    best_score = 1.0 - best.oos_brier
    promoted = (
        best_id != champion_id
        and best_score - champion_score >= promotion_margin
        and best.oos_stability >= best.cal_stability - stability_tolerance
    )
    return SelectionResult(
        agent_id=agent_id,
        selected_id=best_id if promoted else champion_id,
        promoted=promoted,
        champion_score=champion_score,
        best_score=best_score,
    )


def load_champion_configs(path: Path | str) -> dict[str, dict[str, Any]] | None:
    """Lädt ``champion_configs.json`` (``None``, wenn die Datei fehlt).

    Raises:
        ValueError: Datei vorhanden, aber kein JSON-Objekt.
    """
    file = Path(path)
    if not file.exists():
        return None
    data = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"champion_configs.json erwartet ein JSON-Objekt: {file}")
    return data


def load_champion_params(path: Path | str) -> dict[str, dict[str, float | int]]:
    """Lädt die Champion-Parametersätze aus ``champion_configs.json`` (fail-soft).

    Liefert ``{agent_id: params}`` für alle Agent-Blöcke mit einem gültigen
    ``params``-Mapping. Fehlende Datei, kein JSON-Objekt, Agent-Blöcke ohne
    ``params``-Mapping und nicht-zahlige Parameterwerte werden mit einer
    Warnung übersprungen — diese Funktion wirft nie gegen den Aufrufer
    aus (der Live-Betrieb darf an einer defekten Config nicht scheitern).
    """
    file = Path(path)
    if not file.exists():
        return {}
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("champion_configs.json nicht lesbar (%s) — ohne Parametersätze", exc)
        return {}
    if not isinstance(data, dict):
        logger.warning("champion_configs.json erwartet ein JSON-Objekt: %s", file)
        return {}
    params: dict[str, dict[str, float | int]] = {}
    for agent_id, block in data.items():
        if not isinstance(block, Mapping):
            logger.warning("champion_configs.json: Agent %r ohne Objekt-Block — übersprungen", agent_id)
            continue
        raw = block.get("params")
        if not isinstance(raw, Mapping) or not raw:
            logger.warning("champion_configs.json: Agent %r ohne gültiges params-Mapping — übersprungen", agent_id)
            continue
        values: dict[str, float | int] = {}
        valid = True
        for name, value in raw.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                logger.warning(
                    "champion_configs.json: Agent %r, Parameter %r nicht-zahlig — Satz übersprungen",
                    agent_id,
                    name,
                )
                valid = False
                break
            values[str(name)] = int(value) if isinstance(value, int) else float(value)
        if valid:
            params[str(agent_id)] = values
    return params


def build_configs_artifact(
    current: Mapping[str, tuple[Mapping[str, float | int], float]],
    previous: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Neues ``champion_configs.json``: pro Agent ``version``/``params``/``score``.

    Unveränderte Parametersätze behalten ihren Block und ihre Version;
    ein geänderter Parametersatz (Promotion) bumpt die Version um 1.
    """
    prev: Mapping[str, Mapping[str, Any]] = previous or {}
    artifact: dict[str, dict[str, Any]] = {}
    for agent_id, (params, score) in current.items():
        entry = prev.get(agent_id)
        if entry is not None and dict(entry.get("params") or {}) == dict(params):
            artifact[agent_id] = dict(entry)
        else:
            version = 1 if entry is None else int(entry.get("version", 0)) + 1
            artifact[agent_id] = {"version": version, "params": dict(params), "score": score}
    return artifact


def write_json_atomic(path: Path | str, payload: Mapping[str, Any]) -> Path:
    """Schreibt JSON atomar (tmp-Datei + ``Path.replace``) — Reader sehen nie halbe Stände."""
    out = Path(path)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(out)
    return out
