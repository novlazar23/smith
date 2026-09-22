"""Shadow-Survival-Tracker für zugelassene Evolved Agents (Stufe 2).

Die Promotion SHADOW → ACTIVE ist ein manueller Schritt mit drei
preregistrierten Kriterien (siehe README). Dieser Tracker protokolliert
die ersten beiden mechanisch — ohne je auf Zulassung oder Selektion
Einfluss zu nehmen (Fail-Soft wie das Stufe-1-Shadow-Log):

1. **Survival**: Anzahl der durchgehenden täglichen Re-Checks, die der
   Agent bestanden hat (``consecutive_passes``). Ab
   ``PROMOTION_MIN_PASSES`` (14 Tage) ist Kriterium 1 erfüllt
   (``promotion_status`` → ``ready=True``).
2. **Stabilität**: OOS-Score-History der letzten ``MAX_SCORE_HISTORY``
   Beobachtungsläufe — Grundlage für den manuellen Stabilitäts-Review
   (Kriterium 2); wird nur angezeigt, kein zusätzliches Gate.

Zustand im Sidecar ``evolved_agents_shadow.json`` (neben
``evolved_agents.json`` auf dem Shared-Volume). Der Orchestrator-
Vertrag von ``evolved_agents.json`` bleibt unverändert.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SHADOW_FILENAME = "evolved_agents_shadow.json"
#: Mindestzahl durchgehender täglicher Re-Checks für Kriterium 1
#: (README: "übersteht mindestens 14 Tage durchgehend die täglichen
#: Re-Checks").
PROMOTION_MIN_PASSES = 14
#: Länge der Score-History (neueste Läufe, älteste fallen ab).
MAX_SCORE_HISTORY = 30


def load_shadow_state(path: str | Path) -> dict[str, dict[str, Any]]:
    """Lädt den Shadow-Status fail-soft (fehlend/korrupt → leeres Dict)."""
    file = Path(path)
    if not file.exists():
        return {}
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("Shadow-Tracker %s nicht lesbar — starte leer", file.name, exc_info=True)
        return {}
    if not isinstance(data, dict):
        return {}
    return {name: entry for name, entry in data.items() if isinstance(entry, dict)}


def update_shadow_state(
    state: Mapping[str, Mapping[str, Any]],
    artifact: Mapping[str, Mapping[str, Any]],
    previous: Mapping[str, Mapping[str, Any]],
    run_at: str,
) -> dict[str, dict[str, Any]]:
    """Neuer Shadow-Status nach einem Lauf (reine Funktion).

    Args:
        state: bisheriger Status (``load_shadow_state``).
        artifact: neues ``evolved_agents.json`` (``name → entry`` mit
            ``code``/``score``/``admitted_at``) — die Überlebenden.
        previous: Bestand vor dem Lauf (Code-Vergleich erkennt
            geänderte Agenten, deren Streak neu startet).
        run_at: ISO-Zeitstempel dieses Laufs.
    """
    new_state: dict[str, dict[str, Any]] = {}
    for name, entry in artifact.items():
        score = float(entry.get("score", 0.0))
        old = state.get(name) or {}
        prev_entry = previous.get(name)
        same_agent = prev_entry is not None and prev_entry.get("code") == entry.get("code")
        if old and same_agent:
            passes = int(old.get("consecutive_passes", 0)) + 1
        else:
            passes = 0  # neu zugelassen (oder Code geändert) → Streak neu
            if old:
                logger.info("Shadow-Tracker: Agent %s mit geändertem Code — Streak neu", name)
        history: list[float] = [float(s) for s in (old.get("score_history") or [])]
        history.append(score)
        new_state[name] = {
            "admitted_at": str(entry.get("admitted_at", run_at)),
            "consecutive_passes": passes,
            "score_history": history[-MAX_SCORE_HISTORY:],
            "last_checked": run_at,
        }
    for name in state:
        if name not in artifact:
            logger.info("Shadow-Tracker: Agent %s entfernt (nicht mehr im Ensemble)", name)
    return new_state


def promotion_status(state: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Pro Agent: Re-Check-Zähler, Readiness (Kriterium 1) und
    Score-Spread (Basis für den Stabilitäts-Review, Kriterium 2)."""
    out: dict[str, dict[str, Any]] = {}
    for name, entry in state.items():
        history = [float(s) for s in (entry.get("score_history") or [])]
        passes = int(entry.get("consecutive_passes", 0))
        out[name] = {
            "passes": passes,
            "ready": passes >= PROMOTION_MIN_PASSES,
            "min_score": min(history) if history else None,
            "max_score": max(history) if history else None,
            "spread": (max(history) - min(history)) if history else None,
        }
    return out


def write_shadow_state(path: str | Path, state: Mapping[str, Mapping[str, Any]]) -> Path:
    """Atomares Schreiben (selbes Muster wie die übrigen Artefakte)."""
    from .evolve import write_json_atomic

    return write_json_atomic(path, state)
