"""Champion-Challenger-Feed: Evaluations-Artefakt → Status-Overrides.

Der Feed liest ein externes Evaluations-Artefakt (JSON), das pro Agent die
Metriken der Champion- und Challenger-Version enthält, leitet über den
``AgentOptimizer`` die Status-Overrides ab und macht sie für das Ensemble
verfügbar.

Das Artefakt wird von einem separaten Evaluations-Job erzeugt (Backtest- oder
Live-Bewertung). Der Feed erfindet keine Metriken — er ist der reine Adapter
zwischen Evaluations-Ergebnissen und ``AgentOptimizer``/Ensemble.

Artefakt-Format (JSON-Objekt, Key = ``agent_id``)::

    {
      "trend": {
        "champion":   {"version": "v1", "oos_score": 0.70, "calibration_score": 0.80,
                       "stability_score": 0.95, "marginal_contribution": 0.01,
                       "shadow_days": 10, "samples": 100},
        "challenger": {"version": "v2", "oos_score": 0.75, "calibration_score": 0.82,
                       "stability_score": 0.94, "marginal_contribution": 0.02,
                       "shadow_days": 10, "samples": 100},
        "new_risks": [],
        "shadow_success": true
      }
    }

``new_risks`` und ``shadow_success`` sind optional (Default ``[]`` bzw.
``True``) und entsprechen den Defaults von ``AgentVersionPair``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from packages.governance.champion_challenger import (
    AgentOptimizer,
    AgentVersion,
    AgentVersionPair,
    ChampionChallengerConfig,
)
from packages.schemas.agent_report import AgentStatus

# Re-Kalibrierung: champion=Kalibrierungs-, challenger=OOS-Fenster desselben
# Agenten (keine echten Challenger-Varianten). Negativer Schwellwert erlaubt
# bis 0,05 OOS-Abfall (Score=Brier), ohne zu degradieren; nur >0,05 → SHADOW.
REQUALIFICATION_CONFIG = ChampionChallengerConfig(min_oos_improvement=-0.05)


def _version(agent_id: str, role: str, raw: Mapping[str, Any]) -> AgentVersion:
    """Baut eine ``AgentVersion`` aus einem Artefakt-Block.

    Fehlende Metrik-Felder fallen auf die ``AgentVersion``-Defaults zurück;
    ``version`` wird als String mit Default ``"unknown"`` übernommen.
    """
    if not isinstance(raw, Mapping):
        raise ValueError(f"Agent '{agent_id}': '{role}'-Block muss ein Objekt sein")
    return AgentVersion(
        agent_id=agent_id,
        version=str(raw.get("version", "unknown")),
        oos_score=float(raw.get("oos_score", 0.0)),
        calibration_score=float(raw.get("calibration_score", 0.0)),
        stability_score=float(raw.get("stability_score", 1.0)),
        marginal_contribution=float(raw.get("marginal_contribution", 0.0)),
        shadow_days=int(raw.get("shadow_days", 0)),
        samples=int(raw.get("samples", 0)),
    )


def load_version_pairs(path: Path | str) -> dict[str, AgentVersionPair]:
    """Liest das Artefakt und liefert pro ``agent_id`` ein ``AgentVersionPair``.

    Args:
        path: Pfad zum JSON-Artefakt.

    Raises:
        ValueError: Wenn das Artefakt kein JSON-Objekt ist, ein Agent-Block
            kein Objekt ist oder die ``champion``/``challenger``-Blöcke fehlen.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Evaluations-Artefakt muss ein JSON-Objekt sein")
    pairs: dict[str, AgentVersionPair] = {}
    for agent_id, block in data.items():
        key = str(agent_id)
        if not isinstance(block, Mapping):
            raise ValueError(f"Agent '{key}': Block muss ein Objekt sein")
        if "champion" not in block or "challenger" not in block:
            raise ValueError(f"Agent '{key}': 'champion' und 'challenger' erforderlich")
        pairs[key] = AgentVersionPair(
            champion=_version(key, "champion", block["champion"]),
            challenger=_version(key, "challenger", block["challenger"]),
            new_risks=tuple(str(r) for r in (block.get("new_risks") or ())),
            shadow_success=bool(block.get("shadow_success", True)),
        )
    return pairs


def load_status_overrides(
    path: Path | str,
    config: ChampionChallengerConfig | None = None,
) -> dict[str, AgentStatus]:
    """Leitet aus dem Artefakt die Status-Overrides pro Agent ab.

    ``config=None`` nutzt die Standard-Promotion-Konfiguration (Challenger
    muss Champion OOS um mindestens ``min_oos_improvement`` schlagen). Für
    die Re-Kalibrierung des laufenden Ensembles ``REQUALIFICATION_CONFIG``
    übergeben, damit ein stabiler Agent (OOS ≈ Kalibrierung) ``ACTIVE`` bleibt.

    Args:
        path: Pfad zum JSON-Artefakt.
        config: Champion-Konfiguration für den Optimizer (None = Default).

    Returns:
        Mapping ``agent_id`` → empfohlener ``AgentStatus`` (leer bei leerem
        Artefakt).
    """
    pairs = load_version_pairs(path)
    if not pairs:
        return {}
    optimizer = AgentOptimizer(config)
    decisions = optimizer.optimize_many(pairs)
    return dict(optimizer.status_overrides(decisions))
