"""Parameter-Registry für die eval-seitige Agenten-Evolution.

Mappt die vier regelbasierten ACTIVE-Agenten auf ihre Parameter-Datenklassen
(inkl. Suchraum) und baut Agent-Instanzen mit einem gegebenen Parametersatz.
Die Defaults der Datenklassen entsprechen den bisherigen harten Konstanten
(byte-identisches Verhalten, durch den Capturing-Lauf verifiziert).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from packages.agents.base import BaseAgent, BaseParams, ParamSpace
from packages.agents.mean_reversion_agent import MeanReversionAgent, MeanReversionParams
from packages.agents.trend_agent import TrendAgent, TrendParams
from packages.agents.volatility_regime_agent import VolatilityRegimeAgent, VolatilityRegimeParams
from packages.agents.volume_conviction_agent import VolumeConvictionAgent, VolumeConvictionParams

# agent_id → Parameter-Datenklasse (stabile Reihenfolge = Ensemble-Reihenfolge)
PARAM_CLASSES: dict[str, type[BaseParams]] = {
    "trend": TrendParams,
    "mean_reversion": MeanReversionParams,
    "volatility_regime": VolatilityRegimeParams,
    "volume_conviction": VolumeConvictionParams,
}

# Stabile Agenten-Reihenfolge (Iteration für Evolution und Summary)
AGENT_TYPES: tuple[str, ...] = tuple(PARAM_CLASSES)


def default_params(agent_id: str) -> BaseParams:
    """Standardsatz (bisherige harte Konstanten) eines Agenten-Typs."""
    return PARAM_CLASSES[agent_id]()


def param_space(agent_id: str) -> ParamSpace:
    """Suchraum pro Feld: (Typ "int"/"float", Minimum, Maximum, Schrittweite)."""
    return dict(PARAM_CLASSES[agent_id].PARAM_SPACE)


def build_agent(agent_id: str, params: BaseParams | Mapping[str, Any] | None = None) -> BaseAgent:
    """Agent-Instanz mit Parametersatz (``None`` → Defaults, Dict → ``from_dict``).

    Raises:
        KeyError: Unbekannter Agenten-Typ.
    """
    if params is None or isinstance(params, Mapping):
        values: dict[str, float | int] = dict(params) if params is not None else {}
    else:
        values = dict(params.to_dict())
    match agent_id:
        case "trend":
            return TrendAgent(params=TrendParams.from_dict(values))
        case "mean_reversion":
            return MeanReversionAgent(params=MeanReversionParams.from_dict(values))
        case "volatility_regime":
            return VolatilityRegimeAgent(params=VolatilityRegimeParams.from_dict(values))
        case "volume_conviction":
            return VolumeConvictionAgent(params=VolumeConvictionParams.from_dict(values))
    raise KeyError(f"Unbekannter Agenten-Typ: {agent_id!r}")
