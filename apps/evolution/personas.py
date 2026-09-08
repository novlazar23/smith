"""LLM-Phase „Diskutieren": Personas schlagen preregistrierbare Hypothesen vor.

Ein Aufruf pro Zyklus, in dem drei Rollen diskutieren (Forscher, Skeptiker,
Risiko). Der LLM liefert ein JSON-Array von Vorschlägen, das hier gegen
das Zoo-Manifest (``param_specs``) und das Preregistrierungsschema validiert
wird. Ungültige Vorschläge werden verworfen (mit Log), nicht „repariert" —
Reparieren wäre bereits eine nachträgliche Einflussnahme.

Der LLM entscheidet **nichts**: er registriert nur Kandidaten; der Judge
(„Beurteilen") ist deterministisch und kennt den LLM nicht.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from packages.llm.client import LLMClient
from packages.llm.errors import LLMError
from packages.strategies import describe, list_strategies

from .models import NAME_PATTERN, DecisionRule, Proposal, TestPlan, Variant

logger = logging.getLogger(__name__)

#: Standard-Timeout für den Diskussion-Aufruf (Code-Generierung braucht
#: Minuten, nicht Sekunden).
PERSONA_TIMEOUT = 600.0

ALLOWED_INDICATORS = (
    "sma, ema, rsi, atr, rolling_vwap, bollinger, macd, stochastic, roc, "
    "donchian, keltner, supertrend, _rsi_series, _atr_series"
)

_SYSTEM_PROMPT = (
    "Du moderierst eine Strategies-Entwicklungs-Diskussion mit drei Rollen: "
    "FORSCHER (schlägt mechanismusplausible, falsifizierbare Hypothesen vor), "
    "SKEPTIKER (greift an: Overfitting, Kosten-Sensitivität, Clone-Verdacht "
    "gegenüber der Strategie-Bibliothek), RISIKO (fordert strenge "
    "Entscheidungsregeln: OOS-Marge, Max-Drawdown, Mindest-Trades).\n"
    "Harte Regeln: "
    "1. Maximal 3 Vorschläge; lieber 1 starker als 3 schwache. "
    "2. Vorschläge werden VOR dem Test PREREGISTRIERT und danach nur noch "
    "mechanisch beurteilt — nichts ist verhandelbar. "
    "3. 'config'-Vorschläge: family = bestehende Zoo-Strategie, params "
    "innerhalb des Manifests, keine Default-Werte (das wäre die Baseline). "
    "4. 'mechanism'-Vorschläge: family = neuer snake_case-Name, "
    "variant.code = vollständiger Quelltext einer einzigen RuleStrategy-"
    "Subklasse (Format s. user-Nachricht). "
    "5. Jeder Vorschlag braucht einen claim: 1-2 Sätze mit der mechanischen "
    "These (WARUM es funktionieren sollte). "
    "6. Antworte NUR mit einem JSON-Array (kein Markdown, keine Kommentare)."
)

_CODE_FORMAT = (
    "Format für mechanism-Code (exakt diese Imports, genau eine Klasse, "
    "kein 'from __future__'-Import — das Jail erlaubt nur typing/numpy/packages/relativ): "
    "from typing import ClassVar / "
    "from packages.backtesting.core import Candle / "
    "from packages.backtesting.strategies import SignalAction, StrategySignal / "
    "from . import indicators as ta / from ._common import RuleStrategy. "
    f"Verfügbare Indikatoren (ohne Lookahead): {ALLOWED_INDICATORS}. "
    "Verfügbares API: self.params, self._arrays() → (open, high, low, close, volume), "
    "self._signal(candle, action, conviction, reason), self._conviction(strength), "
    "self.n_bars_seen. Long-only: BUY = Entry, SELL = Exit. "
    "Fenster = die letzten 300 Kerzen (Vergangenheit + aktuelle Kerze). "
    "ClassVars: strategy_name, description, param_specs "
    "(dict name → (default, min, max)), min_bars (≥ längste Indikator-Periode + 2, ≤ 300)."
)


def _zoo_manifest() -> str:
    """Zoo-Strategien mit Parameter-Manifest (Kontext für die Personas)."""
    lines: list[str] = []
    for name in list_strategies():
        info = describe(name)
        params = ", ".join(
            f"{key}: {spec['default']:g} [{spec['min']:g}..{spec['max']:g}]"
            for key, spec in info["params"].items()
        ) or "keine"
        lines.append(f"- {name} ({info['min_bars']} Warmup): {params}")
    return "\n".join(lines)


def build_messages(digest: str, max_proposals: int) -> list[dict[str, str]]:
    """System- und User-Nachricht für den Diskussion-Aufruf."""
    user = (
        f"Maximale Anzahl Vorschläge: {max_proposals}.\n"
        f"{_CODE_FORMAT}\n\n"
        "Zoo-Strategien (Manifest): \n"
        f"{_zoo_manifest()}\n\n"
        f"Evidenz-Digest:\n{digest}\n\n"
        "Liefere jetzt das JSON-Array der Vorschläge."
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_proposals(raw: str) -> list[dict[str, Any]]:
    """Extrahiert das JSON-Array aus der LLM-Antwort (robust gegen Fences)."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end <= start:
        raise ValueError("kein JSON-Array in der Antwort gefunden")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, list):
        raise ValueError("Antwort ist kein JSON-Array")
    return [item for item in data if isinstance(item, dict)]


def _validate_config(item: dict[str, Any]) -> Proposal | None:
    family = item.get("family")
    if not isinstance(family, str) or family not in list_strategies():
        logger.warning("config-Proposal verworfen: unbekannte Strategie %r", family)
        return None
    params_raw = item.get("params") or {}
    if not isinstance(params_raw, dict) or not params_raw:
        logger.warning("config-Proposal verworfen: leere/ungültige params (%r)", family)
        return None
    specs = describe(family)["params"]
    params: dict[str, float] = {}
    for key, value in params_raw.items():
        if key not in specs or not isinstance(value, (int, float)):
            logger.warning("config-Proposal verworfen: Parameter %r=%r ungültig (%r)", key, value, family)
            return None
        lo, hi = float(specs[key]["min"]), float(specs[key]["max"])
        if not lo <= float(value) <= hi:
            logger.warning("config-Proposal verworfen: %s=%s außerhalb [%s,%s]", key, value, lo, hi)
            return None
        params[str(key)] = float(value)
    claim = str(item.get("claim", "")).strip()
    if len(claim) < 10:
        logger.warning("config-Proposal verworfen: claim zu kurz (%r)", family)
        return None
    try:
        return Proposal(
            family=family,
            kind="config",
            claim=claim,
            variant=Variant(strategy=family, params=params),
            test_plan=_optional_test_plan(item),
            decision_rule=_optional_decision_rule(item),
        )
    except Exception as exc:
        logger.warning("config-Proposal %r nicht validierbar: %s", family, exc)
        return None


def _validate_mechanism(item: dict[str, Any]) -> Proposal | None:
    family = item.get("family")
    if not isinstance(family, str) or not NAME_PATTERN.match(family):
        logger.warning("mechanism-Proposal verworfen: Name %r ungültig", family)
        return None
    code = item.get("code")
    if not isinstance(code, str) or len(code.strip()) < 100:
        logger.warning("mechanism-Proposal verworfen: Code fehlt oder zu kurz (%r)", family)
        return None
    claim = str(item.get("claim", "")).strip()
    if len(claim) < 10:
        logger.warning("mechanism-Proposal verworfen: claim zu kurz (%r)", family)
        return None
    try:
        return Proposal(
            family=family,
            kind="mechanism",
            claim=claim,
            variant=Variant(
                strategy=family,
                params={},
                code=code,
                code_file=f"packages/strategies/{family}.py",
            ),
            test_plan=_optional_test_plan(item),
            decision_rule=_optional_decision_rule(item),
        )
    except Exception as exc:
        logger.warning("mechanism-Proposal %r nicht validierbar: %s", family, exc)
        return None


def _optional_test_plan(item: dict[str, Any]) -> TestPlan | None:
    raw = item.get("test_plan")
    if not isinstance(raw, dict) or not raw:
        return None
    try:
        return TestPlan.model_validate(raw)
    except Exception as exc:
        logger.warning("test_plan verworfen (Defaults gelten): %s", exc)
        return None


def _optional_decision_rule(item: dict[str, Any]) -> DecisionRule | None:
    raw = item.get("decision_rule")
    if not isinstance(raw, dict) or not raw:
        return None
    try:
        return DecisionRule.model_validate(raw)
    except Exception as exc:
        logger.warning("decision_rule verworfen (Defaults gelten): %s", exc)
        return None


def propose(
    client: LLMClient,
    digest: str,
    *,
    max_proposals: int = 3,
) -> list[Proposal]:
    """Führt die Persona-Diskussion aus und liefert validierte Vorschläge.

    Alle Fehlerszenarien (LLM-Timeout, schlechtes JSON, ungültige Vorschläge)
    sind **nicht fatal**: der Zyklus fährt ohne LLM-Vorschläge weiter.
    """
    client.timeout = PERSONA_TIMEOUT
    try:
        raw = client.complete(build_messages(digest, max_proposals), temperature=0.2)
    except LLMError as exc:
        logger.warning("Persona-Diskussion fehlgeschlagen (%s: %s) — ohne LLM-Vorschläge", exc.code, exc)
        return []
    except Exception as exc:
        logger.warning("Persona-Diskussion unerwarteter Fehler (%s: %s)", type(exc).__name__, exc)
        return []

    try:
        items = parse_proposals(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("Persona-Antwort nicht parsbar: %s — ohne LLM-Vorschläge", exc)
        return []

    proposals: list[Proposal] = []
    for item in items[:max_proposals]:
        proposal = _validate_mechanism(item) if item.get("kind") == "mechanism" else _validate_config(item)
        if proposal is not None:
            proposals.append(proposal)
    logger.info("Persona-Diskussion: %d Vorschläge, %d valide", len(items), len(proposals))
    return proposals
