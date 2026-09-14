"""LLM-Phase für Stufe 2: Personas schlagen neue Agenten-Logik vor.

Gleiche Muster wie ``apps.evolution.personas``: ein Aufruf, in dem drei
Rollen diskutieren (Forscher, Skeptiker, Risiko). Der LLM liefert ein
JSON-Array von Vorschlägen (``name``/``claim``/``code``), das hier gegen
Name-Muster, Sperrliste und Mindestlängen validiert wird. Ungültige
Vorschläge werden verworfen (mit Log), nicht „repariert" — Reparieren
wäre bereits eine nachträgliche Einflussnahme.

Der LLM entscheidet **nichts**: er registriert nur Kandidaten; Zulassung
und Re-Prüfung sind deterministisch (``agent_evolve``). Alle
Fehlerszenarien (LLM-Timeout, schlechtes JSON, ungültige Vorschläge)
sind nicht-fatal: der Lauf fährt ohne LLM-Vorschläge weiter.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from packages.llm.client import LLMClient
from packages.llm.errors import LLMError

from .agent_sandbox import NAME_PATTERN

logger = logging.getLogger(__name__)

#: Standard-Timeout für den Vorschlags-Aufruf (Code-Generierung braucht
#: Minuten, nicht Sekunden).
PROPOSER_TIMEOUT = 600.0

_SYSTEM_PROMPT = (
    "Du moderierst eine Agenten-Entwicklungs-Diskussion mit drei Rollen: "
    "FORSCHER (schlägt mechanismusplausible, falsifizierbare Prognose-Logik "
    "für die Kursrichtung über einen 15-Minuten-Horizont vor), "
    "SKEPTIKER (greift an: Overfitting auf das aktuelle Fenster, Redundanz "
    "mit bestehenden Agenten, Lookahead-Fallen), "
    "RISIKO (fordert strenge Eingrenzung: konservative Wahrscheinlichkeiten, "
    "keine Extremwerte, klare Invalidierung). "
    "Harte Regeln: "
    "1. Maximal N Vorschläge; lieber 1 starker als 3 schwache. "
    "2. Vorschläge werden VOR dem Test registriert und danach nur noch "
    "mechanisch beurteilt — nichts ist verhandelbar. "
    "3. Jeder Vorschlag: name (snake_case, neu, nicht aus der Sperrliste), "
    "claim (1-2 Sätze: WARUM die Logik funktionieren sollte), "
    "code (vollständiger Quelltext der predict-Funktion, Format s. user-Nachricht). "
    "4. Antworte NUR mit einem JSON-Array (kein Markdown, keine Kommentare)."
)

_CODE_FORMAT = (
    "Format für die Agenten-Logik (exakt dieser Vertrag): eine einzige "
    "Funktion 'def predict(open, high, low, close, volume)' auf "
    "Modul-Ebene. Erlaubte Imports: 'import numpy as np' und optional "
    "'import math'. Die fünf Parameter sind float-NDArrays mit den "
    "letzten bis zu 200 Fünf-Minuten-Kerzen (aufsteigend, aktuellste Kerze "
    "zuletzt). Rückgabe: 3er-Tupel (p_up, p_down, p_range) — "
    "Wahrscheinlichkeiten dafür, dass der Kurs in den nächsten 15 Minuten "
    "steigt / fällt / seitwärts bleibt (alle ≥ 0, Summe > 0; die "
    "Normalisierung auf 1.0 erfolgt automatisch). Kein Lookahead: es gibt "
    "keine Daten ab der aktuellsten Kerze. Keine anderen Imports, keine "
    "Modulebene-Aussagen, keine Nebenwirkungen."
)


def build_messages(
    digest: str,
    taken_names: tuple[str, ...],
    max_candidates: int,
) -> list[dict[str, str]]:
    """System- und User-Nachricht für den Vorschlags-Aufruf."""
    user = (
        f"Maximale Anzahl Vorschläge: {max_candidates}.\n"
        f"{_CODE_FORMAT}\n\n"
        f"Besetzte Namen (Sperrliste, nicht wiederverwenden): {', '.join(sorted(taken_names)) or 'keine'}\n\n"
        f"Evidenz-Digest (aktuelle Ensemble-Performance, OOS = Out-of-Sample):\n{digest}\n\n"
        "Liefere jetzt das JSON-Array der Vorschläge."
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT.replace("N", str(max_candidates))},
        {"role": "user", "content": user},
    ]


def parse_agent_proposals(raw: str) -> list[dict[str, Any]]:
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


def _validate(item: dict[str, Any], taken_names: frozenset[str]) -> dict[str, str] | None:
    name = item.get("name")
    if not isinstance(name, str) or not NAME_PATTERN.match(name):
        logger.warning("Agent-Vorschlag verworfen: Name %r ungültig", name)
        return None
    if name in taken_names:
        logger.warning("Agent-Vorschlag verworfen: Name %r bereits besetzt", name)
        return None
    claim = str(item.get("claim", "")).strip()
    if len(claim) < 10:
        logger.warning("Agent-Vorschlag verworfen: claim zu kurz (%r)", name)
        return None
    code = item.get("code")
    if not isinstance(code, str) or len(code.strip()) < 100:
        logger.warning("Agent-Vorschlag verworfen: Code fehlt oder zu kurz (%r)", name)
        return None
    return {"name": name, "claim": claim, "code": code}


def propose(
    client: LLMClient,
    digest: str,
    taken_names: tuple[str, ...],
    *,
    max_candidates: int = 3,
) -> list[dict[str, str]]:
    """Führt die Persona-Diskussion aus und liefert validierte Vorschläge.

    Alle Fehlerszenarien sind **nicht fatal**: der Lauf fährt ohne
    LLM-Vorschläge weiter (leere Liste mit Warning-Log).
    """
    client.timeout = PROPOSER_TIMEOUT
    try:
        raw = client.complete(build_messages(digest, taken_names, max_candidates), temperature=0.3)
    except LLMError as exc:
        logger.warning("Agent-Proposer fehlgeschlagen (%s: %s) — ohne LLM-Vorschläge", exc.code, exc)
        return []
    except Exception as exc:
        logger.warning("Agent-Proposer unerwarteter Fehler (%s: %s)", type(exc).__name__, exc)
        return []

    try:
        items = parse_agent_proposals(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("Agent-Proposer-Antwort nicht parsbar: %s — ohne LLM-Vorschläge", exc)
        return []

    taken = frozenset(taken_names)
    proposals: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in items[:max_candidates]:
        proposal = _validate(item, taken)
        if proposal is not None and proposal["name"] not in seen:
            seen.add(proposal["name"])
            proposals.append(proposal)
    logger.info("Agent-Proposer: %d Vorschläge, %d valide", len(items), len(proposals))
    return proposals
