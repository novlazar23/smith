"""LLM-Phase für Stufe 2: ein Persona-Panel schlägt neue Agenten-Logik vor.

Jede Persona aus ``PERSONAS`` (unterschiedliche Markt-Priors: Trend,
Reversion, Mikrostruktur, Regime) erhält **einen eigenen LLM-Aufruf**
mit ihrem Prior und schlägt darin Kandidaten vor (``name``/``claim``/
``code``). Die Vorschläge aller Personas werden gemergt, nach Namen
dedupliziert und auf ``max_candidates`` gekappt — die Diversifikation
der Priors stattet das Gate-Panel mit unabhängigeren Kandidaten aus.
Der Skeptiker-Teil der Diskussion entfällt als LLM-Rolle, weil die
deterministischen Gates (OOS/LOO/Stabilität in ``agent_evolve``) diese
Funktion besser und nachprüfbar übernehmen.

Der LLM liefert pro Persona ein JSON-Array, das hier gegen
Name-Muster, Sperrliste und Mindestlängen validiert wird. Ungültige
Vorschläge werden verworfen (mit Log), nicht „repariert" — Reparieren
wäre bereits eine nachträgliche Einflussnahme.

Der LLM entscheidet **nichts**: er registriert nur Kandidaten; Zulassung
und Re-Prüfung sind deterministisch (``agent_evolve``). Alle
Fehlerszenarien (LLM-Timeout, schlechtes JSON, ungültige Vorschläge)
sind nicht-fatal — pro Persona: der Lauf fährt mit den Vorschlägen der
übrigen Personas weiter, bei Totalausfall ohne Vorschläge.
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

#: Das Persona-Panel: (id, Markt-Prior). Jede Persona bekommt einen
#: eigenen LLM-Aufruf — der Prior ist eine Arbeits-Hypothese, keine
#: Tatsache (im Prompt so erklärt, damit nicht der Prior selbst
#: implementiert wird).
PERSONAS: tuple[tuple[str, str], ...] = (
    (
        "trend",
        "Kursbewegungen zeigen kurzfristige Autokorrelation: Momentum, "
        "Brüche und Trend-Struktur haben über einen 15-Minuten-Horizont "
        "Richtungsvorhersagekraft.",
    ),
    (
        "reversion",
        "Übertriebene 15-Minuten-Bewegungen revertieren zum Mittelwert: "
        "Extreme (ausgeprägte Z-Score-/Bollinger-Bänder, RSI-Extremzonen) "
        "vorhersagen eher Gegenbewegung als Fortsetzung.",
    ),
    (
        "mikrostruktur",
        "Volumen- und Orderflow-Signale vorwegnehmen die Preisbewegung: "
        "Up/Down-Volumen-Asymmetrie, OBV-Drift und Range-Nutzung "
        "(Teilnahme) sind vorwärts-indikativ für die nächste Bewegung.",
    ),
    (
        "regime",
        "Der Markt verbringt die meisten Kerzen seitwärts in "
        "Volatilitäts-Clustern: Squeeze- und Regimewechsel-Vorläufer "
        "trennen die Wahrscheinlichkeit für Richtungs- von der für "
        "Seitwärts-Ausbrüche.",
    ),
)

#: Gemeinsame harte Regeln für alle Persona-Aufrufe (N = Platzhalter
#: für die Vorschlags-Obergrenze).
_CONTRACT = (
    "Harte Regeln: "
    "1. Maximal N Vorschläge; lieber 1 starker als N schwache. "
    "2. Vorschläge werden VOR dem Test registriert und danach nur noch "
    "mechanisch beurteilt — nichts ist verhandelbar. Overfitting auf "
    "das aktuelle Fenster, Redundanz mit bestehenden Agenten und "
    "Lookahead-Fallen sind Ausschlusskriterien. "
    "3. Jeder Vorschlag: name (snake_case, neu, nicht aus der Sperrliste), "
    "claim (1-2 Sätze: WARUM die Logik funktionieren sollte), "
    "code (vollständiger Quelltext der predict-Funktion, Format s. "
    "user-Nachricht). "
    "4. Konservative Wahrscheinlichkeiten, keine Extremwerte, klare "
    "Invalidierung. "
    "5. Antworte NUR mit einem JSON-Array (kein Markdown, keine Kommentare)."
)

_ROLE_TEMPLATE = (
    "Du bist die {name}-Persona in einer Agenten-Entwicklungs-Diskussion "
    "und schlägst mechanismusplausible, falsifizierbare Prognose-Logik "
    "für die Kursrichtung über einen 15-Minuten-Horizont vor. "
    "Dein Markt-Prior: {prior} Behandle den Prior als Hypothese, nicht "
    "als Tatsache — die Logik muss ohne ihn plausibel bleiben."
)

#: Legacy-Prompt (``build_messages`` ohne ``role``): ein Aufruf, in dem
#: drei Rollen simuliert diskutieren.
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
    *,
    role: str | None = None,
) -> list[dict[str, str]]:
    """System- und User-Nachricht für den Vorschlags-Aufruf.

    Mit ``role`` (einer Persona-ID aus ``PERSONAS``) erhält der Aufruf
    den Persona-Prompt; ohne bleibt der Legacy-Diskussions-Prompt.
    """
    user = (
        f"Maximale Anzahl Vorschläge: {max_candidates}.\n"
        f"{_CODE_FORMAT}\n\n"
        f"Besetzte Namen (Sperrliste, nicht wiederverwenden): {', '.join(sorted(taken_names)) or 'keine'}\n\n"
        f"Evidenz-Digest (aktuelle Ensemble-Performance, OOS = Out-of-Sample):\n{digest}\n\n"
        "Liefere jetzt das JSON-Array der Vorschläge."
    )
    if role is None:
        system = _SYSTEM_PROMPT
    else:
        prior = dict(PERSONAS)[role]
        system = _ROLE_TEMPLATE.format(name=role, prior=prior) + " " + _CONTRACT
    return [
        {"role": "system", "content": system.replace("N", str(max_candidates))},
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
    personas: tuple[tuple[str, str], ...] = PERSONAS,
) -> list[dict[str, str]]:
    """Führt das Persona-Panel aus und liefert validierte Vorschläge.

    Jede Persona erhält einen eigenen LLM-Aufruf; die Vorschläge werden
    gemergt, nach Namen dedupliziert und auf ``max_candidates`` gekappt.
    Fehlerszenarien sind pro Persona **nicht fatal** (Warning-Log,
    übrige Personas laufen weiter); bei Totalausfall: leere Liste.
    """
    client.timeout = PROPOSER_TIMEOUT
    taken = frozenset(taken_names)
    # ponytail: sequentielle Persona-Aufrufe (4 x ~20-60 s im Daily-Lauf);
    # bei Engpass im Zeitbudget auf ThreadPoolExecutor umstellen.
    per_persona = max(1, max_candidates // max(1, len(personas)))
    proposals: list[dict[str, str]] = []
    seen: set[str] = set()
    for persona in personas:
        persona_id = persona[0]
        try:
            raw = client.complete(
                build_messages(digest, taken_names, per_persona, role=persona_id),
                temperature=0.3,
            )
        except LLMError as exc:
            logger.warning("Agent-Proposer[%s] fehlgeschlagen (%s: %s)", persona_id, exc.code, exc)
            continue
        except Exception as exc:
            logger.warning(
                "Agent-Proposer[%s] unerwarteter Fehler (%s: %s)", persona_id, type(exc).__name__, exc
            )
            continue
        try:
            items = parse_agent_proposals(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("Agent-Proposer[%s]-Antwort nicht parsbar: %s", persona_id, exc)
            continue
        accepted = 0
        for item in items[:per_persona]:
            proposal = _validate(item, taken)
            if proposal is not None and proposal["name"] not in seen:
                seen.add(proposal["name"])
                proposals.append(proposal)
                accepted += 1
        logger.info("Agent-Proposer[%s]: %d Vorschläge, %d neu/valide", persona_id, len(items), accepted)
        if len(proposals) >= max_candidates:
            break
    return proposals[:max_candidates]
