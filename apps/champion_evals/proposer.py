"""LLM-Phase für Stufe 2: Persona-Panel mit Skeptiker-Diskussion.

Dreirundige Diskussion (Muster: multi-agent deliberation):

1. **Vorschlag:** Jede Persona aus ``PERSONAS`` (unterschiedliche
   Markt-Priors) erhält einen eigenen LLM-Aufruf mit ihrem Prior und
   schlägt Kandidaten vor (``name``/``claim``/``code``). Die Aufrufe
   laufen parallel (``ThreadPoolExecutor``); die Vorschläge werden in
   Persona-Reihenfolge gemergt, nach Namen dedupliziert und auf
   ``max_candidates`` gekappt.
2. **Kritik:** Ein SKEPTIKER-Aufruf bewertet alle Pool-Kandidaten auf
   Overfitting, Redundanz, Lookahead und Fragilität. Die Kritik ist
   **advisory** — der SKEPTIKER entscheidet nichts (Zulassung bleiben
   die deterministischen Gates in ``agent_evolve``), er liefert nur
   Verbesserungsvorschläge an Runde 3.
3. **Revision:** Jede Persona überarbeitet ihren (kritisierten)
   Vorschlag einmalig; ist die Kritik unbegründet oder die Revision
   ungültig, bleibt der Originalvorschlag stehen (Name bleibt bei
   allen Revisionen unverändert — Preregistrierungs-Identität).

Alle Runden sind fail-soft: eine ausgefallene Persona/Kritik/Revision
kostet nur ihren eigenen Beitrag; bei Totalausfall läuft der Lauf
ohne LLM-Vorschläge weiter. Der LLM liefert pro Aufruf ein
JSON-Array, das gegen Name-Muster, Sperrliste und Mindestlängen
validiert wird — ungültige Vorschläge werden verworfen (mit Log),
nicht „repariert" (Reparieren wäre bereits nachträgliche
Einflussnahme).
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from packages.llm.client import LLMClient
from packages.llm.errors import LLMError

from .agent_sandbox import NAME_PATTERN

logger = logging.getLogger(__name__)

#: Standard-Timeout für einen Vorschlags-/Kritik-/Revisions-Aufruf
#: (Code-Generierung braucht Minuten, nicht Sekunden).
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
    (
        "ereignis",
        "Ereignisse hinterlassen Signaturen im Preisband: Volumen-Spitzen "
        "mit ungewöhnlicher Range-Expansion (Nachrichten, Liquiditäts-"
        "Schocks) werden danach häufiger fortgesetzt oder revertiert — "
        "Spike-Größe, Wick-Struktur und das Verhalten nach dem Spike "
        "sind vorwärts-indikativ.",
    ),
    (
        "liquiditaet",
        "Kursbewegungen jagen Liquidität: lange Dochte über vorherige "
        "Extrema sind Stop-Läufe und kehren oft zurück; Docht-Länge, "
        "Docht-Seite und das Verhältnis von Kerzenkörper zur Range "
        "zeigen, welche Seite Liquidität absorbiert.",
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

_SKEPTIC_SYSTEM = (
    "Du bist der SKEPTIKER in einer Agenten-Entwicklungs-Diskussion. "
    "Kritisier jeden vorliegenden Agenten-Vorschlag konkret auf: "
    "Overfitting auf das aktuelle Fenster, Redundanz mit bestehenden "
    "Agenten, Lookahead-Fallen, numerische Fragilität (Division durch "
    "~0, Extremwerte) und fehlende Invalidierung. Nenne pro Vorschlag "
    "genau das, was überarbeitet werden sollte. Du entscheidest nichts "
    "und darfst keine Vorschläge verwerfen — deine Kritik dient nur "
    "der Überarbeitung durch die jeweiligen Personas. "
    "Antworte NUR mit einem JSON-Array von {name, critique} "
    "(1-3 Sätze pro critique, kein Markdown, keine Kommentare)."
)

_REFINE_SYSTEM = (
    "Du überarbeitest einen Agenten-Vorschlag nach Kritik des SKEPTIKERS. "
    "Ist die Kritik berechtigt (Overfitting auf das aktuelle Fenster, "
    "Redundanz mit bestehenden Agenten, Lookahead, numerische Fragilität, "
    "fehlende Invalidierung), überarbeite den Code einmalig und gezielt; "
    "ist sie unbegründet, liefere den Vorschlag unverändert. Der Name "
    "bleibt in jedem Fall unverändert. Das Code-Format gilt weiterhin "
    "(def predict(open, high, low, close, volume, timestamps) -> "
    "(p_up, p_down, p_range)). Antworte NUR mit einem JSON-Array mit "
    "genau einem Element ({name, claim, code}); kein Markdown, keine "
    "Kommentare."
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
    "Funktion 'def predict(open, high, low, close, volume, timestamps)' "
    "auf Modul-Ebene. Erlaubte Imports: 'import numpy as np' und optional "
    "'import math'. Die ersten fünf Parameter sind float-NDArrays mit den "
    "letzten bis zu 200 Fünf-Minuten-Kerzen (aufsteigend, aktuellste Kerze "
    "zuletzt); 'timestamps' ist ein int64-NDArray mit dem Unix-Zeitpunkt "
    "jeder Kerze in Nanosekunden (UTC) — z. B. Tageszeit über "
    "'(timestamps // 1_000_000_000) % 86400' (Sekunden im UTC-Tag). "
    "Rückgabe: 3er-Tupel (p_up, p_down, p_range) — Wahrscheinlichkeiten "
    "dafür, dass der Kurs in den nächsten 15 Minuten steigt / fällt / "
    "seitwärts bleibt (alle ≥ 0, Summe > 0; die Normalisierung auf 1.0 "
    "erfolgt automatisch). Kein Lookahead: es gibt keine Daten ab der "
    "aktuellsten Kerze. Keine anderen Imports, keine Modulebene-Aussagen, "
    "keine Nebenwirkungen."
)


def build_messages(
    digest: str,
    taken_names: tuple[str, ...],
    max_candidates: int,
    *,
    role: str | None = None,
    archive_digest: str = "",
) -> list[dict[str, str]]:
    """System- und User-Nachricht für den Vorschlags-Aufruf.

    Mit ``role`` (einer Persona-ID aus ``PERSONAS``) erhält der Aufruf
    den Persona-Prompt; ohne bleibt der Legacy-Diskussions-Prompt.
    Nicht-leeres ``archive_digest`` (Gate-Reflexion aus dem
    Kandidaten-Archiv) ergänzt eine Sektion zwischen Sperrliste und
    Evidenz-Digest; leer (Default) bleibt der Prompt unverändert.
    """
    archive_section = (
        "Bekannte frühere Kandidaten aus dem Archiv (Gate-Reflexion): "
        "nicht erneut vorschlagen; Near-Misses (Score knapp unter der "
        f"Hurdle) gezielt reparieren statt neu zu würfeln:\n{archive_digest}\n\n"
        if archive_digest
        else ""
    )
    user = (
        f"Maximale Anzahl Vorschläge: {max_candidates}.\n"
        f"{_CODE_FORMAT}\n\n"
        f"Besetzte Namen (Sperrliste, nicht wiederverwenden): {', '.join(sorted(taken_names)) or 'keine'}\n\n"
        + archive_section
        + f"Evidenz-Digest (aktuelle Ensemble-Performance, OOS = Out-of-Sample):\n{digest}\n\n"
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


def _balanced(text: str, start: int, open_ch: str, close_ch: str) -> str | None:
    """Balanciertes Bracket-Matching von ``text[start]`` (Öffner) aus,
    Strings und Escapes ausgenommen."""
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_agent_proposals(raw: str) -> list[dict[str, Any]]:
    """Extrahiert das JSON-Array (oder ein einzelnes JSON-Objekt) aus der
    LLM-Antwort (robust gegen Fences und Prosa).

    Balanciertes Bracket-Matching von Kandidatenpositionen statt
    erstes/letztes Bracket: Prosa mit Klammern vor oder nach dem JSON
    bricht die Extraktion nicht mehr (Produktionsfehler der
    Refine-Runde: ``[siehe …]``-Prosa, ``rfind(']')``). Das Modell
    liefert die Antwort teils als einzelnes Objekt statt Array
    (beobachtet in der Refine-Runde) — wird akzeptiert und in eine
    Liste verpackt.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    for open_ch, close_ch, first in (("[", "]", '{"'), ("{", "}", '"')):
        pos = 0
        while True:
            start = text.find(open_ch, pos)
            if start < 0:
                break
            pos = start + 1
            j = start + 1
            while j < len(text) and text[j].isspace():
                j += 1
            if j >= len(text) or text[j] not in first:
                continue
            candidate = _balanced(text, start, open_ch, close_ch)
            if candidate is None:
                break
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict)]
            if isinstance(data, dict):
                return [data]
    raise ValueError("kein JSON-Array/-Objekt in der Antwort gefunden")


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


def _round_proposals(
    client: LLMClient,
    digest: str,
    taken_names: tuple[str, ...],
    taken: frozenset[str],
    per_persona: int,
    personas: tuple[tuple[str, str], ...],
    archive_digest: str = "",
) -> list[dict[str, str]]:
    """Runde 1: Jede Persona schlägt parallel vor; Merge in Persona-Reihenfolge.

    Jeder überlebende Vorschlag trägt seine ``persona`` (ID) — der
    Aufrufer (``agent_evolve``) nutzt sie für die Archiv-Zuordnung.
    """

    def call_one(persona_id: str) -> list[dict[str, str]]:
        try:
            raw = client.complete(
                build_messages(digest, taken_names, per_persona, role=persona_id, archive_digest=archive_digest),
                temperature=0.3,
            )
        except LLMError as exc:
            logger.warning("Agent-Proposer[%s] fehlgeschlagen (%s: %s)", persona_id, exc.code, exc)
            return []
        except Exception as exc:
            logger.warning(
                "Agent-Proposer[%s] unerwarteter Fehler (%s: %s)", persona_id, type(exc).__name__, exc
            )
            return []
        try:
            items = parse_agent_proposals(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("Agent-Proposer[%s]-Antwort nicht parsbar: %s", persona_id, exc)
            return []
        out: list[dict[str, str]] = []
        for item in items[:per_persona]:
            proposal = _validate(item, taken)
            if proposal is not None:
                proposal["persona"] = persona_id
                out.append(proposal)
        logger.info("Agent-Proposer[%s]: %d Vorschläge, %d valide", persona_id, len(items), len(out))
        return out

    merged: list[dict[str, str]] = []
    seen: set[str] = set()
    with ThreadPoolExecutor(max_workers=max(1, len(personas))) as executor:
        futures = [executor.submit(call_one, persona[0]) for persona in personas]
        for future in futures:
            for proposal in future.result():
                if proposal["name"] not in seen:
                    seen.add(proposal["name"])
                    merged.append(proposal)
    return merged


def _round_critique(client: LLMClient, digest: str, pool: list[dict[str, str]]) -> dict[str, str]:
    """Runde 2: SKEPTIKER kritisiert alle Pool-Kandidaten (advisory)."""
    user = (
        f"Evidenz-Digest (aktuelles Basis-Ensemble, OOS = Out-of-Sample):\n{digest}\n\n"
        "Vorliegende Vorschläge:\n"
        f"{json.dumps(pool, ensure_ascii=False, indent=2)}\n\n"
        "Liefere jetzt das JSON-Array der Kritiken ({name, critique})."
    )
    try:
        raw = client.complete(
            [
                {"role": "system", "content": _SKEPTIC_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
    except LLMError as exc:
        logger.warning("Agent-Proposer[skeptiker] fehlgeschlagen (%s: %s)", exc.code, exc)
        return {}
    except Exception as exc:
        logger.warning("Agent-Proposer[skeptiker] unerwarteter Fehler (%s: %s)", type(exc).__name__, exc)
        return {}
    try:
        items = parse_agent_proposals(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("Agent-Proposer[skeptiker]-Antwort nicht parsbar: %s", exc)
        return {}
    critiques = {
        item["name"]: item["critique"].strip()
        for item in items
        if isinstance(item.get("name"), str)
        and isinstance(item.get("critique"), str)
        and len(item["critique"].strip()) >= 10
    }
    logger.info("Agent-Proposer[skeptiker]: %d Kritiken für %d Vorschläge", len(critiques), len(pool))
    return critiques


def _round_refine(
    client: LLMClient,
    pool: list[dict[str, str]],
    critiques: dict[str, str],
    taken: frozenset[str],
) -> list[dict[str, str]]:
    """Runde 3: Jede Persona revidiert ihren kritisierten Vorschlag (parallel)."""

    def refine_one(proposal: dict[str, str]) -> dict[str, str]:
        critique = critiques.get(proposal["name"])
        if critique is None:
            return proposal
        user = (
            "Ursprünglicher Vorschlag:\n"
            f"{json.dumps(proposal, ensure_ascii=False, indent=2)}\n\n"
            f"Kritik des SKEPTIKERS:\n{critique}\n\n"
            "Liefere jetzt das überarbeitete JSON-Array."
        )
        try:
            raw = client.complete(
                [
                    {"role": "system", "content": _REFINE_SYSTEM},
                    {"role": "user", "content": user},
                ],
                temperature=0.3,
            )
        except LLMError as exc:
            logger.warning(
                "Agent-Proposer[refine:%s] fehlgeschlagen (%s: %s) — Vorschlag unverändert",
                proposal["name"],
                exc.code,
                exc,
            )
            return proposal
        except Exception as exc:
            logger.warning(
                "Agent-Proposer[refine:%s] unerwarteter Fehler (%s: %s) — Vorschlag unverändert",
                proposal["name"],
                type(exc).__name__,
                exc,
            )
            return proposal
        try:
            items = parse_agent_proposals(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            # Extrakt der Rohantwort (diagnostisch: Trunkation vs. Fehlbildung vs. Prosa)
            logger.warning(
                "Agent-Proposer[refine:%s]-Antwort nicht parsbar: %s (Extrakt: %.300r) — Vorschlag unverändert",
                proposal["name"],
                exc,
                raw,
            )
            return proposal
        if not items:
            logger.warning("Agent-Proposer[refine:%s]: leere Antwort — Vorschlag unverändert", proposal["name"])
            return proposal
        revised = _validate(items[0], taken)
        if revised is None:
            logger.warning(
                "Agent-Proposer[refine:%s]: Überarbeitung ungültig — Vorschlag unverändert", proposal["name"]
            )
            return proposal
        logger.info("Agent-Proposer[refine:%s]: Vorschlag überarbeitet", proposal["name"])
        # Name und Persona bleiben bei der Preregistrierung identisch,
        # auch wenn der LLM in der Antwort einen anderen Namen liefert.
        return {
            "name": proposal["name"],
            "claim": revised["claim"],
            "code": revised["code"],
            "persona": proposal["persona"],
        }

    with ThreadPoolExecutor(max_workers=max(1, len(pool))) as executor:
        futures = [executor.submit(refine_one, proposal) for proposal in pool]
        return [future.result() for future in futures]


def propose(
    client: LLMClient,
    digest: str,
    taken_names: tuple[str, ...],
    *,
    max_candidates: int = 3,
    personas: tuple[tuple[str, str], ...] = PERSONAS,
    archive_digest: str = "",
) -> list[dict[str, str]]:
    """Führt die dreirundige Persona-Diskussion aus.

    Runde 1 (parallele Persona-Vorschläge) → Cap auf ``max_candidates``
    → Runde 2 (Skeptiker-Kritik, advisory) → Runde 3 (Revision pro
    Vorschlag). Fehlerszenarien sind auf jeder Ebene **nicht fatal**:
    ausgefallene Aufrufe fallen auf die vorherige Stufe zurück
    (Kritik fehlgeschlagen → Originalvorschläge; Revision ungültig →
    Originalvorschlag; Totalausfall → leere Liste).

    ``archive_digest`` (Gate-Reflexion) wird nur an Runde 1 weiterge-
    reicht; die Rückgabe trägt pro Vorschlag das ``persona``-Feld.
    """
    client.timeout = PROPOSER_TIMEOUT
    taken = frozenset(taken_names)
    per_persona = max(1, -(-max_candidates // max(1, len(personas))))

    pool = _round_proposals(
        client, digest, taken_names, taken, per_persona, personas, archive_digest=archive_digest
    )[:max_candidates]
    if not pool:
        return []

    critiques = _round_critique(client, digest, pool)
    if not critiques:
        logger.info("Agent-Proposer: ohne Skeptiker-Kritik — Vorschläge unverändert weitergereicht")
        return pool

    return _round_refine(client, pool, critiques, taken)
