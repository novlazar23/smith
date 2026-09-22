"""Dauerhaftes Kandidaten-Archiv für die Evolved Agents (Stufe 2).

Eine JSONL-Zeile pro geprüftem Kandidaten-Lauf in
``evolved_agents_archive.jsonl`` (neben ``evolved_agents.json``,
Shared-Volume ``backtest_reports``). Das Archiv ist die dauerhafte
Stage-2-Kandidaten-Memory über die nächtlichen Läufe hinweg und dient
zwei Zwecken:

- **Retesting-Schutz:** Kandidaten mit identischem Code-Hash (SHA-256)
  werden nicht erneut evaluiert — tote Enden (Jail-abgelehnte oder
  gate-schwache Logiken) tauchen nicht in jeder Nacht erneut im
  Replay auf. Das Muster spiegelt das graveyard von ``apps/evolution``
  (append-only JSONL, Dedup gegen tote Varianten), hier als eigenständige
  Datei ohne Import aus der anderen Pipeline.
- **Gate-Reflexion:** die jüngsten Ablehnungen (Near-Misses: Score
  knapp unter der Hurdle) werden als kompaktes Digest in den
  Persona-Prompt des Proposers eingespeist, damit der LLM Near-Misses
  gezielt repariert statt neu zu würfeln.

**Fail-Soft-Vertrag:** Lesen bricht nie — fehlende Datei → leere
Liste, korrupte Zeilen werden übersprungen (mit ``logger.warning``).
Schreiben ist **append-only** (eine Zeile pro Kandidat, kein Rewrite),
konsistent mit ``sequential_test.append_jsonl``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .sequential_test import append_jsonl

logger = logging.getLogger(__name__)

#: Archiv-Datei neben ``evolved_agents.json`` (append-only JSONL).
ARCHIVE_FILENAME = "evolved_agents_archive.jsonl"
#: Anzahl der jüngsten Ablehnungen, die dem Proposal-Prompt (Gate-
#: Reflexion) und der Dedup-Limit-Logik übergeben werden.
DEFAULT_LIMIT = 15


def code_hash(code: str) -> str:
    """SHA-256-Hash des Kandidaten-Codes (Retesting-Erkennung).

    Identischer Code → identischer Hash, unabhängig von Name/Claim —
    derselbe Vorschlag taucht also unter neuem Namen trotzdem im
    Archiv auf und wird nicht zweimal evaluiert.
    """
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def record_candidates(path: str | Path, entries: Sequence[Mapping[str, Any]]) -> Path:
    """Hängt ``entries`` als JSONL-Zeilen an das Archiv (append-only).

    Leere ``entries`` ändern nichts (Datei wird nicht angelegt oder
    berührt). Der Pfad wird zurückgegeben (konsistent mit
    ``append_jsonl``).
    """
    out = Path(path)
    for entry in entries:
        append_jsonl(out, entry)
    return out


def load_archive(path: str | Path) -> list[dict[str, Any]]:
    """Lädt alle Archiv-Einträge in Datei-Reihenfolge (fail-soft).

    Fehlende Datei → ``[]``. Korrupte Zeilen oder nicht-Objekt-Zeilen
    werden übersprungen (``logger.warning``) — ein defektes Archiv darf
    den Lauf nicht abbrechen (Fail-Soft, konsistent mit
    ``trial_ledger.load_trial_count``).
    """
    file = Path(path)
    if not file.exists():
        return []
    try:
        text = file.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Kandidaten-Archiv %s nicht lesbar (%s) — leer", file, exc)
        return []
    entries: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry: Any = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning("Korrupte Zeile in %s übersprungen (%s)", file.name, exc)
            continue
        if isinstance(entry, dict):
            entries.append(entry)
        else:
            logger.warning("Nicht-Objekt-Zeile in %s übersprungen", file.name)
    return entries


def known_code_hashes(path: str | Path) -> set[str]:
    """Alle Code-Hashes im Archiv (Retesting-Schutz: ``code_hash``-Dedup)."""
    return {str(entry["code_hash"]) for entry in load_archive(path) if entry.get("code_hash")}


def recent_rejections(path: str | Path, limit: int = DEFAULT_LIMIT) -> list[dict[str, Any]]:
    """Die jüngsten abgelehnten Kandidaten (``admitted == False``).

    Datei-Reihenfolge ist chronologisch (append-only) → neueste zuerst.
    Dedup nach ``(name, code_hash)``, wobei das NEUESTE Vorkommnis
    gewinnt (z. B. ein Kandidat, der in Lauf 1 abgelehnt und in Lauf 2
    erneut geprüft wurde → nur der Lauf-2-Eintrag zählt). Auf
    ``limit`` gekappt.
    """
    rejected = [entry for entry in load_archive(path) if entry.get("admitted") is False]
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for entry in reversed(rejected):
        key = (str(entry.get("name", "")), str(entry.get("code_hash", "")))
        if key in seen:
            continue
        seen.add(key)
        out.append(entry)
        if len(out) >= limit:
            break
    return out


def format_archive_digest(entries: Sequence[Mapping[str, Any]], limit: int = DEFAULT_LIMIT) -> str:
    """Kompaktes Deutsch-Text-Block für den Persona-Prompt (Gate-Reflexion).

    Eine Zeile pro Eintrag, neueste zuerst (``entries`` soll bereits in
    dieser Reihenfolge vorliegen, s. ``recent_rejections``):

        - {name} ({persona}, {run_at-Datum}): Score {score:.4f}, Gate:
          {reasons per "; "}, Code {code_hash[:12]}

    ``score`` None → ``Score n/a``; leere ``reasons`` →
    ``Gate: (ohne Metriken)``; ``persona`` None (handgeschriebener
    Kandidat) → ``handgeschrieben``; leere ``entries`` → ``""``.
    """
    lines: list[str] = []
    for entry in list(entries)[:limit]:
        name = str(entry.get("name", "?"))
        persona = entry.get("persona")
        persona_text = str(persona) if persona else "handgeschrieben"
        date = str(entry.get("run_at", ""))[:10] or "?"
        score = entry.get("score")
        score_text = "n/a" if score is None else f"{float(score):.4f}"
        reasons = [str(reason) for reason in (entry.get("reasons") or [])]
        gate_text = "; ".join(reasons) if reasons else "(ohne Metriken)"
        lines.append(
            f"- {name} ({persona_text}, {date}): Score {score_text}, "
            f"Gate: {gate_text}, Code {str(entry.get('code_hash', ''))[:12]}"
        )
    return "\n".join(lines)
