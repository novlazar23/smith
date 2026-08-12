"""Status-Klassifikation für News-Events.

Bereitgestellt:
    - NewsStatus (RUMOR, INITIAL, CONFIRMATION, UPDATE, CORRECTION, RETRACTION)
    - classify_news(item, history) -> NewsStatus — Status basierend auf History und Keywords
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class NewsStatus(StrEnum):
    """News-Event-Status (Spec §13 News-Agent).

    RUMOR: Unbestätigte Gerüchte
    INITIAL: Erste Bestätigung einer Nachricht
    CONFIRMATION: Bestätigung von Informationen
    UPDATE: Aktualisierung bestehender Information
    CORRECTION: Korrektur früherer Information
    RETRACTION: Rückzug von Information
    """

    RUMOR = "RUMOR"
    INITIAL = "INITIAL"
    CONFIRMATION = "CONFIRMATION"
    UPDATE = "UPDATE"
    CORRECTION = "CORRECTION"
    RETRACTION = "RETRACTION"


# Keyword-Mappings für Status-Klassifikation
_STATUS_KEYWORDS: dict[NewsStatus, list[str]] = {
    NewsStatus.RUMOR: [
        "rumor", "rumoured", "rumored", "reportedly", "allegedly",
        "unconfirmed", "speculation", "speculative", "gossip",
        "hoax", "fake", "fraud", "scam", "hack", "hacked", "leak",
    ],
    NewsStatus.INITIAL: [
        "announces", "announced", "launches", "launched", "introduces",
        "introduced", "starts", "starting", "begins", "begun", "opening",
        "initial", "first", "debut", "premiere", "new listing",
        "unveils", "unveiled", "reveals", "reveal",
    ],
    NewsStatus.CONFIRMATION: [
        "confirms", "confirmed", "verifies", "verified", "validated",
        "acknowledges", "acknowledged", "corroborates", "substantiates",
        "proves", "proof", "certifies", "certified", "authenticated",
        "authentication", "endorsement", "endorse", "backing",
    ],
    NewsStatus.UPDATE: [
        "update", "updates", "updated", "progress", "progressing",
        "developing", "developed", "evolving", "expands", "expanded",
        "extension", "extension of", "phase two", "v2", "version 2",
        "new phase", "next phase", "follow-up", "follow up",
        "further", "additional", "more information",
    ],
    NewsStatus.CORRECTION: [
        "correction", "corrects", "corrected", "error", "errors",
        "mistake", "mistakes", "clarification", "clarify",
        "discrepancy", "discrepancies", "inaccuracy", "inaccuracies",
        "misleading", "misled", "amends", "amendment",
    ],
    NewsStatus.RETRACTION: [
        "retracts", "retracted", "withdrawal", "withdrawn",
        "revokes", "revoked", "cancel", "canceled", "cancelled",
        "discontinues", "discontinued", "shuts down", "shutdown",
        "delisted", "delisting", "ceases", "ceased", "终止",
    ],
}


@dataclass(frozen=True)
class NewsMatch:
    """Ergebnis einer Klassifikations-Übereinstimmung.

    Felder:
        status: Der ermittelte Status.
        score: Treffer-Stärke (0-1).
        matching_keywords: Liste der gefundenen Keywords.
    """

    status: NewsStatus
    score: float
    matching_keywords: list[str] = field(default_factory=list)


def _match_keywords(text: str, keywords: list[str]) -> list[str]:
    """Sucht Keywords (case-insensitive) in einem Text.

    Args:
        text: Zu durchsuchender Text.
        keywords: Liste der zu suchenden Keywords.

    Returns:
        Liste der gefundenen Keywords.
    """
    if not text:
        return []
    text_lower = text.lower()
    return [kw for kw in keywords if kw.lower() in text_lower]


def _calculate_status_score(
    text: str, keywords: list[str]
) -> tuple[NewsStatus, float, list[str]]:
    """Berechnet den besten Status-Score für einen Text.

    Args:
        text: Zu analysierender Text (title + body).
        keywords: Keywords für einen bestimmten Status.

    Returns:
        Tuple aus Status, Score und gefundenen Keywords.
    """
    found = _match_keywords(text, keywords)
    if not found:
        return NewsStatus.INITIAL, 0.0, []

    score = min(len(found) / max(len(keywords), 1), 1.0)
    return NewsStatus(keywords[0] if keywords else "INITIAL"), score, found


def classify_news(
    item: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
) -> NewsStatus:
    """Klassifiziert ein News-Event basierend auf Keywords und History.

    Kombiniert Keyword-Matching mit historischem Vergleich:
        - Gleiche Quelle + ähnlicher Titel → UPDATE/CONFIRMATION
        - Negative Keywords → RUMOR
        - Positive Keywords → INITIAL/CONFIRMATION

    Args:
        item: News-Event mit title, body, source_name, event_identity.
        history: Liste früherer News-Events für historischen Vergleich.

    Returns:
        Klassifizierter NewsStatus.
    """
    title = item.get("title", "")
    body = item.get("body", "") or ""
    text = f"{title} {body}".strip().lower()

    best_match: NewsMatch = NewsMatch(
        status=NewsStatus.INITIAL, score=0.0, matching_keywords=[]
    )

    # Keyword-Matching durch alle Status
    for status, keywords in _STATUS_KEYWORDS.items():
        if not keywords:
            continue
        found = _match_keywords(text, keywords)
        if found:
            score = min(len(found) / max(len(keywords), 1) * 2, 1.0)
            if score > best_match.score:
                best_match = NewsMatch(
                    status=status,
                    score=score,
                    matching_keywords=found,
                )

    # Historischer Vergleich: Gleiche Quelle + ähnlicher Titel
    if history:
        historical_match = _compare_history(item, history)
        if historical_match and historical_match.score > best_match.score * 0.5:
            # Historischer Match hat höhere Priorität
            best_match = historical_match

    return best_match.status


def _compare_history(
    item: dict[str, Any], history: list[dict[str, Any]]
) -> NewsMatch | None:
    """Vergleicht ein News-Event mit der History.

    Gleiche Quelle + ähnlicher Titel bedeutet UPDATE oder CONFIRMATION.

    Args:
        item: Das zu vergleichende News-Event.
        history: Liste früherer Events.

    Returns:
        Best-match Result oder None.
    """
    title = item.get("title", "")
    source = item.get("source_name", "")

    for past in history:
        past_source = past.get("source_name", "")
        past_title = past.get("title", "")
        past_identity = past.get("event_identity", "")
        current_identity = item.get("event_identity", "")

        # Gleiche Quelle und ähnliche event_identity
        if source and source == past_source:
            similarity = _title_similarity(title, past_title)
            if similarity >= 0.6:
                past_status = past.get("status", "INITIAL")
                if past_status == "CONFIRMATION":
                    return NewsMatch(
                        status=NewsStatus.UPDATE,
                        score=similarity,
                        matching_keywords=["historical_match"],
                    )
                elif past_status in ("INITIAL", "UPDATE"):
                    return NewsMatch(
                        status=NewsStatus.CONFIRMATION,
                        score=similarity,
                        matching_keywords=["historical_match"],
                    )

        # Gleiche event_identity → identisch
        if past_identity and current_identity == past_identity:
            return NewsMatch(
                status=NewsStatus.UPDATE,
                score=1.0,
                matching_keywords=["same_identity"],
            )

    return None


def _title_similarity(title_a: str, title_b: str) -> float:
    """Berechnet die Jaccard-Ähnlichkeit zweier Titel."""
    if not title_a or not title_b:
        return 0.0
    words_a = set(title_a.lower().split())
    words_b = set(title_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union) if union else 0.0
