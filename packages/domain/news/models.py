"""Domain models for the News analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class NewsStatus(StrEnum):
    """News-Status für die Lebenszyklus-Erkennung."""

    RUMOR = "rumor"
    INITIAL = "initial"
    CONFIRMATION = "confirmation"
    UPDATE = "update"
    CORRECTION = "correction"
    RETRACTION = "retraction"


@dataclass(frozen=True, slots=True)
class NewsEvent:
    """Ein normalisierter News-Event mit stabiler Identitaet."""

    id: str
    title: str
    body: str
    source_name: str
    source_type: str
    url_hash: str
    published_at: datetime
    received_at: datetime
    entities: list[str] = field(default_factory=list)
    instruments: list[str] = field(default_factory=list)
    language: str = "en"
    revision: int = 1
    status: NewsStatus = NewsStatus.INITIAL


@dataclass(frozen=True, slots=True)
class EntityMatch:
    """Ein gematchter Entity-Eintrag mit Konfidenz."""

    entity: str
    confidence: float
    type: str


@dataclass(frozen=True, slots=True)
class NewsCluster:
    """Cluster von aehnlichen News-Events."""

    cluster_id: str
    event_ids: list[str] = field(default_factory=list)
    representative: str = ""
