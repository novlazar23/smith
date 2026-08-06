"""Persistence — Data Persistence Layer für PostgreSQL, ClickHouse und Redis.

Stellt ein Repository-Pattern zur Verfügung, das Domänen-Entitäten
persistiert und abfragt. PostgreSQL für relationale Daten (Graph-States,
Entscheidungen), ClickHouse für Zeitreihen (Candles, Trades).
"""

from __future__ import annotations

from .base import Repository, StorageBackend
from .sqlalchemy.repository import SQLAlchemyRepository

__all__ = [
    "Repository",
    "SQLAlchemyRepository",
    "StorageBackend",
]
