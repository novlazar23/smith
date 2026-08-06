"""Base — Abstrakte Repository- und Storage-Protocols."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class StorageBackend(StrEnum):
    """Unterstützte Storage-Backends."""

    POSTGRESQL = "postgresql"
    CLICKHOUSE = "clickhouse"
    REDIS = "redis"


@dataclass(frozen=True)
class QueryFilter:
    """Einfacher Filter für Repository-Abfragen."""

    field: str
    operator: str = "="
    value: Any = None

    def __post_init__(self) -> None:
        valid_ops = ("=", "!=", ">", ">=", "<", "<=", "in", "like")
        if self.operator not in valid_ops:
            raise ValueError(f"Unsupported operator: {self.operator}")


@dataclass(frozen=True)
class PaginationParams:
    """Paginierungsparameter."""

    offset: int = 0
    limit: int = 100
    order_by: str = "created_at"
    order_direction: str = "desc"


class Repository(ABC, Generic[T]):
    """Abstrakte Repository für CRUD-Operationen."""

    @abstractmethod
    async def get_by_id(self, entity_id: str) -> T | None:
        """Holt ein Entity by ID."""

    @abstractmethod
    async def create(self, entity: T) -> T:
        """Erstellt ein neues Entity."""

    @abstractmethod
    async def update(self, entity_id: str, data: dict[str, Any]) -> T | None:
        """Aktualisiert ein bestehendes Entity."""

    @abstractmethod
    async def delete(self, entity_id: str) -> bool:
        """Löscht ein Entity."""

    @abstractmethod
    async def list_items(
        self,
        filters: list[QueryFilter] | None = None,
        pagination: PaginationParams | None = None,
    ) -> list[T]:
        """Listet Entities mit optionalen Filtern."""


class StorageEngine(ABC):
    """Abstrakter Storage-Engine für Backend-spezifische Operations."""

    @abstractmethod
    def is_connected(self) -> bool:
        """Prüft, ob die Verbindung aktiv ist."""

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Health-Check des Backends."""

    @abstractmethod
    def create_repository(self, entity_type: type[T]) -> Repository[T]:
        """Erstellt ein Repository für den Entity-Type."""


class ConnectionConfig(ABC):
    """Abstrakte Konfiguration für Storage-Verbindungen."""

    @property
    @abstractmethod
    def backend(self) -> StorageBackend:
        """Das verwendete Storage-Backend."""

    @abstractmethod
    def get_connection_string(self) -> str:
        """Verbindungszeichenkette für das Backend."""
