"""Repository — Generic SQLAlchemy Repository-Implementierung."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ..base import PaginationParams, QueryFilter, Repository
from .models import Base

if TYPE_CHECKING:
    from typing import TypeVar

    _T = TypeVar("_T", bound=Base)
else:
    _T = Base  # type: ignore[assignment]


class SQLAlchemyRepository(Repository[_T]):
    """Generic Repository für SQLAlchemy-Entities.

    Unterstützt CRUD-Operationen und Abfragen mit Filtern.
    """

    def __init__(self, session: Session, model_type: type[_T]) -> None:
        self._session = session
        self._model_type = model_type

    async def get_by_id(self, entity_id: str) -> _T | None:
        """Holt ein Entity by ID (primärer Schlüssel)."""
        try:
            stmt = select(self._model_type).where(self._model_type.id == entity_id)  # type: ignore[attr-defined]
            result = self._session.execute(stmt).scalar_one_or_none()
            return result
        except SQLAlchemyError:
            return None

    async def create(self, entity: _T) -> _T:
        """Erstellt ein neues Entity."""
        try:
            self._session.add(entity)
            self._session.commit()
            self._session.refresh(entity)
            return entity
        except SQLAlchemyError:
            self._session.rollback()
            raise

    async def update(self, entity_id: str, data: dict[str, Any]) -> _T | None:
        """Aktualisiert ein bestehendes Entity."""
        entity = await self.get_by_id(entity_id)
        if entity is None:
            return None
        try:
            for key, value in data.items():
                if hasattr(entity, key):
                    setattr(entity, key, value)
            if hasattr(entity, "updated_at"):
                setattr(entity, "updated_at", datetime.now(UTC))
            self._session.commit()
            self._session.refresh(entity)
            return entity
        except SQLAlchemyError:
            self._session.rollback()
            return None

    async def delete(self, entity_id: str) -> bool:
        """Löscht ein Entity."""
        entity = await self.get_by_id(entity_id)
        if entity is None:
            return False
        try:
            self._session.delete(entity)
            self._session.commit()
            return True
        except SQLAlchemyError:
            self._session.rollback()
            return False

    async def list_items(
        self,
        filters: list[QueryFilter] | None = None,
        pagination: PaginationParams | None = None,
    ) -> list[_T]:
        """Listet Entities mit optionalen Filtern und Paginierung."""
        try:
            stmt = select(self._model_type)

            if filters is not None:
                for filt in filters:
                    column = getattr(self._model_type, filt.field, None)
                    if column is None:
                        continue
                    if filt.operator == "=":
                        stmt = stmt.where(column == filt.value)
                    elif filt.operator == "!=":
                        stmt = stmt.where(column != filt.value)
                    elif filt.operator == ">":
                        stmt = stmt.where(column > filt.value)
                    elif filt.operator == ">=":
                        stmt = stmt.where(column >= filt.value)
                    elif filt.operator == "<":
                        stmt = stmt.where(column < filt.value)
                    elif filt.operator == "<=":
                        stmt = stmt.where(column <= filt.value)
                    elif filt.operator == "in":
                        stmt = stmt.where(column.in_(filt.value))
                    elif filt.operator == "like":
                        stmt = stmt.where(column.like(filt.value))

            page = pagination or PaginationParams()
            if page.order_by:
                order_col = getattr(self._model_type, page.order_by, None)
                if order_col is not None:
                    order = order_col.desc() if page.order_direction == "desc" else order_col.asc()
                    stmt = stmt.order_by(order)

            if page is not None:
                stmt = stmt.offset(page.offset).limit(page.limit)

            result = self._session.execute(stmt).scalars().all()
            return list(result)
        except SQLAlchemyError:
            return []

    async def count(
        self,
        filters: list[QueryFilter] | None = None,
    ) -> int:
        """Zählt Entities mit optionalen Filtern."""
        try:
            stmt = select(func.count(self._model_type.id))  # type: ignore[attr-defined]
            if filters is not None:
                for filt in filters:
                    column = getattr(self._model_type, filt.field, None)
                    if column is None:
                        continue
                    if filt.operator == "=":
                        stmt = stmt.where(column == filt.value)
            result = self._session.execute(stmt).scalar_one()
            return result or 0
        except SQLAlchemyError:
            return 0

    async def exists(self, entity_id: str) -> bool:
        """Prüft, ob ein Entity mit der ID existiert."""
        count = await self.count(filters=[QueryFilter(field="id", value=entity_id)])
        return count > 0
