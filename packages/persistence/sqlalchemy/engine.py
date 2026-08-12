"""Engine — SQLAlchemy Engine und Session Management.

Verwaltet PostgreSQL-Verbindungen, Sessions und Metadata.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from sqlalchemy import create_engine as sa_create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


@dataclass(frozen=True)
class DatabaseConfig:
    """PostgreSQL-Verbindungskonfiguration."""

    host: str = "localhost"
    port: int = 5432
    database: str = "trading_orchestra"
    user: str = "trading_user"
    password: str = os.environ.get("DB_PASSWORD", "trading_password")
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10

    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class SQLAlchemyEngine:
    """SQLAlchemy-Engine mit Session-Management."""

    def __init__(self, config: DatabaseConfig | None = None) -> None:
        self._config = config or DatabaseConfig()
        self._engine: Engine | None = None
        self._session_factory: sessionmaker[Session] | None = None

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            self._engine = sa_create_engine(
                self._config.url,
                echo=self._config.echo,
                pool_size=self._config.pool_size,
                max_overflow=self._config.max_overflow,
            )
        return self._engine

    @property
    def session_factory(self) -> sessionmaker[Session]:
        if self._session_factory is None:
            self._session_factory = sessionmaker(bind=self.engine)
        return self._session_factory

    def get_session(self) -> Session:
        """Erstellt eine neue Datenbank-Session."""
        return self.session_factory()

    def is_connected(self) -> bool:
        """Prüft, ob die Datenbankverbindung aktiv ist."""
        if self._engine is None:
            return False
        try:
            with self.engine.connect() as conn:
                return conn is not None
        except Exception:
            return False

    async def health_check(self) -> dict[str, Any]:
        """Health-Check der PostgreSQL-Verbindung."""
        result: dict[str, Any] = {
            "backend": "postgresql",
            "connected": self.is_connected(),
            "config": {
                "host": self._config.host,
                "port": self._config.port,
                "database": self._config.database,
            },
        }
        if self.is_connected():
            result["message"] = "healthy"
        else:
            result["message"] = "not connected"
        return result


# Globale Instanz für einfache Nutzung
_default_engine: SQLAlchemyEngine | None = None


def get_engine(config: DatabaseConfig | None = None) -> SQLAlchemyEngine:
    """Holt die globale Engine-Instanz (Singleton)."""
    global _default_engine
    if _default_engine is None:
        _default_engine = SQLAlchemyEngine(config)
    return _default_engine


def get_session(config: DatabaseConfig | None = None) -> Session:
    """Holt eine neue Session über die globale Engine."""
    return get_engine(config).get_session()


def create_engine(config: DatabaseConfig | None = None) -> SQLAlchemyEngine:
    """Erstellt eine neue Engine-Instanz (factory)."""
    return SQLAlchemyEngine(config)
