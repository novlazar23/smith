"""Database Configuration - PostgreSQL, ClickHouse, Redis.

Pydantic v2 Settings-Klassen für alle Datenbanken mit ENV-Defaults.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PostgreSQLConfig(BaseModel):
    """PostgreSQL-Verbindungskonfiguration."""

    host: str = Field(default="localhost", description="DB-Host.")
    port: int = Field(default=5432, ge=1, le=65535, description="DB-Port.")
    database: str = Field(default="trading", description="Datenbankname.")
    user: str = Field(default="trading", description="Benutzername.")
    password: str = Field(default="", description="Passwort.")
    pool_size: int = Field(default=10, ge=1, description="Verbindungspool-Größe.")
    max_overflow: int = Field(default=5, ge=0, description="Max. Overflow-Verbindungen.")
    echo: bool = Field(default=False, description="SQL-Echo aktivieren.")

    @property
    def url(self) -> str:
        """PostgreSQL-Connection-URL."""
        if self.password:
            return (
                f"postgresql+asyncpg://{self.user}:{self.password}"
                f"@{self.host}:{self.port}/{self.database}"
            )
        return (
            f"postgresql+asyncpg://{self.user}"
            f"@{self.host}:{self.port}/{self.database}"
        )


class ClickHouseConfig(BaseModel):
    """ClickHouse-Verbindungskonfiguration."""

    host: str = Field(default="localhost", description="CH-Host.")
    port: int = Field(default=8123, ge=1, le=65535, description="CH-HTTP-Port.")
    database: str = Field(default="trading", description="Datenbankname.")
    user: str = Field(default="default", description="Benutzername.")
    password: str = Field(default="", description="Passwort.")
    secure: bool = Field(default=False, description="HTTPS/SSL verwenden.")
    compress: bool = Field(default=True, description="Kompression aktivieren.")

    @property
    def url(self) -> str:
        """ClickHouse-Connection-URL."""
        scheme = "https" if self.secure else "http"
        return f"{scheme}://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class RedisConfig(BaseModel):
    """Redis-Verbindungskonfiguration."""

    host: str = Field(default="localhost", description="Redis-Host.")
    port: int = Field(default=6379, ge=1, le=65535, description="Redis-Port.")
    db: int = Field(default=0, ge=0, description="Redis-DB-Index.")
    password: str = Field(default="", description="Passwort.")
    decode_responses: bool = Field(default=True, description="Strings zurückgeben.")
    max_connections: int = Field(default=20, ge=1, description="Max. Verbindungen.")

    @property
    def url(self) -> str:
        """Redis-Connection-URL."""
        if self.password:
            return f"redis://{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class DatabaseConfig(BaseModel):
    """Gesamte Datenbank-Konfiguration."""

    postgresql: PostgreSQLConfig = Field(default_factory=PostgreSQLConfig)
    clickhouse: ClickHouseConfig = Field(default_factory=ClickHouseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
