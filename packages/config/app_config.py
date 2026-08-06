"""Application Configuration — Zentrales Config-Objekt.

Bündelt alle Subsystem-Konfigurationen.
Lädt aus configs/trading.yaml (standard) oder ENV.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .base import ConfigLoader
from .database import DatabaseConfig
from .observability import ObservabilityConfig
from .streaming import StreamingConfig
from .trading import TradingConfig


class AppConfig(BaseModel):
    """Zentrales Anwendungskonfigurations-Objekt.

    Bündelt Database, Trading, Streaming, Observability.
    Lädt Defaults und überschreibt mit YAML/ENV.
    """

    app_name: str = Field(default="trading-orchestra", description="Anwendungsname.")
    version: str = Field(default="0.1.0", description="Semantische Version.")
    environment: str = Field(
        default="development",
        description="Umfeld: development, staging, production.",
    )

    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    @classmethod
    def from_file(cls, config_dir: str = "configs") -> AppConfig:
        """Lädt Konfiguration aus configs/ und ENV.

        Reihenfolge: Defaults → YAML → ENV.
        """
        loader = ConfigLoader(config_dir=config_dir)
        yaml_config = loader.load_yaml("trading.yaml")
        full_config = loader.apply_env_override(yaml_config)
        return cls.model_validate(full_config)

    @classmethod
    def default(cls) -> AppConfig:
        """Erstellt AppConfig mit reinen Defaults (kein Datei-Lookup)."""
        return cls()

    def is_production(self) -> bool:
        """Prüft ob Production-Modus."""
        return self.environment == "production"

    def is_live_trading(self) -> bool:
        """Prüft ob Live-Trading aktiviert."""
        return self.trading.mode == "live"
