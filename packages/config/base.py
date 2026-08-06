"""Base Config Loader - YAML/JSON + ENV Override.

Ladt Konfiguration aus Datei und uberschreibt mit ENV-Variablen.
Unterstitzt Pydantic v2 Settings mit TOML-Datei-Loading.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml


class ConfigLoader:
    """Ladt Konfiguration aus Dateien mit ENV-Override.

    Lade-reihenfolge (Prioritat steigt):
    1. Datei (YAML → JSON)
    2. ENV-Variablen (prefix_XXX)

    Beispiel: TRADING_RISK_MAX_DRAWDOWN=0.05 uberschreibt den YAML-Wert.
    """

    def __init__(
        self,
        config_dir: str | Path | None = None,
        env_prefix: str = "APP",
    ) -> None:
        self.config_dir = Path(config_dir) if config_dir else Path("configs")
        self.env_prefix = env_prefix

    def load_yaml(self, filename: str) -> dict[str, Any]:
        """Ladt eine YAML-Datei."""
        filepath = self.config_dir / filename
        if not filepath.exists():
            return {}
        with filepath.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}

    def load_json(self, filename: str) -> dict[str, Any]:
        """Ladt eine JSON-Datei."""
        filepath = self.config_dir / filename
        if not filepath.exists():
            return {}
        with filepath.open(encoding="utf-8") as f:
            return json.load(f)

    def load(self, *filenames: str) -> dict[str, Any]:
        """Ladt mehrere Dateien (YAML priorisiert).

        Gibt vereinigte Konfiguration zurück.
        """
        result: dict[str, Any] = {}
        for fname in filenames:
            if fname.endswith((".yaml", ".yml")):
                result.update(self.load_yaml(fname))
            elif fname.endswith(".json"):
                result.update(self.load_json(fname))
        return result

    def apply_env_override(
        self, config: dict[str, Any]
    ) -> dict[str, Any]:
        """Uberschreibt Konfiguration mit ENV-Variablen.

        ENV-Variablen mussen dem Schema {env_prefix}_{section}_{key} entsprechen.
        Beispiel: APP_DATABASE_HOST=localhost
        """
        result = config.copy()
        prefix = f"{self.env_prefix}_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                env_key = key[len(prefix):]
                self._set_nested(result, env_key, value)
        return result

    @staticmethod
    def _set_nested(data: dict[str, Any], dotted_key: str, value: str) -> None:
        """Setzt einen verschachtelten Dict-Eintrag über dotted key."""
        parts = dotted_key.lower().split("_")
        current = data
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = ConfigLoader._parse_value(value)

    @staticmethod
    def _parse_value(value: str) -> str | int | float | bool | None:
        """Parse string to appropriate type."""
        if value.lower() in ("true", "yes"):
            return True
        if value.lower() in ("false", "no"):
            return False
        if value.lower() in ("none", "null"):
            return None
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value
