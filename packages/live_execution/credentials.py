"""Env-basierte Live-Execution-Config für CCXT und KeyRing.

Die Funktion ist bewusst schlank: Sie bildet Live-Venues und Credentials
aus Umgebungsvariablen auf. Verschlüsselte Tokens werden hier nicht
aufgelöst, sondern als ``apiKeyToken`` / ``secretToken`` an das Gateway
gegeben, das sie über den übergebenen :class:`KeyRing` entschlüsselt.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from packages.live_execution.gateway import GatewayExecutionError
from packages.security.hardening.encryption import KeyRing

__all__ = [
    "MASTER_KEY_ENV",
    "VENUES_ENV",
    "build_ccxt_config_from_env",
    "load_key_ring_from_env",
]

MASTER_KEY_ENV = "LIVE_MASTER_KEY_B64"
VENUES_ENV = "LIVE_VENUES"


def _split_venues(raw: str) -> list[str]:
    return [venue.strip().lower() for venue in raw.split(",") if venue.strip()]


def _env_value(env: Mapping[str, str], key: str) -> str:
    return str(env.get(key, "")).strip()


def load_key_ring_from_env(env: Mapping[str, str] | None = None) -> KeyRing | None:
    """Lädt den Master Key aus ``LIVE_MASTER_KEY_B64``.

    ponytail: Der Env-Key wird als Version 1 geladen; versionierte Env-Keys
    kommen, wenn Production-Key-Rotation über Env gesteuert werden muss.
    """
    source = os.environ if env is None else env
    raw = _env_value(source, MASTER_KEY_ENV)
    if not raw:
        return None
    ring = KeyRing()
    ring.add_key(1, raw)
    return ring


def build_ccxt_config_from_env(
    env: Mapping[str, str] | None = None,
) -> tuple[list[str], dict[str, dict[str, Any]], KeyRing | None]:
    """Buildet Venues, CCXT-Config und optional KeyRing aus Env.

    Unterstützte Variablen pro Venue (``{VENUE}`` = Venue in Großbuchstaben):
    - ``LIVE_{VENUE}_API_KEY`` / ``LIVE_{VENUE}_API_SECRET``
    - ``LIVE_{VENUE}_API_KEY_TOKEN`` / ``LIVE_{VENUE}_API_SECRET_TOKEN``
    """
    source = os.environ if env is None else env
    venues = _split_venues(_env_value(source, VENUES_ENV) or "binance")
    key_ring = load_key_ring_from_env(source)

    config: dict[str, dict[str, Any]] = {}
    for venue in venues:
        suffix = venue.upper().replace("-", "_")
        cfg: dict[str, Any] = {"enableRateLimit": True}

        key_token = _env_value(source, f"LIVE_{suffix}_API_KEY_TOKEN")
        secret_token = _env_value(source, f"LIVE_{suffix}_API_SECRET_TOKEN")
        api_key = _env_value(source, f"LIVE_{suffix}_API_KEY")
        api_secret = _env_value(source, f"LIVE_{suffix}_API_SECRET")

        if key_token or secret_token:
            if key_ring is None:
                raise GatewayExecutionError(
                    "Verschlüsselte Live-Credentials gesetzt, "
                    f"aber {MASTER_KEY_ENV} fehlt."
                )
            if key_token:
                cfg["apiKeyToken"] = key_token
            if secret_token:
                cfg["secretToken"] = secret_token
        else:
            if api_key:
                cfg["apiKey"] = api_key
            if api_secret:
                cfg["secret"] = api_secret

        if "apiKeyToken" in cfg or "secretToken" in cfg or "apiKey" in cfg or "secret" in cfg:
            config[venue] = cfg

    return venues, config, key_ring
