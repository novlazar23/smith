"""Deterministic tests for env-based live execution credentials."""

from __future__ import annotations

import base64

import pytest
from packages.live_execution.credentials import (
    build_ccxt_config_from_env,
    load_key_ring_from_env,
)
from packages.live_execution.gateway import GatewayExecutionError
from packages.security.hardening.encryption import KeyRing

MASTER_KEY = base64.urlsafe_b64encode(b"\x01" * 32).decode()


def _keyring() -> KeyRing:
    ring = KeyRing()
    ring.add_key(1, MASTER_KEY)
    return ring


def test_no_env_gives_default_venue_without_config() -> None:
    venues, config, key_ring = build_ccxt_config_from_env({})
    assert venues == ["binance"]
    assert config == {}
    assert key_ring is None


def test_master_key_env_builds_keyring() -> None:
    key_ring = load_key_ring_from_env({"LIVE_MASTER_KEY_B64": MASTER_KEY})
    assert key_ring is not None
    assert key_ring.current_version == 1


def test_plaintext_env_credentials_are_collected() -> None:
    env = {
        "LIVE_VENUES": "binance, bybit",
        "LIVE_BINANCE_API_KEY": "plain-key",
        "LIVE_BINANCE_API_SECRET": "plain-secret",
        "LIVE_BYBIT_API_KEY": "bybit-key",
    }
    venues, config, key_ring = build_ccxt_config_from_env(env)
    assert venues == ["binance", "bybit"]
    assert key_ring is None
    assert config["binance"]["apiKey"] == "plain-key"
    assert config["binance"]["secret"] == "plain-secret"
    assert config["bybit"]["apiKey"] == "bybit-key"
    assert "secret" not in config["bybit"]


def test_encrypted_env_tokens_require_master_key() -> None:
    ring = _keyring()
    env = {
        "LIVE_VENUES": "binance",
        "LIVE_MASTER_KEY_B64": MASTER_KEY,
        "LIVE_BINANCE_API_KEY_TOKEN": ring.encrypt("plain-key"),
        "LIVE_BINANCE_API_SECRET_TOKEN": ring.encrypt("plain-secret"),
    }
    venues, config, key_ring = build_ccxt_config_from_env(env)
    assert venues == ["binance"]
    assert key_ring is not None
    assert "apiKeyToken" in config["binance"]
    assert "secretToken" in config["binance"]
    assert "apiKey" not in config["binance"]
    assert "secret" not in config["binance"]


def test_encrypted_env_tokens_without_master_key_fail_closed() -> None:
    env = {
        "LIVE_VENUES": "binance",
        "LIVE_BINANCE_API_KEY_TOKEN": "token",
    }
    with pytest.raises(GatewayExecutionError, match="LIVE_MASTER_KEY_B64"):
        build_ccxt_config_from_env(env)
