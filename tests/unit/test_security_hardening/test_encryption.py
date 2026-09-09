"""Deterministic tests for AES-256-GCM KeyRing encryption (EPIC-16 WP06)."""

from __future__ import annotations

import base64

import pytest
from packages.security.hardening.encryption import (
    DecryptionError,
    EncryptionUnavailableError,
    KeyRing,
    decode_master_key,
)

# Fixed 32-byte keys, encoded as URL-safe base64.
KEY_V1 = base64.urlsafe_b64encode(b"\x01" * 32).decode()
KEY_V2 = base64.urlsafe_b64encode(b"\x02" * 32).decode()


def test_roundtrip() -> None:
    ring = KeyRing()
    ring.add_key(1, KEY_V1)
    token = ring.encrypt("my-secret-api-key")
    assert ring.decrypt(token) == "my-secret-api-key"


def test_token_is_base64_and_reversible() -> None:
    ring = KeyRing()
    ring.add_key(1, KEY_V1)
    token = ring.encrypt("abc")
    raw = base64.urlsafe_b64decode(token)
    # version (1 byte) + nonce (12) + plaintext (3) + tag (16)
    assert len(raw) == 32


def test_different_versions_give_different_tokens() -> None:
    ring = KeyRing()
    ring.add_key(1, KEY_V1)
    ring.add_key(2, KEY_V2)
    t1 = ring.encrypt("abc")
    ring.set_current(2)
    t2 = ring.encrypt("abc")
    assert t1 != t2


def test_decrypt_wrong_version_fails() -> None:
    ring = KeyRing()
    ring.add_key(1, KEY_V1)
    t1 = ring.encrypt("abc")
    ring.add_key(2, KEY_V2)
    ring.set_current(2)
    # Token from v1 can no longer be decrypted once v1 is deprecated.
    ring.deprecate(1)
    with pytest.raises(DecryptionError):
        ring.decrypt(t1)
    # But the v2 token still works.
    assert ring.decrypt(ring.encrypt("xyz")) == "xyz"


def test_corrupted_token_fails() -> None:
    ring = KeyRing()
    ring.add_key(1, KEY_V1)
    token = ring.encrypt("abc")
    # Corrupt a char in the ciphertext region (not the first, which may
    # coincidentally be unchanged).
    mid = len(token) // 2
    orig = token[mid]
    replacement = "A" if orig != "A" else "B"
    corrupted = token[:mid] + replacement + token[mid + 1 :]
    assert corrupted != token
    with pytest.raises(DecryptionError):
        ring.decrypt(corrupted)


def test_non_b64_token_fails() -> None:
    ring = KeyRing()
    ring.add_key(1, KEY_V1)
    with pytest.raises(DecryptionError):
        ring.decrypt("not-a-token!!!")


def test_add_key_rejects_wrong_length() -> None:
    bad = base64.urlsafe_b64encode(b"\x01" * 31).decode()
    with pytest.raises(ValueError):
        decode_master_key(bad)


def test_add_key_rejects_bad_base64() -> None:
    with pytest.raises(ValueError):
        decode_master_key("###not-base64###")


def test_add_key_rejects_duplicate_version() -> None:
    ring = KeyRing()
    ring.add_key(1, KEY_V1)
    with pytest.raises(ValueError):
        ring.add_key(1, KEY_V2)


def test_cryptography_unavailable_raises() -> None:
    ring = KeyRing()
    ring.add_key(1, KEY_V1)
    # Simulate absence of the cryptography package by poisoning __import__.
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("cryptography"):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = fake_import
    try:
        with pytest.raises(EncryptionUnavailableError):
            ring.encrypt("abc")
    finally:
        builtins.__import__ = real_import
