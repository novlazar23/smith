"""AES-256-GCM Encryption at Rest für API-Keys (EPIC-16 WP06).

Master Keys sind URL-safe base64 encodierte, exakt 32 Byte lange Strings.
Tokens tragen die Key-Version als erstes Byte und enthalten Version +
Nonce + Ciphertext + GCM-Tag in einem stabilen base64url-Format.

Wichtig: Keys, Tokens und Plaintext-Secrets werden nirgends geloggt.
"""

from __future__ import annotations

import base64
import os
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

__all__ = [
    "MASTER_KEY_LENGTH",
    "DecryptionError",
    "EncryptionUnavailableError",
    "KeyRing",
]

MASTER_KEY_LENGTH = 32
_NONCE_LENGTH = 12
_GCM_TAG_LENGTH = 16
_MAX_VERSION = 255  # Token-Format hält die Version in einem Byte


class EncryptionUnavailableError(RuntimeError):
    """Das ``cryptography``-Paket ist nicht verfügbar.

    Es gibt absichtlich keinen Fallback auf schwächere Kryptografie.
    """


class DecryptionError(ValueError):
    """Token ist malformed, Version unbekannt/deprecated oder Auth fehlgeschlagen."""


def _load_aesgcm() -> type[AESGCM]:
    """Lädt AESGCM lazy; wirft EncryptionUnavailableError statt zu fallen zurück."""
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:
        raise EncryptionUnavailableError(
            "Das 'cryptography'-Paket ist nicht installiert — "
            "AES-256-GCM kann nicht verwendet werden. Es gibt keinen "
            "Fallback auf schwächere Kryptografie."
        ) from exc
    return AESGCM


def decode_master_key(key_b64: str) -> bytes:
    """Decodiert einen URL-safe base64 Master Key und validiert die Länge."""
    try:
        raw = base64.urlsafe_b64decode(key_b64.encode("ascii"))
    except (ValueError, UnicodeEncodeError) as exc:
        raise ValueError("Master Key muss URL-safe base64 encoded sein") from exc
    if len(raw) != MASTER_KEY_LENGTH:
        raise ValueError(
            f"Master Key muss exakt {MASTER_KEY_LENGTH} Byte lang sein "
            f"(ist {len(raw)})"
        )
    return raw


class KeyRing:
    """Versionierte AES-256-Master-Keys mit Rotation-Support.

    - ``current_version`` — mit dieser Version wird verschlüsselt
    - deprecated Versionen können nicht mehr entschlüsselt werden
    - alte Versionen bleiben bis zur expliziten Deprecation verwendbar
    """

    def __init__(self, nonce_factory: Callable[[], bytes] | None = None) -> None:
        self._keys: dict[int, bytes] = {}
        self._current_version: int | None = None
        self._deprecated: set[int] = set()
        # ponytail: os.urandom-Default; injectable Factory für deterministische Tests
        self._nonce_factory = nonce_factory or (lambda: os.urandom(_NONCE_LENGTH))

    @property
    def current_version(self) -> int | None:
        return self._current_version

    @property
    def deprecated_versions(self) -> frozenset[int]:
        return frozenset(self._deprecated)

    def has_version(self, version: int) -> bool:
        return version in self._keys

    def add_key(self, version: int, key_b64: str) -> None:
        """Fügt eine neue Key-Version hinzu (die erste wird current)."""
        if version < 1 or version > _MAX_VERSION:
            raise ValueError(f"Key-Version muss zwischen 1 und {_MAX_VERSION} sein")
        raw = decode_master_key(key_b64)
        if version in self._keys:
            raise ValueError(f"Key-Version {version} existiert bereits")
        self._keys[version] = raw
        if self._current_version is None:
            self._current_version = version

    def set_current(self, version: int) -> None:
        """Setzt die Version für neue Verschlüsselungen; deprecated Versionen sind ausgeschlossen."""
        if version not in self._keys:
            raise ValueError(f"Unbekannte Key-Version {version}")
        if version in self._deprecated:
            raise ValueError(f"Key-Version {version} ist deprecated")
        self._current_version = version

    def deprecate(self, version: int) -> None:
        """Depreciert eine Version — bestehende Daten damit sind nicht mehr lesbar."""
        if version not in self._keys:
            raise ValueError(f"Unbekannte Key-Version {version}")
        if version == self._current_version:
            raise ValueError("Die aktuelle Key-Version kann nicht deprecated werden")
        self._deprecated.add(version)

    def encrypt(self, plaintext: str) -> str:
        """Verschlüsselt mit der aktuellen Key-Version (AES-256-GCM)."""
        aesgcm = _load_aesgcm()
        version = self._current_version
        if version is None:
            raise RuntimeError("KeyRing hat keine aktuelle Key-Version")
        nonce = self._nonce_factory()
        if len(nonce) != _NONCE_LENGTH:
            raise ValueError(f"Nonce-Factory muss {_NONCE_LENGTH} Byte liefern")
        key = self._keys[version]
        ciphertext = aesgcm(key).encrypt(nonce, plaintext.encode("utf-8"), None)
        blob = bytes((version,)) + nonce + ciphertext
        return base64.urlsafe_b64encode(blob).decode("ascii")

    def decrypt(self, token: str) -> str:
        """Entschlüsselt ein Token; schlägt fehl bei unbekannter/deprecated Version oder Auth-Fehler."""
        aesgcm = _load_aesgcm()
        try:
            raw = base64.urlsafe_b64decode(token.encode("ascii"))
        except (ValueError, UnicodeEncodeError) as exc:
            raise DecryptionError("Token ist kein gültiges URL-safe base64") from exc
        min_length = 1 + _NONCE_LENGTH + _GCM_TAG_LENGTH
        if len(raw) < min_length:
            raise DecryptionError("Token ist zu kurz")
        version = raw[0]
        nonce = raw[1 : 1 + _NONCE_LENGTH]
        ciphertext = raw[1 + _NONCE_LENGTH :]
        key = self._keys.get(version)
        if key is None:
            raise DecryptionError(f"Unbekannte Key-Version {version}")
        if version in self._deprecated:
            raise DecryptionError(f"Key-Version {version} ist deprecated")
        try:
            plaintext = aesgcm(key).decrypt(nonce, ciphertext, None)
        except Exception as exc:
            raise DecryptionError("Ciphertext-Authentifizierung fehlgeschlagen") from exc
        return plaintext.decode("utf-8")
