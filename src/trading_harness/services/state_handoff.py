"""Encrypted, portable runtime-state hand-on/handoff coordination."""

from __future__ import annotations

import base64
import json
import os
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from pydantic import BaseModel, Field


class HandoffError(RuntimeError):
    """Base error for portable-state operations."""


class HandoffPasswordError(HandoffError):
    """The bundle could not be authenticated with the configured password."""


class HandoffConflictError(HandoffError):
    """Another node owns a non-expired handoff lease."""


class HandoffManifest(BaseModel):
    version: int = 1
    generation: int = Field(default=0, ge=0)
    owner_node: str = ""
    lease_until: datetime | None = None
    created_at: datetime
    files: dict[str, str]


class HandoffCoordinator:
    """Encrypt state files and coordinate exclusive continuation between nodes."""

    def __init__(
        self,
        data_dir: Path | str,
        bundle_path: Path | str,
        password: str,
        node_id: str,
        *,
        lease_seconds: int = 300,
    ) -> None:
        if len(password) < 8:
            raise ValueError("state handoff password must contain at least 8 characters")
        if not node_id.strip():
            raise ValueError("state node id must not be empty")
        self.data_dir = Path(data_dir)
        self.bundle_path = Path(bundle_path)
        self._password = password.encode("utf-8")
        self.node_id = node_id
        self.lease_seconds = lease_seconds

    def hand_off(self) -> HandoffManifest:
        """Release ownership and export the current local state."""
        previous = self._read_manifest_optional()
        manifest = HandoffManifest(
            generation=(previous.generation + 1 if previous else 1),
            owner_node="",
            lease_until=None,
            created_at=datetime.now(UTC),
            files=self._collect_files(),
        )
        self._write_payload(manifest.model_dump(mode="json"))
        return manifest

    def hand_on(self) -> HandoffManifest:
        """Acquire the bundle lease and atomically restore its files."""
        manifest = self._read_manifest()
        now = datetime.now(UTC)
        if (
            manifest.owner_node
            and manifest.owner_node != self.node_id
            and manifest.lease_until is not None
            and manifest.lease_until > now
        ):
            raise HandoffConflictError(
                f"state is leased by {manifest.owner_node} until {manifest.lease_until.isoformat()}"
            )
        self._restore_files(manifest.files)
        manifest.owner_node = self.node_id
        manifest.lease_until = now + timedelta(seconds=self.lease_seconds)
        manifest.created_at = now
        self._write_payload(manifest.model_dump(mode="json"))
        return manifest

    def renew(self) -> HandoffManifest:
        """Renew an existing lease owned by this node with current local state."""
        manifest = self._read_manifest()
        if manifest.owner_node != self.node_id:
            raise HandoffConflictError(f"state lease is not owned by {self.node_id}")
        manifest.generation += 1
        manifest.created_at = datetime.now(UTC)
        manifest.lease_until = manifest.created_at + timedelta(seconds=self.lease_seconds)
        manifest.files = self._collect_files()
        self._write_payload(manifest.model_dump(mode="json"))
        return manifest

    def _collect_files(self) -> dict[str, str]:
        if not self.data_dir.exists():
            return {}
        result: dict[str, str] = {}
        for path in sorted(self.data_dir.rglob("*")):
            if path.is_file() and not path.is_symlink() and ".corrupt-" not in path.name:
                relative = path.relative_to(self.data_dir).as_posix()
                result[relative] = base64.b64encode(path.read_bytes()).decode("ascii")
        return result

    def _restore_files(self, files: dict[str, str]) -> None:
        decoded: list[tuple[Path, bytes]] = []
        root = self.data_dir.resolve()
        for relative, encoded_content in files.items():
            pure = PurePosixPath(relative)
            if pure.is_absolute() or ".." in pure.parts or not pure.parts:
                raise ValueError(f"unsafe state path: {relative}")
            target = (self.data_dir / Path(*pure.parts)).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"unsafe state path: {relative}")
            decoded.append((target, base64.b64decode(encoded_content, validate=True)))
        for target, decoded_content in decoded:
            target.parent.mkdir(parents=True, exist_ok=True)
            self._atomic_write(target, decoded_content, mode=0o600)

    def _read_manifest_optional(self) -> HandoffManifest | None:
        if not self.bundle_path.exists():
            return None
        return self._read_manifest()

    def _read_manifest(self) -> HandoffManifest:
        envelope = json.loads(self.bundle_path.read_text(encoding="utf-8"))
        try:
            salt = base64.b64decode(envelope["salt"], validate=True)
            nonce = base64.b64decode(envelope["nonce"], validate=True)
            ciphertext = base64.b64decode(envelope["ciphertext"], validate=True)
            plaintext = AESGCM(self._derive_key(salt)).decrypt(nonce, ciphertext, b"smith-v1")
        except (InvalidTag, KeyError, ValueError) as exc:
            raise HandoffPasswordError("invalid password or damaged state bundle") from exc
        return HandoffManifest.model_validate_json(plaintext)

    def _write_payload(self, payload: dict[str, Any]) -> None:
        salt = os.urandom(16)
        nonce = os.urandom(12)
        plaintext = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ciphertext = AESGCM(self._derive_key(salt)).encrypt(nonce, plaintext, b"smith-v1")
        envelope = {
            "format": "smith-state-v1",
            "kdf": "scrypt-n16384-r8-p1",
            "salt": base64.b64encode(salt).decode("ascii"),
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
        }
        self._atomic_write(
            self.bundle_path,
            (json.dumps(envelope, sort_keys=True, indent=2) + "\n").encode("utf-8"),
            mode=0o600,
        )

    def _write_payload_for_test(self, payload: dict[str, Any]) -> None:
        defaults = {
            "version": 1,
            "generation": 1,
            "owner_node": "",
            "lease_until": None,
            "created_at": datetime.now(UTC).isoformat(),
        }
        self._write_payload({**defaults, **payload})

    def _derive_key(self, salt: bytes) -> bytes:
        return Scrypt(salt=salt, length=32, n=2**14, r=8, p=1).derive(self._password)

    @staticmethod
    def _atomic_write(path: Path, content: bytes, *, mode: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
        try:
            os.fchmod(fd, mode)
            with os.fdopen(fd, "wb") as stream:
                fd = -1
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp_name, path)
        finally:
            if fd >= 0:
                os.close(fd)
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
