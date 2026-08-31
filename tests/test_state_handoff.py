from __future__ import annotations

import json
from pathlib import Path

import pytest

from trading_harness.services.state_handoff import (
    HandoffConflictError,
    HandoffCoordinator,
    HandoffPasswordError,
)


def test_export_import_round_trip_is_encrypted(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "shadow.json").write_text('{"secret":"value"}', encoding="utf-8")
    bundle = tmp_path / "handoff" / "state.enc.json"

    source = HandoffCoordinator(data, bundle, "correct horse battery staple", "node-a")
    manifest = source.hand_off()

    raw = bundle.read_text(encoding="utf-8")
    assert "value" not in raw
    assert manifest.owner_node == ""

    (data / "shadow.json").unlink()
    target = HandoffCoordinator(data, bundle, "correct horse battery staple", "node-b")
    imported = target.hand_on()

    assert imported.owner_node == "node-b"
    assert json.loads((data / "shadow.json").read_text(encoding="utf-8")) == {
        "secret": "value"
    }


def test_wrong_password_fails_without_overwriting_local_state(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "state.json").write_text("original", encoding="utf-8")
    bundle = tmp_path / "state.enc.json"
    HandoffCoordinator(data, bundle, "right-password", "node-a").hand_off()
    (data / "state.json").write_text("local", encoding="utf-8")

    with pytest.raises(HandoffPasswordError):
        HandoffCoordinator(data, bundle, "wrong-password", "node-b").hand_on()

    assert (data / "state.json").read_text(encoding="utf-8") == "local"


def test_active_owner_blocks_second_node(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    bundle = tmp_path / "state.enc.json"
    first = HandoffCoordinator(data, bundle, "password", "node-a", lease_seconds=300)
    first.hand_off()
    first.hand_on()

    with pytest.raises(HandoffConflictError):
        HandoffCoordinator(data, bundle, "password", "node-b", lease_seconds=300).hand_on()


def test_path_traversal_in_bundle_is_rejected(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    bundle = tmp_path / "state.enc.json"
    coordinator = HandoffCoordinator(data, bundle, "password", "node-a")
    coordinator._write_payload_for_test({"files": {"../escape": "bad"}})

    with pytest.raises(ValueError, match="unsafe state path"):
        coordinator.hand_on()
