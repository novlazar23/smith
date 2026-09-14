"""Tests für das Hot-Reload der Champion-Parametersätze (champion_configs.json)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from apps.orchestrator_service import service as service_module
from apps.orchestrator_service.service import OrchestratorServiceConfig, build_service

from .conftest import FakeDB, StubProvider


def _write_configs(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "champion_configs.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _bump(path: Path) -> None:
    # Festes, vom echten Mtime deutlich verschiedenes Zeitstempel-Paar
    os.utime(path, ns=(1_000_000_000_000_000_000, 1_000_000_001_000_000_000))


def _config(tmp_path: Path, configs_path: Path | None) -> OrchestratorServiceConfig:
    return OrchestratorServiceConfig(
        instruments=("BTC/USDT",),
        heartbeat_path=tmp_path / "heartbeat",
        champion_configs_path=configs_path,
    )


class TestChampionConfigReload:
    """Hot-Reload: Parametersätze werden bei Artefakt-Änderung pro Zyklus neu geladen."""

    def test_first_cycle_loads_existing_artifact(
        self, tmp_path: Path, fake_db: FakeDB, stub_provider: StubProvider
    ) -> None:
        path = _write_configs(tmp_path, {"trend": {"version": 1, "params": {"ema_fast": 7}, "score": 0.5}})
        service = build_service(
            config=_config(tmp_path, path), provider=stub_provider, db=fake_db
        )
        assert service._champion_params is None  # Start: noch nicht geladen
        service.run_cycle()  # Artefakt existiert bereits beim Start → erster Zyklus lädt
        assert service._champion_params == {"trend": {"ema_fast": 7}}

    def test_reloads_when_artifact_changes(
        self, tmp_path: Path, fake_db: FakeDB, stub_provider: StubProvider
    ) -> None:
        path = _write_configs(tmp_path, {"trend": {"version": 1, "params": {"ema_fast": 7}, "score": 0.5}})
        service = build_service(
            config=_config(tmp_path, path), provider=stub_provider, db=fake_db
        )
        service.run_cycle()
        assert service._champion_params == {"trend": {"ema_fast": 7}}

        # Artefakt-Update (Promotion): neuer Parametersatz
        _write_configs(tmp_path, {"trend": {"version": 2, "params": {"ema_fast": 9}, "score": 0.6}})
        _bump(path)
        service.run_cycle()
        assert service._champion_params == {"trend": {"ema_fast": 9}}

    def test_unchanged_artifact_keeps_params(
        self, tmp_path: Path, fake_db: FakeDB, stub_provider: StubProvider
    ) -> None:
        path = _write_configs(tmp_path, {"trend": {"version": 1, "params": {"ema_fast": 7}, "score": 0.5}})
        service = build_service(
            config=_config(tmp_path, path), provider=stub_provider, db=fake_db
        )
        service.run_cycle()
        service.run_cycle()  # unverändertes Artefakt → kein Reload
        assert service._champion_params == {"trend": {"ema_fast": 7}}

    def test_keeps_last_good_value_on_load_error(
        self, tmp_path: Path, fake_db: FakeDB, stub_provider: StubProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = _write_configs(tmp_path, {"trend": {"version": 1, "params": {"ema_fast": 7}, "score": 0.5}})
        service = build_service(
            config=_config(tmp_path, path), provider=stub_provider, db=fake_db
        )
        service.run_cycle()
        assert service._champion_params == {"trend": {"ema_fast": 7}}

        def _boom(p: Path) -> dict[str, dict[str, float | int]]:
            raise OSError("Artefakt nicht lesbar")

        monkeypatch.setattr(service_module, "load_champion_params", _boom)
        _write_configs(tmp_path, {"trend": {"version": 2, "params": {"ema_fast": 9}, "score": 0.6}})
        _bump(path)
        service.run_cycle()  # Reload-Fehler → bisherige Parametersätze behalten
        assert service._champion_params == {"trend": {"ema_fast": 7}}

    def test_picks_up_artifact_after_first_evolve_run(
        self, tmp_path: Path, fake_db: FakeDB, stub_provider: StubProvider
    ) -> None:
        service = build_service(
            config=_config(tmp_path, tmp_path / "champion_configs.json"),
            provider=stub_provider,
            db=fake_db,
        )
        assert service._champion_params is None  # Start ohne Artefakt (fail-soft)
        _write_configs(tmp_path, {"trend": {"version": 1, "params": {"ema_fast": 7}, "score": 0.5}})
        service.run_cycle()  # erst nach dem ersten Evolutions-Lauf vorhanden
        assert service._champion_params == {"trend": {"ema_fast": 7}}

    def test_missing_artifact_keeps_no_params(
        self, tmp_path: Path, fake_db: FakeDB, stub_provider: StubProvider
    ) -> None:
        service = build_service(
            config=_config(tmp_path, tmp_path / "fehlt.json"), provider=stub_provider, db=fake_db
        )
        service.run_cycle()
        assert service._champion_params is None

    def test_without_path_has_no_params(
        self, tmp_path: Path, fake_db: FakeDB, stub_provider: StubProvider
    ) -> None:
        service = build_service(
            config=_config(tmp_path, None), provider=stub_provider, db=fake_db
        )
        service.run_cycle()
        assert service._champion_params is None
