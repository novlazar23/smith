"""Tests für den db-init One-shot-Bootstrap (apps/db_init.py).

Alle drei Backends (PostgreSQL, ClickHouse, Redpanda) werden monkeypatched,
damit der Bootstrap ohne laufende Container testbar ist.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest
from apps import db_init
from packages.persistence.clickhouse.engine import ClickHouseConfig
from packages.persistence.sqlalchemy.engine import DatabaseConfig

_ENV_VARS = (
    "DB_HOST",
    "DB_PORT",
    "DB_NAME",
    "DB_USER",
    "DB_PASSWORD",
    "CH_HOST",
    "CH_PORT",
    "CH_DB",
    "CH_PASSWORD",
    "REDPANDA_SERVERS",
)


@pytest.fixture
def init_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patcht alle Backends von ``apps.db_init`` und sammelt alle Aufrufe.

    Return dict keys:
      - ``pg_configs`` / ``ch_configs``: Config-Objekte, die an die Engines gehen
      - ``pg_engines`` / ``ch_engines``: Fake-Engine-Instanzen
      - ``create_all_calls``: Bindings, an die ``Base.metadata.create_all`` ging
      - ``list_topics_calls``: Anzahl der ``list_topics``-Aufrufe
      - ``admins`` / ``new_topics``: Erzeugte Redpanda-Objekte
      - ``pg_fail`` / ``ch_fail`` / ``rp_fail``: Flags, um einen Schritt scheitern zu lassen
      - ``admin_class``: Fake-AdminClient-Klasse (``topics`` pro Test setzbar)
    """
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    captured: dict[str, Any] = {
        "pg_configs": [],
        "ch_configs": [],
        "pg_engines": [],
        "ch_engines": [],
        "create_all_calls": [],
        "list_topics_calls": [],
        "admins": [],
        "new_topics": [],
        "pg_fail": False,
        "ch_fail": False,
        "rp_fail": False,
    }

    class FakePGEngine:
        def __init__(self, config: DatabaseConfig) -> None:
            self.config = config
            self.engine = MagicMock(name="sa_engine")

    def fake_sa_engine(config: DatabaseConfig) -> FakePGEngine:
        captured["pg_configs"].append(config)
        engine = FakePGEngine(config)
        captured["pg_engines"].append(engine)
        return engine

    class _Meta:
        @staticmethod
        def create_all(bind: Any) -> None:
            if captured["pg_fail"]:
                raise RuntimeError("pg create_all failed")
            captured["create_all_calls"].append(bind)

    class FakeBase:
        metadata = _Meta

    def fake_inspect(bind: Any) -> MagicMock:
        inspector = MagicMock(name="inspector")
        inspector.get_table_names.return_value = ["trading_graph_states", "final_decisions"]
        return inspector

    class FakeCHEngine:
        def __init__(self, config: ClickHouseConfig) -> None:
            self.config = config

        def create_tables(self) -> None:
            if captured["ch_fail"]:
                raise RuntimeError("ch create_tables failed")
            captured["ch_engines"].append(self)

    def fake_ch_factory(config: ClickHouseConfig) -> FakeCHEngine:
        captured["ch_configs"].append(config)
        return FakeCHEngine(config)

    class FakeNewTopic:
        def __init__(
            self, topic: str, num_partitions: int = 1, replication_factor: int = 1, **kwargs: Any
        ) -> None:
            self.topic = topic
            self.num_partitions = num_partitions
            self.replication_factor = replication_factor
            captured["new_topics"].append(self)

    class FakeAdminClient:
        topics: ClassVar[set[str]] = {"market_data"}

        def __init__(self, conf: dict[str, str]) -> None:
            self.conf = conf
            self.create_topics_calls: list[list[Any]] = []
            captured["admins"].append(self)

        def list_topics(self, *args: Any, **kwargs: Any) -> set[str]:
            if captured["rp_fail"]:
                raise RuntimeError("rp list_topics failed")
            captured["list_topics_calls"].append(self)
            return self.__class__.topics

        def create_topics(self, new_topics: list[Any], *args: Any, **kwargs: Any) -> None:
            self.create_topics_calls.append(new_topics)

    monkeypatch.setattr(db_init, "SQLAlchemyEngine", fake_sa_engine)
    monkeypatch.setattr(db_init, "Base", FakeBase)
    monkeypatch.setattr(db_init, "inspect", fake_inspect)
    monkeypatch.setattr(db_init, "create_ch_engine", fake_ch_factory)
    monkeypatch.setattr(db_init, "AdminClient", FakeAdminClient)
    monkeypatch.setattr(db_init, "NewTopic", FakeNewTopic)
    captured["admin_class"] = FakeAdminClient
    return captured


def _run_main() -> int:
    """Führt ``main()`` aus und gibt den Exit-Code zurück."""
    with pytest.raises(SystemExit) as excinfo:
        db_init.main()
    return excinfo.value.code


class TestEnvDefaults:
    """Compose-Defaults ohne gesetzte Env-Vars."""

    def test_pg_config_uses_compose_defaults(self, init_env: dict[str, Any]) -> None:
        assert _run_main() == 0
        config = init_env["pg_configs"][0]
        assert config.host == "postgres"
        assert config.port == 5432
        assert config.database == "trading"
        assert config.user == "orchestra"
        assert config.password == ""

    def test_ch_config_uses_compose_defaults(self, init_env: dict[str, Any]) -> None:
        assert _run_main() == 0
        config = init_env["ch_configs"][0]
        assert config.host == "clickhouse"
        assert config.port == 8123
        assert config.database == "trading_events"
        assert config.password == ""

    def test_explicit_env_overrides_defaults(
        self, monkeypatch: pytest.MonkeyPatch, init_env: dict[str, Any]
    ) -> None:
        monkeypatch.setenv("DB_HOST", "pg-custom")
        monkeypatch.setenv("DB_PORT", "6432")
        monkeypatch.setenv("DB_NAME", "custom_db")
        monkeypatch.setenv("DB_USER", "custom_user")
        monkeypatch.setenv("DB_PASSWORD", "pg-secret")
        monkeypatch.setenv("CH_HOST", "ch-custom")
        monkeypatch.setenv("CH_PORT", "9123")
        monkeypatch.setenv("CH_DB", "custom_events")
        monkeypatch.setenv("CH_PASSWORD", "ch-secret")
        monkeypatch.setenv("REDPANDA_SERVERS", "rp-custom:9093")

        assert _run_main() == 0

        pg = init_env["pg_configs"][0]
        assert (pg.host, pg.port, pg.database, pg.user, pg.password) == (
            "pg-custom",
            6432,
            "custom_db",
            "custom_user",
            "pg-secret",
        )
        ch = init_env["ch_configs"][0]
        assert (ch.host, ch.port, ch.database, ch.password) == (
            "ch-custom",
            9123,
            "custom_events",
            "ch-secret",
        )
        assert init_env["admins"][0].conf == {"bootstrap.servers": "rp-custom:9093"}


class TestBackendContact:
    """Alle drei Backends werden bei einem erfolgreichen Lauf kontaktiert."""

    def test_pg_create_all_called_with_engine(self, init_env: dict[str, Any]) -> None:
        assert _run_main() == 0
        assert len(init_env["create_all_calls"]) == 1
        assert init_env["create_all_calls"][0] is init_env["pg_engines"][0].engine

    def test_pg_table_count_logged(self, init_env: dict[str, Any], caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="apps.db_init"):
            assert _run_main() == 0
        table_records = [record for record in caplog.records if "postgres" in record.message]
        assert table_records
        assert table_records[0].__dict__["count"] == 2

    def test_ch_create_tables_called(self, init_env: dict[str, Any]) -> None:
        assert _run_main() == 0
        assert len(init_env["ch_engines"]) == 1
        assert init_env["ch_engines"][0].config is init_env["ch_configs"][0]

    def test_admin_list_topics_called_with_default_servers(self, init_env: dict[str, Any]) -> None:
        assert _run_main() == 0
        assert len(init_env["list_topics_calls"]) == 1
        assert init_env["admins"][0].conf == {"bootstrap.servers": "redpanda:9092"}

    def test_success_logs_db_init_complete(self, init_env: dict[str, Any], caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="apps.db_init"):
            assert _run_main() == 0
        assert "db-init complete" in caplog.text


class TestRedpandaTopic:
    """market_data-Topic wird nur angelegt, wenn es fehlt."""

    def test_existing_topic_not_recreated(self, init_env: dict[str, Any]) -> None:
        init_env["admin_class"].topics = {"market_data"}
        assert _run_main() == 0
        assert init_env["admins"][0].create_topics_calls == []
        assert init_env["new_topics"] == []

    def test_missing_topic_created_with_expected_shape(self, init_env: dict[str, Any]) -> None:
        init_env["admin_class"].topics = {"some_other_topic"}
        assert _run_main() == 0

        admin = init_env["admins"][0]
        assert len(admin.create_topics_calls) == 1
        topics = admin.create_topics_calls[0]
        assert len(topics) == 1

        new_topic = topics[0]
        assert new_topic.topic == "market_data"
        assert new_topic.num_partitions == 3
        assert new_topic.replication_factor == 1


class TestFailureExit:
    """Jeder fehlschlagende Schritt beendet mit Exit-Code 1."""

    FAIL_KEYS: ClassVar[dict[str, str]] = {"postgres": "pg_fail", "clickhouse": "ch_fail", "redpanda": "rp_fail"}

    @pytest.mark.parametrize("step", ["postgres", "clickhouse", "redpanda"])
    def test_step_failure_exits_nonzero(self, init_env: dict[str, Any], step: str) -> None:
        init_env[self.FAIL_KEYS[step]] = True
        with pytest.raises(SystemExit) as excinfo:
            db_init.main()
        assert excinfo.value.code == 1
