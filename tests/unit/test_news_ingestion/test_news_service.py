"""Tests für den News-Ingestion-Service (Persistenz + Endlosschleife).

Abdeckung:
  - persist_events: Bulk-Insert, ON CONFLICT DO NOTHING, doppelte news_id
  - run_forever: once-Modus (persist + last_run_times), Fehler-Resilienz
  - Model-Registrierung: news_events in Base.metadata
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from apps.news_ingestion import service
from apps.news_ingestion.config import NewsConfig, SourceConfig
from packages.persistence.sqlalchemy.models import Base
from sqlalchemy.dialects import postgresql


def make_event(news_id: str) -> dict[str, Any]:
    """Erzeugt ein Fake-News-Event mit allen Feldern aus run_ingestion_cycle."""
    now = datetime.now(UTC)
    return {
        "news_id": news_id,
        "event_identity": "0" * 32,
        "title": f"Title {news_id}",
        "body": "Body text",
        "source_name": "TestSource",
        "source_type": "RSS",
        "url_hash": "a" * 64,
        "published_at": now,
        "received_at": now,
        "entities": ["BTC"],
        "instruments": ["BTC"],
        "language": "en",
        "revision": 1,
        "status": "INITIAL",
    }


def fake_session_factory() -> tuple[MagicMock, MagicMock]:
    """Erzeugt eine Mock-Session-Factory (Context-Manager-kompatibel)."""
    session = MagicMock(name="session")
    session.__enter__.return_value = session
    factory = MagicMock(name="factory", return_value=session)
    return factory, session


def _one_source_config() -> NewsConfig:
    """NewsConfig mit einer einzigen Test-Quelle."""
    return NewsConfig(
        sources=[
            SourceConfig(name="TestSource", url="https://test.com/rss", update_interval=60)
        ]
    )


# ═══════════════════════════════════════════════
# persist_events Tests
# ═══════════════════════════════════════════════


class TestPersistEvents:
    """Tests für persist_events."""

    def test_inserts_all_events(self) -> None:
        events = [make_event("a" * 16), make_event("b" * 16)]
        factory, session = fake_session_factory()
        count = service.persist_events(events, factory)
        assert count == 2
        factory.assert_called_once()
        session.execute.assert_called_once()
        session.commit.assert_called_once()

    def test_on_conflict_do_nothing_on_news_id(self) -> None:
        events = [make_event("a" * 16)]
        factory, session = fake_session_factory()
        service.persist_events(events, factory)
        stmt = session.execute.call_args.args[0]
        sql = str(stmt.compile(dialect=postgresql.dialect()))
        assert "ON CONFLICT" in sql
        assert "news_id" in sql
        assert "DO NOTHING" in sql

    def test_duplicate_news_id_no_error(self) -> None:
        events = [make_event("c" * 16), make_event("c" * 16)]
        factory, session = fake_session_factory()
        count = service.persist_events(events, factory)
        assert count == 2
        session.execute.assert_called_once()
        session.commit.assert_called_once()

    def test_empty_events_skips_insert(self) -> None:
        factory, session = fake_session_factory()
        count = service.persist_events([], factory)
        assert count == 0
        session.execute.assert_not_called()
        session.commit.assert_not_called()


# ═══════════════════════════════════════════════
# run_forever Tests
# ═══════════════════════════════════════════════


class TestRunForever:
    """Tests für run_forever."""

    def test_once_mode_persists_and_updates_last_run_times(self) -> None:
        events = [make_event("d" * 16), make_event("e" * 16)]

        def fake_cycle(
            config: NewsConfig,
            last_run_times: dict[str, datetime] | None = None,
            history: list[dict[str, Any]] | None = None,
        ) -> list[dict[str, Any]]:
            if last_run_times is not None:
                last_run_times["TestSource"] = datetime.now(UTC)
            return events

        with (
            patch.object(service, "run_ingestion_cycle", side_effect=fake_cycle) as mock_cycle,
            patch.object(service, "session_factory") as mock_factory,
            patch.object(service, "persist_events", return_value=2) as mock_persist,
            patch.object(service, "time") as mock_time,
        ):
            last_run_times = service.run_forever(config=_one_source_config(), once=True)

        mock_cycle.assert_called_once()
        mock_factory.assert_called_once()
        mock_persist.assert_called_once()
        persisted, _factory = mock_persist.call_args.args
        assert len(persisted) == 2
        assert persisted == events
        # last_run_times wird an den Zyklus übergeben und nach Rückgabe aktualisiert
        passed_lru = mock_cycle.call_args.args[1]
        assert passed_lru is last_run_times
        assert "TestSource" in last_run_times
        mock_time.sleep.assert_not_called()

    def test_cycle_exception_does_not_stop_loop(self) -> None:
        sentinel = RuntimeError("stop-test")
        sleep_calls = {"n": 0}

        def boom_sleep(_seconds: float) -> None:
            # Erster Sleep ok, zweiter beendet die Schleife → Zyklus 2 wird gefahren
            sleep_calls["n"] += 1
            if sleep_calls["n"] >= 2:
                raise sentinel

        with (
            patch.object(
                service, "run_ingestion_cycle", side_effect=ConnectionError("db down")
            ) as mock_cycle,
            patch.object(service, "session_factory"),
            patch.object(service, "persist_events"),
            patch.object(service, "time") as mock_time,
        ):
            mock_time.sleep.side_effect = boom_sleep
            with pytest.raises(RuntimeError, match="stop-test"):
                service.run_forever(config=_one_source_config(), tick_seconds=1.0)

        # Zyklus-Fehler werden gefangen und die Schleife läuft weiter
        assert mock_cycle.call_count == 2


# ═══════════════════════════════════════════════
# Model-Registrierung Tests
# ═══════════════════════════════════════════════


class TestNewsEventModelRegistration:
    """Tests für die Registrierung des News-Models in Base.metadata."""

    def test_news_events_in_metadata(self) -> None:
        import packages.persistence.sqlalchemy.news  # noqa: F401

        assert "news_events" in Base.metadata.tables
