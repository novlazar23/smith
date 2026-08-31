"""db-init — One-shot-Bootstrap für PostgreSQL, ClickHouse und Redpanda.

Wird als Compose-Service (``db-init``) vor dem Start der App-Services ausgeführt
und stellt sicher, dass alle Persistence-Strukturen existieren:

  1. PostgreSQL:  alle SQLAlchemy-Tabellen (``Base.metadata.create_all``)
  2. ClickHouse:  Datenbank + alle Tabellen (``ClickHouseEngine.create_tables()``)
  3. Redpanda:    Topic ``market_data`` (nur falls nicht vorhanden)

Aufruf: ``python -m apps.db_init``
Beendet sich mit Exit-Code 0 bei Erfolg, 1 bei jedem Fehler.
"""

from __future__ import annotations

import logging
import os
import sys

import packages.persistence.sqlalchemy.models
from confluent_kafka.admin import AdminClient, NewTopic
from packages.persistence.clickhouse.engine import ClickHouseConfig, create_ch_engine
from packages.persistence.sqlalchemy.engine import DatabaseConfig, SQLAlchemyEngine
from sqlalchemy import inspect

try:  # noqa: SIM105
    # News-Models landen in einem späteren Task — bis dahin bewusst überspringen.
    import packages.persistence.sqlalchemy.news  # noqa: F401
except ImportError:
    pass

from packages.persistence.sqlalchemy.models import Base

logger = logging.getLogger(__name__)

TOPIC_NAME = "market_data"
NUM_PARTITIONS = 3
REPLICATION_FACTOR = 1


def _pg_config() -> DatabaseConfig:
    """PostgreSQL-Config aus Env-Vars mit Compose-Defaults."""
    return DatabaseConfig(
        host=os.environ.get("DB_HOST", "postgres"),
        port=int(os.environ.get("DB_PORT", "5432")),
        database=os.environ.get("DB_NAME", "trading"),
        user=os.environ.get("DB_USER", "orchestra"),
        password=os.environ.get("DB_PASSWORD", ""),
    )


def _ch_config() -> ClickHouseConfig:
    """ClickHouse-Config aus Env-Vars mit Compose-Defaults."""
    return ClickHouseConfig(
        host=os.environ.get("CH_HOST", "clickhouse"),
        port=int(os.environ.get("CH_PORT", "8123")),
        database=os.environ.get("CH_DB", "trading_events"),
        password=os.environ.get("CH_PASSWORD", ""),
    )


def init_postgres() -> None:
    """Erstellt alle SQLAlchemy-Tabellen in PostgreSQL."""
    engine = SQLAlchemyEngine(_pg_config())
    Base.metadata.create_all(engine.engine)
    tables = inspect(engine.engine).get_table_names()
    logger.info("postgres tables ready", extra={"count": len(tables), "tables": sorted(tables)})


def init_clickhouse() -> None:
    """Erstellt ClickHouse-Datenbank und -Tabellen."""
    ch_engine = create_ch_engine(_ch_config())
    ch_engine.create_tables()
    logger.info("clickhouse tables ready")


def init_redpanda() -> None:
    """Stellt sicher, dass das Redpanda-Topic ``market_data`` existiert."""
    admin = AdminClient({"bootstrap.servers": os.environ.get("REDPANDA_SERVERS", "redpanda:9092")})
    topics = admin.list_topics()
    if TOPIC_NAME in topics:
        logger.info("redpanda topic exists", extra={"topic": TOPIC_NAME})
        return
    new_topic = NewTopic(TOPIC_NAME, num_partitions=NUM_PARTITIONS, replication_factor=REPLICATION_FACTOR)
    admin.create_topics([new_topic])
    logger.info("redpanda topic created", extra={"topic": TOPIC_NAME})


def main() -> None:
    """Führt die Bootstrap-Sequence aus und beendet mit Exit-Code 0/1."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        init_postgres()
        init_clickhouse()
        init_redpanda()
    except Exception as exc:
        logger.error("db-init failed", exc_info=exc)
        sys.exit(1)
    logger.info("db-init complete")
    sys.exit(0)


if __name__ == "__main__":
    main()
