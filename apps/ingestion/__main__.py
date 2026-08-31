"""Startet den Market-Data-Consumer als langlebigen Service.

Das Service (Compose-Command ``python -m apps.ingestion``) verbraucht das
``MARKET_DATA_TOPIC`` von Redpanda und persistiert Candle-Events in
ClickHouse. Ein Daemon-Thread pflegt periodisch die Heartbeat-Datei
``/tmp/heartbeat`` für den Compose-Healthcheck.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

from apps.ingestion.consumer import DataIngestionService
from apps.ingestion.persistence import ClickHouseMarketDataSink

logger = logging.getLogger(__name__)

HEARTBEAT_PATH = Path("/tmp/heartbeat")
HEARTBEAT_INTERVAL_SECONDS = 30.0


def _heartbeat_loop() -> None:
    """Schreibt periodisch die Heartbeat-Datei für den Healthcheck."""
    while True:
        try:
            HEARTBEAT_PATH.touch()
        except OSError as exc:
            logger.warning("Heartbeat konnte nicht geschrieben werden: %s", exc)
        time.sleep(HEARTBEAT_INTERVAL_SECONDS)


def main() -> None:
    """Startet Heartbeat, ClickHouse-Sink und den Consumer-Loop."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    heartbeat = threading.Thread(target=_heartbeat_loop, name="heartbeat", daemon=True)
    heartbeat.start()

    sink = ClickHouseMarketDataSink()
    sink.ensure_schema()
    logger.info("ClickHouse-Schema ist bereit.")

    service = DataIngestionService(
        topic=os.environ.get("MARKET_DATA_TOPIC", "market_data"),
        bootstrap_servers=os.environ.get("REDPANDA_SERVERS", "redpanda:9092").split(","),
    )
    logger.info(
        "Starte Consumer: topic=%s servers=%s",
        service.topic,
        service.bootstrap_servers,
    )
    service.consume(sink.persist)


if __name__ == "__main__":
    main()
