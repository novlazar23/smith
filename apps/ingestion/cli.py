from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from apps.ingestion.consumer import REDPANDA_AVAILABLE, DataIngestionService

logger = logging.getLogger(__name__)


def create_ingestion_service(topic: str, bootstrap_servers: list[str] | None) -> DataIngestionService:
    """Creates a new ingestion service with the given configuration."""
    return DataIngestionService(
        topic=topic,
        bootstrap_servers=bootstrap_servers,
    )


def start_consumer(args: argparse.Namespace) -> None:
    """Starts consuming from Redpanda and prints events to stdout."""
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s: %(message)s")

    service = create_ingestion_service(
        topic=args.topic,
        bootstrap_servers=args.servers,
    )

    def handler(event: dict[str, Any]) -> None:
        print(json.dumps(event, default=str))
        sys.stdout.flush()

    if not REDPANDA_AVAILABLE:
        logger.error("Redpanda client is not installed. Install it to consume live data.")
        sys.exit(1)

    service.consume(handler, max_messages=args.max_messages)


def process_file(args: argparse.Namespace) -> None:
    """Processes market data from a JSON file."""
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s: %(message)s")

    service = create_ingestion_service(
        topic=args.topic,
        bootstrap_servers=args.servers,
    )

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    with input_path.open(encoding="utf-8") as fh:
        raw_events: list[dict[str, Any]] = json.load(fh)

    results = service.process_batch(raw_events)
    for event in results:
        print(json.dumps(event, default=str))

    logger.info("Processed %d / %d events successfully.", len(results), len(raw_events))


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Trading Orchestra — Data Ingestion CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # consumer
    consumer_parser = sub.add_parser("consume", help="Consume from Redpanda topic")
    consumer_parser.add_argument("--topic", default="market_data", help="Redpanda topic name")
    consumer_parser.add_argument("--servers", nargs="*", default=["localhost:9092"], help="Bootstrap servers")
    consumer_parser.add_argument("--max-messages", type=int, default=None, help="Max messages to consume")
    consumer_parser.set_defaults(func=start_consumer)

    # process-file
    file_parser = sub.add_parser("process-file", help="Process events from a JSON file")
    file_parser.add_argument("input", help="Path to JSON file")
    file_parser.add_argument("--topic", default="market_data", help="Redpanda topic name")
    file_parser.add_argument("--servers", nargs="*", default=["localhost:9092"], help="Bootstrap servers")
    file_parser.set_defaults(func=process_file)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
