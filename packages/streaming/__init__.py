"""Streaming — Event Streaming Layer für Redpanda/Kafka.

Stellt ein einheitliches Producer-/Consumer-Interface bereit,
unabhängig vom konkreten Broker. Unterstützt Topic-Verwaltung,
Schema-Validierung, Event-Replay und Dead-Letter-Queues.
"""

from __future__ import annotations

from .base import Consumer, DeadLetterHandler, Producer, StreamConfig

__all__ = [
    "Consumer",
    "DeadLetterHandler",
    "Producer",
    "StreamConfig",
]
