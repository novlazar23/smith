"""Execution — venue-agnostic execution client.

Provides ``VenueExecutionClient`` which creates and manages exchange-
specific execution clients via ``create_client(venue)``.
Supports multiple venues configured in ``AnalysisRequest``.
"""

from __future__ import annotations

from .venue_client import VenueExecutionClient

__all__ = [
    "VenueExecutionClient",
]
