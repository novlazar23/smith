"""Contrarian Agent — Adversarial Review Agent.

Erzeugt eine Gegenhypothesen zum Mehrheitsvotum (LONG<->SHORT),
wird VOR dem Konsens ausgefuehrt und hat Status SHADOW.
"""

from __future__ import annotations

from .agent import ContrarianAgent
from .models import ContrarianConfig, ContrarianHypothesis

__all__ = [
    "ContrarianAgent",
    "ContrarianConfig",
    "ContrarianHypothesis",
]
