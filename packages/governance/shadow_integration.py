"""Shadow-Integration — operatives Brier/Kalibrierungs-Scoring von Shadow-Entscheidungen.

EPIC-16-P1: Der Orchestrator persistiert pro Shadow-Entscheidung
``probabilities`` (JSON: up/down/range) und den Referenz-Close
(``base_close``). Nach Ablauf des konfigurierten Horizons werden fällige
Entscheidungen gegen den aktuellen Close abgerechnet:

  - ``actual_direction`` (up/down/range) via ``direction_from_return``
  - ``brier_score`` via ``ShadowBrierScore`` (packages/shadow/metrics.py)
  - ``calibration_correct`` (Argmax der Prediction == tatsächliche Richtung)
  - ``scored_at``

Alle Funktionen sind rein und unit-testbar; die einzigen Seiteneffekte von
``score_due_shadow_decisions`` sind die SQL-Aufrufe auf der injizierten
Connection.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from packages.shadow.metrics import ShadowBrierScore
from sqlalchemy import text
from sqlalchemy.engine import Connection

PROBABILITY_KEYS: tuple[str, ...] = ("up", "down", "range")

_HORIZON_UNIT_SECONDS: dict[str, float] = {
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
    "d": 86400.0,
}

SELECT_DUE_SHADOW_DECISIONS = text(
    """\
    SELECT run_id, instrument, created_at, probabilities, base_close, confidence
    FROM shadow_decisions
    WHERE brier_score IS NULL
      AND probabilities IS NOT NULL
      AND base_close IS NOT NULL
      AND created_at <= :cutoff
    ORDER BY created_at ASC
    LIMIT :limit
    """
)

UPDATE_SCORED_SHADOW_DECISION = text(
    """\
    UPDATE shadow_decisions
    SET brier_score = :brier_score,
        actual_direction = :actual_direction,
        calibration_correct = :calibration_correct,
        scored_at = :scored_at
    WHERE run_id = :run_id
    """
)


class _CandleWindowLike(Protocol):
    """Duck-Typ eines Kerzenfensters mit Close-Array."""

    @property
    def close(self) -> NDArray[np.float64]:
        ...


class _CandleProviderLike(Protocol):
    """Duck-Typ eines Kerzen-Providers (z. B. ClickHouseCandleProvider)."""

    def fetch_candles(self, instrument: str, limit: int) -> _CandleWindowLike | None:
        """Liefert die letzten ``limit`` Kerzen oder None bei Fehlschlag."""
        ...


def normalize_probabilities(value: object) -> dict[str, float] | None:
    """Normalisiert ein Wahrscheinlichkeits-Dict auf genau up/down/range (Summe 1.0).

    Akzeptiert Keys in jeder Schreibweise (lower/upper/mixed case) und ignoriert
    fremde Keys. Liefert None, wenn einer der drei Schlüssel fehlt oder ein
    Wert ungültig ist (nicht numerisch, negativ, unendlich) oder die Summe
    null ist.
    """
    if not isinstance(value, dict):
        return None
    raw: dict[str, object] = {}
    for key, val in value.items():
        if isinstance(key, str):
            raw[key.strip().lower()] = val
    values: dict[str, float] = {}
    for key in PROBABILITY_KEYS:
        val = raw.get(key)
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            return None
        number = float(val)
        if not math.isfinite(number) or number < 0:
            return None
        values[key] = number
    total = sum(values.values())
    if total <= 0:
        return None
    return {key: value / total for key, value in values.items()}


def average_report_probabilities(reports: object) -> dict[str, float] | None:
    """Mittelt die normalisierten Wahrscheinlichkeiten von Agenten-Berichten.

    Iteriert ``reports`` defensiv: Objekte ohne gültige ``probabilities``
    (z. B. ``object()``-Stubs) werden ignoriert. Liefert None, wenn kein
    gültiger Bericht vorhanden ist.
    """
    if reports is None:
        return None
    # ponytail: Duck-Typ-Grenze — die Container-Form ist hier nicht typisiert;
    # Upgrade path: fester Report-Listentyp, wenn Reports immer AgentReport sind.
    source: Any = reports
    try:
        items = list(source)
    except TypeError:
        return None
    totals: dict[str, float] = dict.fromkeys(PROBABILITY_KEYS, 0.0)
    count = 0
    for report in items:
        probabilities = normalize_probabilities(getattr(report, "probabilities", None))
        if probabilities is None:
            continue
        for key in PROBABILITY_KEYS:
            totals[key] += probabilities[key]
        count += 1
    if count == 0:
        return None
    total = sum(totals.values())
    if total <= 0:
        return None
    return {key: value / total for key, value in totals.items()}


def direction_from_return(return_fraction: float, range_threshold: float) -> str:
    """Klassifiziert eine (relative) Rendite als "up", "down" oder "range".

    ``range_threshold`` ist ein positiver Anteil (z. B. 0.001 = 0.1 %).

    ponytail: Einfache Range-Heuristik = fester Return-Threshold statt
    ATR-/Regime-basierter Klassifikation; Upgrade path: ATR- oder
    Regime-basierte Range-Klassifikation.
    """
    if return_fraction > range_threshold:
        return "up"
    if return_fraction < -range_threshold:
        return "down"
    return "range"


def horizon_to_seconds(horizon: str) -> float:
    """Konvertiert eine Horizon-Angabe (z. B. "15m", "1h", "1d") in Sekunden.

    Unterstützt die Einheiten ``s``, ``m``, ``h``, ``d``. Wirft ValueError
    bei unklaren oder ungültigen Werten (fehlende Einheit, negatives oder
    nicht-numerisches Präfix, unbekannte Einheit).
    """
    raw = horizon.strip().lower()
    if not raw:
        raise ValueError(f"Ungültiger Horizon: {horizon!r}")
    unit = raw[-1]
    if unit not in _HORIZON_UNIT_SECONDS:
        raise ValueError(
            f"Ungültiger Horizon {horizon!r}: Einheit muss eine von "
            f"{sorted(_HORIZON_UNIT_SECONDS)} sein"
        )
    try:
        value = float(raw[:-1])
    except ValueError:
        raise ValueError(f"Ungültiger Horizon: {horizon!r}") from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Ungültiger Horizon (muss > 0 sein): {horizon!r}")
    return value * _HORIZON_UNIT_SECONDS[unit]


def _rows_to_dicts(raw_rows: object) -> list[dict[str, Any]]:
    """Konvertiert SELECT-Zeilen (SQLAlchemy ``Row`` oder Test-``dict``) zu Dicts."""
    if not isinstance(raw_rows, (list, tuple)):
        return []
    rows: list[dict[str, Any]] = []
    for row in raw_rows:
        if isinstance(row, dict):
            rows.append(dict(row))
            continue
        mapping = getattr(row, "_mapping", None)
        if mapping is None:
            continue
        rows.append(dict(mapping))
    return rows


def _load_probabilities(raw: object) -> dict[str, float] | None:
    """Liest normalisierte Wahrscheinlichkeiten aus einem JSON-Spaltenwert.

    Akzeptiert ein Dict (SQLAlchemy-JSON) oder einen JSON-String.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    return normalize_probabilities(raw)


def _argmax_direction(probabilities: dict[str, float]) -> str:
    """Argmax der Prediction; bei Gleichstand: up, dann down, dann range."""
    return max(PROBABILITY_KEYS, key=probabilities.__getitem__)


def score_due_shadow_decisions(
    *,
    conn: Connection,
    provider: _CandleProviderLike,
    horizon_seconds: float,
    range_threshold: float,
    now: datetime,
    limit: int = 50,
) -> int:
    """Bewertet fällige Shadow-Entscheidungen und persistiert das Ergebnis.

    Eine Entscheidung ist fällig, wenn ``created_at <= now - horizon_seconds``
    und sie noch nicht bewertet wurde (``brier_score IS NULL``). Pro Zeile
    wird der aktuelle Close abgefragt (``provider.fetch_candles(instrument, 1)``),
    die Richtung klassifiziert, der Brier Score über ``ShadowBrierScore``
    berechnet und die Zeile aktualisiert (``brier_score``, ``actual_direction``,
    ``calibration_correct``, ``scored_at``).

    Zeilen ohne gültige Wahrscheinlichkeiten, mit ``base_close <= 0`` oder ohne
    abrufbaren Close werden übersprungen.

    Returns:
        Anzahl der erfolgreich aktualisierten Zeilen.
    """
    cutoff = now - timedelta(seconds=horizon_seconds)
    result = conn.execute(
        SELECT_DUE_SHADOW_DECISIONS,
        parameters={"cutoff": cutoff, "limit": limit},
    )
    rows = _rows_to_dicts(result.fetchall() if hasattr(result, "fetchall") else [])
    brier = ShadowBrierScore()
    scored = 0
    for row in rows:
        run_id = row["run_id"]
        instrument = row["instrument"]
        base_close = row["base_close"]
        confidence = row["confidence"]

        probabilities = _load_probabilities(row["probabilities"])
        if probabilities is None or base_close is None or base_close <= 0:
            continue

        window = provider.fetch_candles(instrument, 1)
        if window is None or len(window.close) == 0:
            continue
        actual_close = float(window.close[-1])
        return_fraction = (actual_close - float(base_close)) / float(base_close)
        actual_direction = direction_from_return(return_fraction, range_threshold)
        brier_score = brier.add(
            probabilities["up"], probabilities["down"], probabilities["range"], actual_direction
        )
        if confidence is None:
            calibration_correct: bool | None = None
        else:
            calibration_correct = _argmax_direction(probabilities) == actual_direction

        conn.execute(
            UPDATE_SCORED_SHADOW_DECISION,
            parameters={
                "brier_score": brier_score,
                "actual_direction": actual_direction,
                "calibration_correct": calibration_correct,
                "scored_at": now,
                "run_id": run_id,
            },
        )
        conn.commit()
        scored += 1
    return scored
