from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from packages.domain.data_quality.quarantine import QuarantineManager
from packages.domain.data_quality.validator import (
    GapDetector,
    MarketDataValidator,
    Severity,
    ValidationIssue,
    ValidationResult,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReplayConfig:
    """Konfiguration für die deterministische Replay-Engine.

    Felder:
        input_path: Pfad zu Verzeichnis oder Datei mit Markt-Daten (JSON/CSV)
        event_type: Gefilterter Event-Typ, "all" = alle Typen
        speed_multiplier: Faktor für Wiedergabegeschwindigkeit (1.0 = Echtzeit)
        gap_threshold_ms: Schwellwert in ms, oberhalb dessen eine Lücke gemeldet wird
        quarantine_threshold: Quality-Score unter diesem Wert → Quarantäne
        validate_events: Ob Events vor Replay validiert werden
        output_path: Optionaler Pfad zum Schreiben der Replay-Ergebnisse
    """

    input_path: str
    event_type: str = "all"
    speed_multiplier: float = 1.0
    gap_threshold_ms: float = 1000.0
    quarantine_threshold: float = 0.5
    validate_events: bool = True
    output_path: str | None = None


class ReplayEngine:
    """Deterministische Replay-Engine für Backtesting.

    - Sortiert Events nach event_time
    - Erkennt Sequenzlücken
    - Kontrolliert Wiedergabegeschwindigkeit
    - Validiert Events und quarantäniert fehlerhafte
    - Reproduzierbar: gleiche Eingabe → gleiche Ausgabe
    """

    def __init__(self, config: ReplayConfig) -> None:
        self.config = config
        self._validator = MarketDataValidator()
        self._gap_detector = GapDetector()
        self._quarantine_manager = QuarantineManager()
        self._events: list[dict[str, Any]] = []
        self._replayed: list[dict[str, Any]] = []
        self._validation_results: list[ValidationResult] = []
        self._gap_results: list[ValidationResult] = []
        self._replay_log: list[dict[str, Any]] = []
        self._replay_start: float | None = None
        self._replay_end: float | None = None
        self._valid_count: int = 0
        self._invalid_count: int = 0
        self._quarantined_count: int = 0

    # ── Event Loading ──────────────────────────────────────────────────

    def load_events(self) -> list[dict[str, Any]]:
        """Lädt Events aus JSON/CSV-Dateien im input_path.

        Unterstützt:
        - Einzelne JSON-Datei → direkt parsen
        - JSONL-Datei → jede Zeile ein Event
        - Verzeichnis mit JSON/JSONL-Dateien → alle kombinieren
        - CSV-Dateien → erste Zeile als Header, konvertieren zu dicts

        Returns:
            Liste aller geladenen Event-Dicts, deterministisch sortiert
        """
        input_path = Path(self.config.input_path)

        if not input_path.exists():
            logger.error("Input path does not exist: %s", self.config.input_path)
            return []

        events: list[dict[str, Any]] = []

        if input_path.is_file():
            events = self._load_file(input_path)
        elif input_path.is_dir():
            for filepath in sorted(input_path.iterdir()):
                if filepath.is_file():
                    events.extend(self._load_file(filepath))

        # Nach event_type filtern wenn spezifiziert
        if self.config.event_type != "all":
            events = [e for e in events if e.get("type") == self.config.event_type]

        self._events = events
        logger.info("Loaded %d events from %s", len(events), input_path)
        return events

    def _load_file(self, filepath: Path) -> list[dict[str, Any]]:
        """Lädt Events aus einer einzelnen Datei."""
        suffix = filepath.suffix.lower()

        if suffix == ".json":
            return self._load_json(filepath)
        elif suffix == ".jsonl":
            return self._load_jsonl(filepath)
        elif suffix == ".csv":
            return self._load_csv(filepath)

        logger.warning("Unsupported file type: %s, skipping", filepath)
        return []

    def _load_json(self, filepath: Path) -> list[dict[str, Any]]:
        """Lädt Events aus einer JSON-Datei. Unterstützt Array oder einzelne Objekte."""
        try:
            with filepath.open(encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
            logger.warning("Unexpected JSON structure in %s", filepath)
            return []
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load JSON file %s: %s", filepath, exc)
            return []

    def _load_jsonl(self, filepath: Path) -> list[dict[str, Any]]:
        """Lädt Events aus einer JSONL-Datei (ein JSON-Objekt pro Zeile)."""
        events: list[dict[str, Any]] = []
        try:
            with filepath.open(encoding="utf-8") as fh:
                for line_no, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("Skipping invalid JSONL line %d in %s", line_no, filepath)
        except OSError as exc:
            logger.error("Failed to load JSONL file %s: %s", filepath, exc)
        return events

    def _load_csv(self, filepath: Path) -> list[dict[str, Any]]:
        """Lädt Events aus einer CSV-Datei."""
        events: list[dict[str, Any]] = []
        try:
            with filepath.open(encoding="utf-8") as fh:
                lines = fh.readlines()

            if not lines:
                return []

            header = [h.strip() for h in lines[0].strip().split(",")]

            for row_no, line in enumerate(lines[1:], 2):
                line = line.strip()
                if not line:
                    continue
                values = [v.strip() for v in line.split(",")]
                if len(values) != len(header):
                    logger.warning(
                        "CSV row %d column count mismatch in %s", row_no, filepath
                    )
                    continue
                row_dict: dict[str, Any] = dict(zip(header, values, strict=True))
                events.append(row_dict)

        except OSError as exc:
            logger.error("Failed to load CSV file %s: %s", filepath, exc)

        return events

    # ── Sorting ────────────────────────────────────────────────────────

    def sort_events(self, events: list[dict]) -> list[dict]:
        """Sortiert Events nach event_time (oder timestamp).

        Verwendet eine deterministische, stabile Sortierung nach dem
        erstbesten verfügbaren Zeitstempel-Feld pro Event.

        Returns:
            Neuer, sortierter Events-Liste
        """
        def _sort_key(event: dict) -> tuple[str, Any]:
            """Ermittelt primären Sortierschlüssel."""
            for key in ("event_time", "timestamp", "open_time", "close_time"):
                value = event.get(key)
                if value is not None:
                    return (key, value)
            # Fallback: String-Vergleich der kompletten Event-Daten
            # (deterministisch, wenn alle Events unique sind)
            return ("__fallback__", json.dumps(event, sort_keys=True))

        sorted_events = sorted(events, key=_sort_key)
        logger.info("Sorted %d events by time", len(sorted_events))
        return sorted_events

    # ── Gap Detection ──────────────────────────────────────────────────

    def detect_gaps(self, events: list[dict]) -> list[ValidationResult]:
        """Erkennt Sequenzlücken in Event-Strömen.

        Trackt pro (instrument, venue, event_type) die letzte bekannte
        Sequence und meldet Lücken, wenn die aktuelle Sequence nicht
        nahtlos fortfährt. Erkennt auch zeitliche Lücken zwischen
        aufeinanderfolgenden Events.

        Returns:
            Liste von ValidationResult-Objekten für jedes detektierte Gap
        """
        gaps: list[ValidationResult] = []

        for event in events:
            # Schlüsselfelder extrahieren
            composite = {
                "instrument": event.get("instrument", "unknown"),
                "venue": event.get("venue", "unknown"),
                "event_type": event.get("type", event.get("event_type", "unknown")),
            }

            # Sequence aus metadata oder direkt
            sequence = self._extract_sequence(event)
            result = self._gap_detector.check_gap(sequence, composite)

            if not result.is_valid:
                gaps.append(result)

        # Zeitliche Lücken zwischen aufeinanderfolgenden Events prüfen
        for i in range(len(events) - 1):
            curr = events[i]
            nxt = events[i + 1]
            curr_ts = self._resolve_timestamp(curr)
            nxt_ts = self._resolve_timestamp(nxt)

            if curr_ts is None or nxt_ts is None:
                continue

            diff_ms = abs((nxt_ts - curr_ts).total_seconds() * 1000)
            if diff_ms > self.config.gap_threshold_ms:
                gaps.append(ValidationResult(
                    event_type=curr.get("type", "unknown"),
                    is_valid=False,
                    issues=[
                        ValidationIssue(
                            field="time_gap",
                            message=(
                                f"Time gap of {diff_ms:.1f}ms between consecutive events "
                                f"(threshold: {self.config.gap_threshold_ms}ms)"
                            ),
                            severity=Severity.WARNING,
                            value=diff_ms,
                        )
                    ],
                    quality_score=0.5,
                ))

        self._gap_results = gaps
        logger.info("Detected %d gap results across %d events", len(gaps), len(events))
        return gaps

    def _extract_sequence(self, event: dict) -> int:
        """Ermittelt die Sequence-Nummer aus einem Event."""
        # sequence kann direkt im Event oder in metadata stehen
        if "sequence" in event:
            seq = event["sequence"]
        else:
            metadata = event.get("metadata", {})
            if isinstance(metadata, dict) and "sequence" in metadata:
                seq = metadata["sequence"]
            else:
                return 0

        try:
            return int(seq)
        except (ValueError, TypeError):
            return 0

    @staticmethod
    def _resolve_timestamp(event: dict) -> datetime | None:
        """Ermittelt den primären Timestamp aus einem Event."""
        for key in ("event_time", "timestamp", "open_time"):
            value = event.get(key)
            if value is None:
                continue
            if isinstance(value, datetime):
                return value
            if isinstance(value, (int, float)):
                try:
                    return datetime.fromtimestamp(value, tz=UTC)
                except (OSError, OverflowError):
                    continue
            if isinstance(value, str):
                try:
                    return datetime.fromisoformat(value).replace(tzinfo=UTC)
                except (ValueError, TypeError):
                    continue
        return None

    # ── Validation & Quarantine ────────────────────────────────────────

    def validate_and_filter(
        self,
        events: list[dict],
    ) -> tuple[list[dict], list[ValidationResult]]:
        """Validiert Events, filtert fehlerhafte und quarantäniert diese.

        Args:
            events: Liste zu validierender Events

        Returns:
            Tuple von (valid_events, validation_results)
        """
        valid: list[dict] = []
        results: list[ValidationResult] = []

        for event in events:
            result = self._validator.validate(event)
            results.append(result)

            if result.is_valid:
                valid.append(event)
                self._valid_count += 1
            else:
                self._invalid_count += 1

                # Prüfe auf Quarantäne
                if self.config.validate_events and self.config.quarantine_threshold is not None:
                    classification = self._validator.classify_quality(result)
                    if classification == "quarantine":
                        issues_list = [
                            {
                                "field": issue.field,
                                "message": issue.message,
                                "severity": issue.severity,
                            }
                            for issue in result.issues
                        ]
                        self._quarantine_manager.evaluate_and_quarantine(
                            event=event,
                            quality_score=result.quality_score,
                            issues=issues_list,
                        )
                        self._quarantined_count += 1

        self._validation_results = results
        logger.info(
            "Validation complete: %d valid, %d invalid, %d quarantined",
            self._valid_count,
            self._invalid_count,
            self._quarantined_count,
        )
        return valid, results

    # ── Main Replay ────────────────────────────────────────────────────

    def run(self) -> list[dict[str, Any]]:
        """Führt deterministisches Replay durch.

        Ablauf:
        1. Events laden
        2. Sortieren nach event_time
        3. Validieren und filtern
        4. Gap-Detection
        5. Output schreiben (optional)

        Returns:
            Liste der replayed Events (reproduzierbar)
        """
        self._replay_start = time.monotonic()

        # Load
        events = self.load_events()
        if not events:
            logger.warning("No events loaded, replay skipped")
            return []

        # Sort
        events = self.sort_events(events)

        # Validate & filter
        if self.config.validate_events:
            valid_events, _ = self.validate_and_filter(events)
        else:
            valid_events = events

        # Gap detection
        self.detect_gaps(valid_events)

        # Execute
        self._replayed = list(valid_events)
        self._replay_end = time.monotonic()

        # Write output
        if self.config.output_path:
            self.save_replay_log(self.config.output_path)

        logger.info(
            "Replay complete: %d events in %.3fs",
            len(self._replayed),
            (self._replay_end - self._replay_start) if self._replay_start else 0,
        )
        return self._replayed

    async def run_async(self) -> list[dict[str, Any]]:
        """Asynchrones Replay mit Geschwindigkeitskontrolle.

        Spielt Events zeitgesteuert ab, abhängig von speed_multiplier.

        Returns:
            Liste der replayed Events
        """
        self._replay_start = time.monotonic()

        events = self.load_events()
        if not events:
            logger.warning("No events loaded, async replay skipped")
            return []

        events = self.sort_events(events)

        if self.config.validate_events:
            valid_events, _ = self.validate_and_filter(events)
        else:
            valid_events = events

        self.detect_gaps(valid_events)

        # Geschwindigkeitskontrolle: Berechne Ziel-Zeit zwischen Events
        speed = max(self.config.speed_multiplier, 0.01)
        log_interval = max(len(valid_events) // 10, 1)

        for idx, event in enumerate(valid_events):
            self._replayed.append(event)

            # Wiedergabe-Log
            self._replay_log.append({
                "index": idx,
                "event_type": event.get("type", event.get("event_type", "unknown")),
                "instrument": event.get("instrument", "unknown"),
                "venue": event.get("venue", "unknown"),
                "event_time": event.get("event_time", event.get("timestamp")),
                "replayed_at": datetime.now(UTC).isoformat(),
            })

            # Geschwindigkeitskontrolle
            if idx > 0 and speed > 0:
                # Ziel-Delay zwischen Events (Sekunden)
                prev_event = valid_events[idx - 1]
                curr_ts = self._resolve_timestamp(event)
                prev_ts = self._resolve_timestamp(prev_event)
                if curr_ts and prev_ts:
                    actual_delta = (curr_ts - prev_ts).total_seconds()
                    target_delay = actual_delta / speed
                    if target_delay > 0:
                        await asyncio.sleep(min(target_delay, 60.0))
                else:
                    # Fallback: gleichmäßiger Delay
                    await asyncio.sleep(1.0 / speed)

            if (idx + 1) % log_interval == 0:
                logger.debug("Async replay: %d/%d events processed", idx + 1, len(valid_events))

        self._replay_end = time.monotonic()

        if self.config.output_path:
            self.save_replay_log(self.config.output_path)

        logger.info(
            "Async replay complete: %d events in %.3fs",
            len(self._replayed),
            (self._replay_end - self._replay_start) if self._replay_start else 0,
        )
        return self._replayed

    # ── Batch Replay ───────────────────────────────────────────────────

    def replay_batch(
        self,
        events: list[dict],
        handler: Callable[[dict], None],
    ) -> None:
        """Spielt Events in Batch ab (ohne Zeitverzögerung).

        Rufen handler(event) für jedes Event auf.

        Args:
            events: Liste der Events
            handler: Callback für jedes Event
        """
        count = 0
        for event in events:
            try:
                handler(event)
                count += 1
            except Exception as exc:
                logger.error("Handler error for event: %s", exc)
        logger.info("Batch replay complete: %d events processed", count)

    # ── Statistics ─────────────────────────────────────────────────────

    def get_statistics(self) -> dict[str, Any]:
        """Gibt Statistiken des Replay-Vorgangs zurück."""
        duration = (
            (self._replay_end - self._replay_start)
            if self._replay_start is not None and self._replay_end is not None
            else None
        )

        return {
            "total_loaded": len(self._events),
            "total_valid": self._valid_count,
            "total_invalid": self._invalid_count,
            "total_quarantined": self._quarantined_count,
            "total_replayed": len(self._replayed),
            "total_gaps": len(self._gap_results),
            "validation_results": len(self._validation_results),
            "duration_seconds": duration,
            "quarantine_stats": self._quarantine_manager.get_stats(),
        }

    # ── Output ─────────────────────────────────────────────────────────

    def save_replay_log(self, output_path: str) -> None:
        """Speichert Replay-Protokoll als JSON-Datei.

        Enthält alle replayed Events, Validierungsergebnisse, Gaps und
        Statistiken.

        Args:
            output_path: Pfad zur Ausgabedatei
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        stats = self.get_statistics()

        log_data: dict[str, Any] = {
            "metadata": {
                "input_path": self.config.input_path,
                "event_type_filter": self.config.event_type,
                "output_path": output_path,
                "generated_at": datetime.now(UTC).isoformat(),
            },
            "statistics": stats,
            "replay_events": self._replayed,
            "gap_results": [
                {
                    "event_type": r.event_type,
                    "is_valid": r.is_valid,
                    "quality_score": r.quality_score,
                    "issues": [
                        {
                            "field": issue.field,
                            "message": issue.message,
                            "severity": issue.severity,
                        }
                        for issue in r.issues
                    ],
                }
                for r in self._gap_results
            ],
            "replay_log": self._replay_log,
        }

        with Path(output).open(mode="w", encoding="utf-8") as fh:
            json.dump(log_data, fh, indent=2, default=str)

        logger.info("Replay log saved to %s", output)

    # ── Expose QuarantineManager ───────────────────────────────────────

    @property
    def quarantine_manager(self) -> QuarantineManager:
        """Zugriff auf den internen QuarantineManager."""
        return self._quarantine_manager


__all__: list[str] = [
    "ReplayConfig",
    "ReplayEngine",
]

