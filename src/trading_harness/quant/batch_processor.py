"""Batch Processor (Phase 10).

Batch-Verarbeitung für mehrere Symbole mit Chunking und Fortschritts-Tracking.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BatchJob:
    """Einzelner Batch-Job."""
    job_id: str
    symbols: list[str]
    status: str = "pending"  # pending, running, completed, failed
    processed: int = 0
    total: int = 0
    errors: list[str] = field(default_factory=list)
    results: dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchStatus:
    """Gesamtstatus aller Batch-Jobs."""
    total_jobs: int
    completed_jobs: int
    running_jobs: int
    pending_jobs: int
    total_symbols: int
    processed_symbols: int


class BatchProcessor:
    """Verarbeitet mehrere Symbole in Batches."""

    def __init__(self, chunk_size: int = 10, max_workers: int = 1) -> None:
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self._jobs: dict[str, BatchJob] = {}
        self._job_counter = 0

    def create_job(self, symbols: list[str]) -> str:
        """Erstellt einen neuen Batch-Job."""
        self._job_counter += 1
        job_id = f"batch_{self._job_counter}"
        self._jobs[job_id] = BatchJob(
            job_id=job_id, symbols=symbols, total=len(symbols),
        )
        return job_id

    def process(
        self,
        job_id: str,
        processor: Callable[[str], Any],
    ) -> BatchJob:
        """Verarbeitet einen Job."""
        job = self._jobs.get(job_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found")
        job.status = "running"
        for i, symbol in enumerate(job.symbols):
            try:
                result = processor(symbol)
                job.results[symbol] = result
                job.processed += 1
            except Exception as e:  # noqa: BLE001 - bewusste Fehlerisolation pro Symbol
                job.errors.append(f"{symbol}: {e!s}")
        job.status = "completed" if not job.errors else "failed"
        return job

    def process_chunked(
        self,
        job_id: str,
        processor: Callable[[list[str]], dict[str, Any]],
    ) -> BatchJob:
        """Verarbeitet einen Job in Chunks."""
        job = self._jobs.get(job_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found")
        job.status = "running"
        chunks = self._chunk(job.symbols)
        for chunk in chunks:
            try:
                results = processor(chunk)
                for symbol, result in results.items():
                    job.results[symbol] = result
                    job.processed += 1
            except Exception as e:  # noqa: BLE001 - bewusste Fehlerisolation pro Chunk
                for symbol in chunk:
                    job.errors.append(f"{symbol}: {e!s}")
        job.status = "completed" if not job.errors else "failed"
        return job

    def get_job(self, job_id: str) -> BatchJob | None:
        return self._jobs.get(job_id)

    def get_status(self) -> BatchStatus:
        """Gibt Gesamtstatus zurück."""
        jobs = list(self._jobs.values())
        total_symbols = sum(j.total for j in jobs)
        processed = sum(j.processed for j in jobs)
        return BatchStatus(
            total_jobs=len(jobs),
            completed_jobs=sum(1 for j in jobs if j.status == "completed"),
            running_jobs=sum(1 for j in jobs if j.status == "running"),
            pending_jobs=sum(1 for j in jobs if j.status == "pending"),
            total_symbols=total_symbols,
            processed_symbols=processed,
        )

    def _chunk(self, items: list[str]) -> list[list[str]]:
        """Teilt Liste in Chunks."""
        return [items[i:i + self.chunk_size] for i in range(0, len(items), self.chunk_size)]

    def cancel_job(self, job_id: str) -> bool:
        """Bricht einen Job ab."""
        job = self._jobs.get(job_id)
        if job and job.status == "pending":
            job.status = "failed"
            job.errors.append("Cancelled")
            return True
        return False

    def clear_completed(self) -> int:
        """Entfernt abgeschlossene Jobs. Gibt Anzahl zurück."""
        completed = [jid for jid, j in self._jobs.items() if j.status == "completed"]
        for jid in completed:
            del self._jobs[jid]
        return len(completed)
