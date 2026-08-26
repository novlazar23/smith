"""Tests für Batch Processor."""
from __future__ import annotations

import pytest
from trading_harness.quant.batch_processor import BatchProcessor, BatchJob, BatchStatus


class TestBatchProcessor:
    def test_create_job(self):
        processor = BatchProcessor()
        job_id = processor.create_job(["BTCUSDT", "ETHUSDT"])
        assert job_id.startswith("batch_")

    def test_process_simple(self):
        processor = BatchProcessor()
        job_id = processor.create_job(["BTCUSDT", "ETHUSDT"])
        result = processor.process(job_id, lambda s: {"symbol": s})
        assert result.status == "completed"
        assert result.processed == 2

    def test_process_with_error(self):
        def bad_processor(symbol: str) -> dict:
            if symbol == "BAD":
                raise ValueError("Bad symbol")
            return {"symbol": symbol}

        processor = BatchProcessor()
        job_id = processor.create_job(["BTCUSDT", "BAD", "ETHUSDT"])
        result = processor.process(job_id, bad_processor)
        assert result.status == "failed"
        assert len(result.errors) == 1

    def test_process_chunked(self):
        processor = BatchProcessor(chunk_size=2)
        job_id = processor.create_job(["A", "B", "C", "D", "E"])
        result = processor.process_chunked(
            job_id, lambda symbols: {s: {"val": s} for s in symbols}
        )
        assert result.status == "completed"
        assert result.processed == 5

    def test_get_status(self):
        processor = BatchProcessor()
        processor.create_job(["A", "B"])
        status = processor.get_status()
        assert isinstance(status, BatchStatus)
        assert status.total_jobs == 1

    def test_cancel_job(self):
        processor = BatchProcessor()
        job_id = processor.create_job(["A", "B"])
        cancelled = processor.cancel_job(job_id)
        assert cancelled is True
        job = processor.get_job(job_id)
        assert job.status == "failed"

    def test_clear_completed(self):
        processor = BatchProcessor()
        processor.create_job(["A"])
        processor.process("batch_1", lambda s: {"val": s})
        cleared = processor.clear_completed()
        assert cleared == 1

    def test_process_nonexistent_job(self):
        processor = BatchProcessor()
        with pytest.raises(ValueError):
            processor.process("nonexistent", lambda s: s)

    def test_chunking(self):
        processor = BatchProcessor(chunk_size=3)
        chunks = processor._chunk(["A", "B", "C", "D", "E"])
        assert len(chunks) == 2
        assert chunks[0] == ["A", "B", "C"]
        assert chunks[1] == ["D", "E"]

    def test_deterministic(self):
        processor = BatchProcessor()
        processor.create_job(["A", "B"])
        r1 = processor.process("batch_1", lambda s: s)
        processor.create_job(["A", "B"])
        r2 = processor.process("batch_2", lambda s: s)
        assert r1.processed == r2.processed
