from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from trading_harness.llm.client import OpenAICompatibleClient
from trading_harness.models import (
    AgentAnalysisResult,
    AgentGenome,
    AgentSignal,
    MarketSnapshot,
)
from trading_harness.services.agent_analysis_store import PersistedAgentAnalysisStore
from trading_harness.services.agent_runtime import AgentRuntime
from trading_harness.services.db import Database


@pytest.fixture
def sample_agent():
    return AgentGenome(
        id="agent-test-1",
        category="technical",
        indicators=["RSI"],
        prompt_version="1",
        temperature=0.2,
    )


@pytest.fixture
def sample_snapshot():
    return MarketSnapshot(
        id="snap-test-1",
        symbol="BTCUSDT",
        data={"price": 50000.0, "volume": 1234.5},
    )


@pytest.fixture
def mock_llm_client():
    from unittest.mock import AsyncMock

    client = MagicMock(spec=OpenAICompatibleClient)
    client.chat = AsyncMock()
    return client


# ---------------------------------------------------------------------------
# PersistedAgentAnalysisStore fallback tests
# ---------------------------------------------------------------------------


class TestPersistedAgentAnalysisStoreFallback:
    def test_add_and_get(self):
        db = Database("postgresql://nonexistent:5432/test")
        db._ensure_pool()
        store = PersistedAgentAnalysisStore(db)

        signal = AgentSignal(
            run_id="run-1",
            agent_id="agent-1",
            snapshot_id="snap-1",
            category="technical",
            direction="LONG",
            confidence=0.8,
            reasoning="trend up",
        )
        result = AgentAnalysisResult(
            run_id="run-1",
            agent_id="agent-1",
            signal=signal,
            prompt_version="1",
            model_profile="local-main",
        )
        returned = store.add(result)
        assert returned.signal.id == signal.id

        retrieved = store.get(signal.id)
        assert retrieved is not None
        assert retrieved.signal.direction == "LONG"
        assert retrieved.signal.confidence == 0.8

    def test_get_not_found(self):
        db = Database("postgresql://nonexistent:5432/test")
        db._ensure_pool()
        store = PersistedAgentAnalysisStore(db)
        assert store.get("nonexistent-signal") is None

    def test_jsonb_fields_are_adaptable(self):
        """JSONB columns (signals, risks, raw_response) must not contain raw
        dict/list values: psycopg3 cannot adapt them, which previously broke
        every INSERT once Postgres was actually reachable (masked by the
        in-memory fallback). signals/risks must be Jsonb-wrapped, and
        dict values must have a registered JSONB dumper."""
        from psycopg.types.json import Jsonb

        from trading_harness.services.agent_analysis_store import _result_to_row

        signal = AgentSignal(
            run_id="run-1",
            agent_id="agent-1",
            snapshot_id="snap-1",
            category="technical",
            direction="LONG",
            confidence=0.8,
            reasoning="trend up",
            signals=[{"name": "sma", "value": 1.5}],
            risks=["volatility"],
        )
        result = AgentAnalysisResult(
            run_id="run-1",
            agent_id="agent-1",
            signal=signal,
            prompt_version="1",
            model_profile="local-main",
            raw_response={"raw": "llm-output"},
        )
        row = _result_to_row(result)
        assert isinstance(row["signals"], Jsonb), "signals must be Jsonb-wrapped"
        assert isinstance(row["risks"], Jsonb), "risks must be Jsonb-wrapped"
        # raw_response is a dict -> covered by the global JSONB dumper in db.py.
        assert isinstance(row["raw_response"], dict)

    def test_by_run(self):
        db = Database("postgresql://nonexistent:5432/test")
        db._ensure_pool()
        store = PersistedAgentAnalysisStore(db)

        signal1 = AgentSignal(run_id="run-1", agent_id="a1", snapshot_id="s1", category="x", direction="LONG", confidence=0.5, reasoning="")
        signal2 = AgentSignal(run_id="run-1", agent_id="a2", snapshot_id="s2", category="y", direction="SHORT", confidence=0.3, reasoning="")
        signal3 = AgentSignal(run_id="run-2", agent_id="a3", snapshot_id="s3", category="z", direction="NO_TRADE", confidence=0.1, reasoning="")

        store.add(AgentAnalysisResult(run_id="run-1", agent_id="a1", signal=signal1, prompt_version="1", model_profile="local"))
        store.add(AgentAnalysisResult(run_id="run-1", agent_id="a2", signal=signal2, prompt_version="1", model_profile="local"))
        store.add(AgentAnalysisResult(run_id="run-2", agent_id="a3", signal=signal3, prompt_version="1", model_profile="local"))

        results = store.by_run("run-1")
        assert len(results) == 2
        assert all(r.run_id == "run-1" for r in results)

    def test_by_agent(self):
        db = Database("postgresql://nonexistent:5432/test")
        db._ensure_pool()
        store = PersistedAgentAnalysisStore(db)

        signal1 = AgentSignal(run_id="r1", agent_id="agent-x", snapshot_id="s1", category="x", direction="LONG", confidence=0.5, reasoning="")
        signal2 = AgentSignal(run_id="r2", agent_id="agent-y", snapshot_id="s2", category="y", direction="SHORT", confidence=0.3, reasoning="")

        store.add(AgentAnalysisResult(run_id="r1", agent_id="agent-x", signal=signal1, prompt_version="1", model_profile="local"))
        store.add(AgentAnalysisResult(run_id="r2", agent_id="agent-y", signal=signal2, prompt_version="1", model_profile="local"))

        results = store.by_agent("agent-x")
        assert len(results) == 1
        assert results[0].signal.direction == "LONG"

    def test_by_snapshot(self):
        db = Database("postgresql://nonexistent:5432/test")
        db._ensure_pool()
        store = PersistedAgentAnalysisStore(db)

        signal1 = AgentSignal(run_id="r1", agent_id="a1", snapshot_id="snap-a", category="x", direction="LONG", confidence=0.5, reasoning="")
        signal2 = AgentSignal(run_id="r2", agent_id="a2", snapshot_id="snap-b", category="y", direction="SHORT", confidence=0.3, reasoning="")

        store.add(AgentAnalysisResult(run_id="r1", agent_id="a1", signal=signal1, prompt_version="1", model_profile="local"))
        store.add(AgentAnalysisResult(run_id="r2", agent_id="a2", signal=signal2, prompt_version="1", model_profile="local"))

        results = store.by_snapshot("snap-a")
        assert len(results) == 1
        assert results[0].signal.direction == "LONG"

    def test_all(self):
        db = Database("postgresql://nonexistent:5432/test")
        db._ensure_pool()
        store = PersistedAgentAnalysisStore(db)

        signal1 = AgentSignal(run_id="r1", agent_id="a1", snapshot_id="s1", category="x", direction="LONG", confidence=0.5, reasoning="")
        signal2 = AgentSignal(run_id="r2", agent_id="a2", snapshot_id="s2", category="y", direction="SHORT", confidence=0.3, reasoning="")

        store.add(AgentAnalysisResult(run_id="r1", agent_id="a1", signal=signal1, prompt_version="1", model_profile="local"))
        store.add(AgentAnalysisResult(run_id="r2", agent_id="a2", signal=signal2, prompt_version="1", model_profile="local"))

        assert len(store.all()) == 2


# ---------------------------------------------------------------------------
# AgentRuntime persistence tests
# ---------------------------------------------------------------------------


class TestAgentRuntimePersistence:
    @pytest.mark.asyncio
    async def test_persists_on_success(self, sample_agent, sample_snapshot, mock_llm_client):
        mock_llm_client.chat.return_value = {
            "choices": [{"message": {"content": json.dumps({
                "direction": "LONG", "confidence": 0.8, "reasoning": "up", "category": "x"
            })}}]
        }

        store = MagicMock()
        runtime = AgentRuntime(analysis_store=store, llm_client=mock_llm_client)
        result = await runtime.analyze(sample_agent, sample_snapshot, run_id="run-persist-1")

        assert result.signal.direction == "LONG"
        store.add.assert_called_once()
        called_result = store.add.call_args[0][0]
        assert called_result.signal.direction == "LONG"
        assert called_result.run_id == "run-persist-1"

    @pytest.mark.asyncio
    async def test_persists_on_llm_failure(self, sample_agent, sample_snapshot, mock_llm_client):
        mock_llm_client.chat.side_effect = Exception("timeout")

        store = MagicMock()
        runtime = AgentRuntime(analysis_store=store, llm_client=mock_llm_client)
        result = await runtime.analyze(sample_agent, sample_snapshot, run_id="run-persist-2")

        assert result.signal.direction == "NO_TRADE"
        store.add.assert_called_once()
        called_result = store.add.call_args[0][0]
        assert "timeout" in called_result.signal.reasoning

    @pytest.mark.asyncio
    async def test_no_persistence_when_store_none(self, sample_agent, sample_snapshot, mock_llm_client):
        mock_llm_client.chat.return_value = {
            "choices": [{"message": {"content": json.dumps({
                "direction": "NO_TRADE", "confidence": 0.0, "reasoning": "", "category": "x"
            })}}]
        }

        runtime = AgentRuntime(llm_client=mock_llm_client)
        result = await runtime.analyze(sample_agent, sample_snapshot, run_id="run-persist-3")

        assert result.signal.direction == "NO_TRADE"