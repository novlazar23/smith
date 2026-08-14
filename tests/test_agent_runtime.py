from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from trading_harness.llm.client import OpenAICompatibleClient
from trading_harness.models import AgentGenome, MarketSnapshot
from trading_harness.services.agent_runtime import (
    AgentRuntime,
    build_analysis_messages,
    build_analysis_prompt,
    parse_signal_response,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_agent():
    return AgentGenome(
        id="agent-test-1",
        category="technical",
        indicators=["RSI", "MACD"],
        risk_attitude="moderate",
        context_window_strategy="bounded",
        prompt_version="2",
        temperature=0.3,
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
    client = MagicMock(spec=OpenAICompatibleClient)
    client.chat = AsyncMock()
    return client


# ---------------------------------------------------------------------------
# Prompt building tests
# ---------------------------------------------------------------------------

class TestBuildAnalysisPrompt:
    def test_includes_agent_metadata(self, sample_agent, sample_snapshot):
        prompt = build_analysis_prompt(sample_snapshot, sample_agent)
        assert "agent-test-1" in prompt
        assert "technical" in prompt
        assert sample_agent.prompt_version in prompt

    def test_includes_indicators(self, sample_agent, sample_snapshot):
        prompt = build_analysis_prompt(sample_snapshot, sample_agent)
        assert "RSI" in prompt
        assert "MACD" in prompt

    def test_includes_snapshot_data(self, sample_agent, sample_snapshot):
        prompt = build_analysis_prompt(sample_snapshot, sample_agent)
        assert "BTCUSDT" in prompt
        assert "snap-test-1" in prompt
        assert "50000.0" in prompt

    def test_includes_risk_attitude(self, sample_agent, sample_snapshot):
        prompt = build_analysis_prompt(sample_snapshot, sample_agent)
        assert "moderate" in prompt

    def test_includes_context_strategy(self, sample_agent, sample_snapshot):
        prompt = build_analysis_prompt(sample_snapshot, sample_agent)
        assert "bounded" in prompt

    def test_requests_json_structure(self, sample_agent, sample_snapshot):
        prompt = build_analysis_prompt(sample_snapshot, sample_agent)
        assert "direction" in prompt
        assert "confidence" in prompt
        assert "reasoning" in prompt


class TestBuildAnalysisMessages:
    def test_returns_two_messages(self, sample_agent, sample_snapshot):
        messages = build_analysis_messages(sample_snapshot, sample_agent)
        assert len(messages) == 2

    def test_first_message_is_system(self, sample_agent, sample_snapshot):
        messages = build_analysis_messages(sample_snapshot, sample_agent)
        assert messages[0]["role"] == "system"
        assert "trading analysis agent" in messages[0]["content"]

    def test_second_message_contains_prompt(self, sample_agent, sample_snapshot):
        messages = build_analysis_messages(sample_snapshot, sample_agent)
        assert messages[1]["role"] == "user"
        assert "agent-test-1" in messages[1]["content"]


# ---------------------------------------------------------------------------
# Response parsing tests
# ---------------------------------------------------------------------------

class TestParseSignalResponse:
    def test_parses_long_signal(self):
        raw = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "direction": "LONG",
                        "confidence": 0.75,
                        "reasoning": "uptrend detected",
                        "signals": [{"type": "RSI", "value": 30}],
                        "risks": ["low_volume"],
                        "category": "technical",
                    })
                }
            }]
        }
        signal = parse_signal_response(raw, "run-1", "agent-1", "snap-1")
        assert signal.direction == "LONG"
        assert signal.confidence == 0.75
        assert signal.reasoning == "uptrend detected"
        assert len(signal.signals) == 1
        assert len(signal.risks) == 1
        assert signal.run_id == "run-1"
        assert signal.agent_id == "agent-1"
        assert signal.snapshot_id == "snap-1"

    def test_defaults_on_no_trade(self):
        raw = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "direction": "NO_TRADE",
                        "confidence": 0.1,
                        "reasoning": "insufficient data",
                    })
                }
            }]
        }
        signal = parse_signal_response(raw, "run-2", "agent-2", "snap-2")
        assert signal.direction == "NO_TRADE"
        assert signal.confidence == 0.1
        assert signal.signals == []
        assert signal.risks == []

    def test_fallback_on_bad_json(self):
        raw = {
            "choices": [{
                "message": {
                    "content": "not valid json {{{"
                }
            }]
        }
        signal = parse_signal_response(raw, "run-3", "agent-3", "snap-3")
        assert signal.direction == "NO_TRADE"
        assert signal.confidence == 0.0
        assert signal.reasoning == "Parse failure"

    def test_fallback_on_missing_choices(self):
        raw = {"choices": []}
        signal = parse_signal_response(raw, "run-4", "agent-4", "snap-4")
        assert signal.direction == "NO_TRADE"
        assert signal.confidence == 0.0


# ---------------------------------------------------------------------------
# AgentRuntime service tests
# ---------------------------------------------------------------------------

class TestAgentRuntime:
    @pytest.mark.asyncio
    async def test_analyze_success(self, mock_llm_client, sample_agent, sample_snapshot):
        mock_llm_client.chat.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "direction": "LONG",
                        "confidence": 0.8,
                        "reasoning": "bullish",
                        "category": "technical",
                    })
                }
            }]
        }

        runtime = AgentRuntime(llm_client=mock_llm_client)
        result = await runtime.analyze(sample_agent, sample_snapshot, run_id="run-5")

        assert result.run_id == "run-5"
        assert result.agent_id == "agent-test-1"
        assert result.signal.direction == "LONG"
        assert result.signal.confidence == 0.8
        assert result.prompt_version == "2"
        assert result.model_profile is not None

    @pytest.mark.asyncio
    async def test_analyze_llm_failure_returns_no_trade(self, mock_llm_client, sample_agent, sample_snapshot):
        mock_llm_client.chat.side_effect = Exception("connection refused")

        runtime = AgentRuntime(llm_client=mock_llm_client)
        result = await runtime.analyze(sample_agent, sample_snapshot, run_id="run-6")

        assert result.signal.direction == "NO_TRADE"
        assert result.signal.confidence == 0.0
        assert "connection refused" in result.signal.reasoning
        assert "error" in result.raw_response

    @pytest.mark.asyncio
    async def test_analyze_uses_agent_temperature(self, mock_llm_client, sample_agent):
        sample_agent.temperature = 0.3
        mock_llm_client.chat.return_value = {
            "choices": [{"message": {"content": json.dumps({
                "direction": "NO_TRADE", "confidence": 0.0, "reasoning": "", "category": "x"
            })}}]
        }

        runtime = AgentRuntime(llm_client=mock_llm_client)
        await runtime.analyze(sample_agent, MarketSnapshot(symbol="X", data={}))

        call_kwargs = mock_llm_client.chat.call_args
        assert call_kwargs.kwargs["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_analyze_default_temperature(self, mock_llm_client):
        agent = AgentGenome(id="a1", category="macro", prompt_version="1")
        mock_llm_client.chat.return_value = {
            "choices": [{"message": {"content": json.dumps({
                "direction": "NO_TRADE", "confidence": 0.0, "reasoning": "", "category": "x"
            })}}]
        }

        runtime = AgentRuntime(llm_client=mock_llm_client)
        await runtime.analyze(agent, MarketSnapshot(symbol="X", data={}))

        call_kwargs = mock_llm_client.chat.call_args
        assert call_kwargs.kwargs["temperature"] == 0.2

    @pytest.mark.asyncio
    async def test_analyze_batch(self, mock_llm_client, sample_agent, sample_snapshot):
        mock_llm_client.chat.return_value = {
            "choices": [{"message": {"content": json.dumps({
                "direction": "LONG", "confidence": 0.9, "reasoning": "go", "category": "technical"
            })}}]
        }

        agent2 = AgentGenome(id="agent-test-2", category="technical", prompt_version="1")
        runtime = AgentRuntime(llm_client=mock_llm_client)
        results = await runtime.analyze_batch([sample_agent, agent2], sample_snapshot)

        assert len(results) == 2
        for result in results:
            assert result.signal.direction == "LONG"
            assert result.signal.confidence == 0.9

    @pytest.mark.asyncio
    async def test_analyze_creates_run_id_when_none(self, mock_llm_client, sample_agent, sample_snapshot):
        mock_llm_client.chat.return_value = {
            "choices": [{"message": {"content": json.dumps({
                "direction": "NO_TRADE", "confidence": 0.0, "reasoning": "", "category": "x"
            })}}]
        }

        runtime = AgentRuntime(llm_client=mock_llm_client)
        result = await runtime.analyze(sample_agent, sample_snapshot, run_id=None)

        assert result.run_id.startswith("run-")