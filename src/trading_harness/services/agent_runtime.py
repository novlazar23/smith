from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Protocol

from trading_harness.config import get_settings
from trading_harness.llm.client import OpenAICompatibleClient
from trading_harness.models import (
    AgentAnalysisResult,
    AgentGenome,
    AgentSignal,
    MarketSnapshot,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_analysis_prompt(snapshot: MarketSnapshot, agent: AgentGenome) -> str:
    """Construct the analysis prompt for a given agent and snapshot."""

    indicators_section = ""
    if agent.indicators:
        indicators_section = "\nYour methodology relies on these indicators:\n"
        for ind in agent.indicators:
            indicators_section += f"- {ind}\n"

    context_section = ""
    if agent.context_window_strategy:
        context_section = (
            f"\nContext strategy: {agent.context_window_strategy}\n"
        )

    risk_section = ""
    if agent.risk_attitude:
        risk_section = f"\nYour risk attitude is: {agent.risk_attitude}\n"

    prompt = (
        f"Analyze the following market snapshot and produce a structured trading signal.\n"
        f"Agent ID: {agent.id}\n"
        f"Category: {agent.category}\n"
        f"Prompt Version: {agent.prompt_version}\n"
        f"Date/Time: {snapshot.timestamp.isoformat()}\n"
        f"Symbol: {snapshot.symbol}\n"
        f"Snapshot ID: {snapshot.id}\n"
        f"{indicators_section}"
        f"{context_section}"
        f"{risk_section}"
        f"\nMarket Data:\n{json.dumps(snapshot.data, indent=2, default=str)}\n"
        f"\nRespond in JSON with these fields:\n"
        f"  - direction: \"LONG\", \"SHORT\", or \"NO_TRADE\"\n"
        f"  - confidence: float 0.0-1.0\n"
        f"  - reasoning: string explanation\n"
        f"  - signals: list of sub-signals with type/value\n"
        f"  - risks: list of risk factors\n"
        f"  - timestamp: current ISO timestamp"
    )
    return prompt


def build_analysis_messages(snapshot: MarketSnapshot, agent: AgentGenome) -> list[dict]:
    """Build structured message list for the LLM chat API."""

    system_prompt = (
        "You are a trading analysis agent. "
        "Analyze market snapshots and return structured JSON trading signals. "
        "Never return anything other than valid JSON. "
        "If you cannot make a reliable determination, return direction='NO_TRADE'."
    )

    user_prompt = build_analysis_prompt(snapshot, agent)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_signal_response(raw: dict[str, Any], run_id: str, agent_id: str, snapshot_id: str) -> AgentSignal:
    """Extract AgentSignal from LLM response."""
    choices = raw.get("choices", [])
    if not choices:
        return AgentSignal(
            run_id=run_id,
            agent_id=agent_id,
            snapshot_id=snapshot_id,
            category="unknown",
            direction="NO_TRADE",
            confidence=0.0,
            reasoning="No choices in response",
        )

    content = choices[0].get("message", {}).get("content", "{}")

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        logger.error("Failed to parse LLM response as JSON, using defaults")
        data = {"direction": "NO_TRADE", "confidence": 0.0, "reasoning": "Parse failure"}

    return AgentSignal(
        run_id=run_id,
        agent_id=agent_id,
        snapshot_id=snapshot_id,
        category=data.get("category", "unknown"),
        direction=data.get("direction", "NO_TRADE"),
        confidence=float(data.get("confidence", 0.0)),
        reasoning=str(data.get("reasoning", "")),
        signals=data.get("signals", []),
        risks=data.get("risks", []),
    )


# ---------------------------------------------------------------------------
# Deterministischer Analyse-Fallback (LLM nicht verfügbar)
# ---------------------------------------------------------------------------

DETERMINISTIC_SMA_FAST = 3
DETERMINISTIC_SMA_SLOW = 8
DETERMINISTIC_MIN_CLOSES = 8
DETERMINISTIC_MOMENTUM_THRESHOLD = 0.0005
DETERMINISTIC_MAX_CONFIDENCE = 0.95


def _extract_closes(data: dict[str, Any]) -> list[float]:
    """Extrahiert Close-Preise aus Snapshot-Daten.

    Unterstützt ``candles`` (Liste von OHLCV-Dicts) und ``ohlcv``
    (einzelnes Candle-Dict). Ohne nutzbare Close-Daten -> leere Liste.
    """
    candles = data.get("candles")
    if isinstance(candles, list):
        closes = [
            float(c["close"])
            for c in candles
            if isinstance(c, dict) and c.get("close") is not None
        ]
        if closes:
            return closes
    ohlcv = data.get("ohlcv")
    if isinstance(ohlcv, dict) and ohlcv.get("close") is not None:
        return [float(ohlcv["close"])]
    return []


def deterministic_signal(
    snapshot: MarketSnapshot,
    agent: AgentGenome,
    run_id: str,
    llm_error: str,
) -> AgentSignal:
    """Deterministisches SMA-Momentum-Signal als LLM-Fallback.

    Berechnet SMA_fast vs. SMA_slow über die Close-Preise der Snapshot-Daten.
    Ohne ausreichende Candle-Historie (< DETERMINISTIC_MIN_CLOSES) bleibt es
    beim harten ``NO_TRADE`` mit Confidence 0.0 (bestehende Semantik). Der
    LLM-Fehler bleibt im Reasoning sichtbar (Audit-Trail).
    """
    closes = _extract_closes(snapshot.data)
    if len(closes) < DETERMINISTIC_MIN_CLOSES:
        return AgentSignal(
            run_id=run_id,
            agent_id=agent.id,
            snapshot_id=snapshot.id,
            category=agent.category,
            direction="NO_TRADE",
            confidence=0.0,
            reasoning=f"LLM call failed: {llm_error}",
        )

    fast = sum(closes[-DETERMINISTIC_SMA_FAST:]) / DETERMINISTIC_SMA_FAST
    slow = sum(closes[-DETERMINISTIC_SMA_SLOW:]) / DETERMINISTIC_SMA_SLOW
    momentum = (fast - slow) / slow if slow else 0.0
    if momentum > DETERMINISTIC_MOMENTUM_THRESHOLD:
        direction = "LONG"
    elif momentum < -DETERMINISTIC_MOMENTUM_THRESHOLD:
        direction = "SHORT"
    else:
        direction = "NO_TRADE"

    confidence = (
        0.0
        if direction == "NO_TRADE"
        else min(DETERMINISTIC_MAX_CONFIDENCE, 0.5 + abs(momentum) * 100.0)
    )
    return AgentSignal(
        run_id=run_id,
        agent_id=agent.id,
        snapshot_id=snapshot.id,
        category=agent.category,
        direction=direction,
        confidence=confidence,
        reasoning=(
            f"LLM call failed: {llm_error}; deterministic fallback: "
            f"SMA{DETERMINISTIC_SMA_FAST}/SMA{DETERMINISTIC_SMA_SLOW} "
            f"momentum={momentum:.5f}"
        ),
    )


# ---------------------------------------------------------------------------
# Agent Runtime Service
# ---------------------------------------------------------------------------

class _AnalysisStoreProtocol(Protocol):
    """Minimal protocol for analysis result persistence."""

    def add(self, result: AgentAnalysisResult) -> AgentAnalysisResult: ...


class AgentRuntime:
    """Runs agent analysis on market snapshots via LLM.

    Orchestrates the flow:
    1. Load snapshot + agent genome
    2. Build prompt from genome config
    3. Call LLM via OpenAI-compatible client
    4. Parse response into structured signal
    5. Persist result if store provided
    6. Return complete analysis result
    """

    def __init__(
        self,
        analysis_store: _AnalysisStoreProtocol | None = None,
        llm_client: OpenAICompatibleClient | None = None,
    ) -> None:
        settings = get_settings()
        self._client = llm_client or OpenAICompatibleClient(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        )
        self._model_profile = settings.llm_model_main
        self._store = analysis_store

    async def analyze(
        self,
        agent: AgentGenome,
        snapshot: MarketSnapshot,
        run_id: str | None = None,
    ) -> AgentAnalysisResult:
        """Run a single agent's analysis on a snapshot.

        Returns:
            AgentAnalysisResult with structured signal and metadata.
        """
        if run_id is None:
            run_id = f"run-{uuid.uuid4()}"

        # Build messages
        messages = build_analysis_messages(snapshot, agent)

        # Call LLM
        logger.info(
            "AgentRuntime.analyze: run=%s agent=%s snapshot=%s model=%s",
            run_id,
            agent.id,
            snapshot.id,
            self._model_profile,
        )

        try:
            raw_response = await self._client.chat(
                model=self._model_profile,
                messages=messages,
                temperature=agent.temperature if agent.temperature else 0.2,
            )
        except Exception as exc:  # noqa: BLE001 — catch-all for external LLM service failures
            logger.error("LLM call failed for run=%s agent=%s: %s", run_id, agent.id, exc)
            # Deterministischer Fallback: SMA-Momentum statt hartem NO_TRADE.
            # Ohne Candle-Historie bleibt die bisherige NO_TRADE-Semantik.
            signal = deterministic_signal(snapshot, agent, run_id, str(exc))
            result = AgentAnalysisResult(
                run_id=run_id,
                agent_id=agent.id,
                signal=signal,
                prompt_version=agent.prompt_version,
                model_profile=self._model_profile,
                raw_response={"error": str(exc), "fallback": "deterministic_sma"},
            )
            if self._store is not None:
                self._store.add(result)
            return result

        # Parse signal from LLM response
        signal = parse_signal_response(raw_response, run_id, agent.id, snapshot.id)

        result = AgentAnalysisResult(
            run_id=run_id,
            agent_id=agent.id,
            signal=signal,
            prompt_version=agent.prompt_version,
            model_profile=self._model_profile,
            raw_response=raw_response,
        )

        # Persist result if store provided
        if self._store is not None:
            self._store.add(result)

        return result

    async def analyze_batch(
        self,
        agents: list[AgentGenome],
        snapshot: MarketSnapshot,
    ) -> list[AgentAnalysisResult]:
        """Run analysis for multiple agents on the same snapshot.

        Each agent runs independently with its own run_id.
        """
        run_id = f"run-{uuid.uuid4()}"
        results: list[AgentAnalysisResult] = []

        for agent in agents:
            result = await self.analyze(agent=agent, snapshot=snapshot, run_id=run_id)
            results.append(result)

        return results