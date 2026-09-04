"""PromptStrategy — LLM-gesteuerte Long-Only-Backtest-Strategie (Phase 2).

Alle ``llm_every``-ten Kerzen (nach ``min_candles`` Warmup) wird ein
deterministischer OHLCV-Snapshot (`packages.llm.summarize_window`) gebaut
und über den Prompt an den LLM-Client eine Positionsentscheidung für die
nächsten 1-5 Tage abgefragt. Antworten werden über einen SHA-256-Response-
Cache (Key = Modell + Prompt-Version + Snapshot-Text) wiederverwendet,
damit Replay-Läufe deterministisch und ohne neue API-Calls bleiben.

Long-Only-Semantik (wie `AgentEnsembleStrategy._build_signal`):
  - ``BUY``  → StrategySignal BUY (position_size = trade_notional / initial_capital)
  - ``SELL`` → StrategySignal SELL (Position glattstellen, position_size 0.0)
  - ``NONE`` → kein Signal (Abstinenz)

Fehler (LLM-Client-Ausfall, Parse-/Validierungsfehler der Antwort) führen
zur Abstinenz: kein Signal, ``llm_failures`` wird erhöht, ein Lauf bricht
nie durch eine schlechte Antwort ab.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from typing import Any

import numpy as np
from packages.backtesting.core import Candle
from packages.backtesting.strategies import BaseStrategy, SignalAction, StrategySignal
from packages.llm import LLMClient, LLMError, LLMResponseCache, summarize_window

logger = logging.getLogger(__name__)

#: Version des System-Prompts (Teil des Cache-Keys — ein Prompt-Wechsel
#: invalidiert den Cache dadurch automatisch).
PROMPT_VERSION = "1"

SYSTEM_PROMPT = (
    "You are a disciplined crypto swing trader. You receive a deterministic "
    "OHLCV snapshot (1m bars, lookback up to 200 bars) and decide the position "
    "over the next 1-5 days for a LONG-ONLY account.\n"
    "Reply with ONLY a single JSON object, no markdown, with exactly this schema:\n"
    '{"decision": "BUY" | "SELL" | "NONE", "confidence": <0..1>, "up": <0..1>, '
    '"down": <0..1>, "range": <0..1>, "hypothesis": "<= 30 words"}\n'
    "up, down and range must sum to 1.0. BUY = open/increase long, SELL = "
    "close long, NONE = no action. Be conservative and abstain (NONE) unless "
    "the evidence is clearly directional."
)

#: Erlaubte Entscheidungswerte in der LLM-Antwort.
_DECISIONS = frozenset({"BUY", "SELL", "NONE"})

#: Toleranz für die Wahrscheinlichkeits-Summe up+down+range = 1.0.
_PROBABILITY_TOLERANCE = 1e-4

#: Standard-Pfad der Response-Cache-Datei.
_DEFAULT_CACHE_PATH = "/app/backtest_reports/llm_cache/llm_cache.jsonl"


class _NullCache:
    """No-OP-Ersatz für `LLMResponseCache` (gleiche Schnittstelle).

    Wird verwendet, wenn das Cache-Verzeichnis nicht schreibbar ist (z.B.
    außerhalb des Containers): jede Abfrage ist ein Miss, ``put`` ist ein
    No-Op. Die Strategie bleibt damit voll funktionsfähig, nur Replay-Läufe
    fragen in diesem Fall jedes Fenster neu ab.
    """

    def get(self, key: str) -> str | None:
        """Immer None (kein Cache)."""
        return None

    def put(self, key: str, value: str) -> None:
        """No-Op."""
        return None


class PromptStrategy(BaseStrategy):
    """LLM-Prompt-Strategie mit deterministischem Snapshot und Response-Cache."""

    def __init__(
        self,
        instrument: str,
        *,
        client: LLMClient | None = None,
        cache_path: str | None = None,
        llm_every: int = 15,
        min_candles: int = 120,
        initial_capital: float = 100_000.0,
        trade_notional: float = 2000.0,
    ) -> None:
        """Initialisiert die Strategie (``client=None`` → `LLMClient.from_env`).

        Args:
            instrument: Handelspaar, z.B. ``"BTC/USDT"``.
            client: LLM-Client (in Tests injizierbar); Default aus Umgebung.
            cache_path: Pfad der JSONL-Cache-Datei; nicht schreibbares
                Verzeichnis → No-OP-Cache (siehe `_NullCache`).
            llm_every: LLM-Abfrage alle N Bars (>= 1).
            min_candles: Mindestanzahl Kerzen vor der ersten Abfrage (>= 30).
            initial_capital: Startkapital (für position_size).
            trade_notional: Ziel-Nominal pro BUY (für position_size).
        """
        super().__init__(name="prompt-strategy")
        if llm_every < 1:
            raise ValueError("llm_every muss >= 1 sein")
        if min_candles < 30:
            raise ValueError("min_candles muss >= 30 sein")
        self.instrument = instrument
        self.llm_every = llm_every
        self.min_candles = min_candles
        self.initial_capital = initial_capital
        self.trade_notional = trade_notional
        # Runner-Vertrag (default_config/run_backtest): warmup = Mindestkerzen.
        self.candle_limit = min_candles
        self.client = client if client is not None else LLMClient.from_env()
        self.model_name: str = self.client.model
        self.cache: LLMResponseCache | _NullCache = self._open_cache(cache_path)
        self._window: deque[Candle] = deque(maxlen=200)
        self._bars_seen = 0
        self.llm_calls = 0
        self.llm_cache_hits = 0
        self.llm_failures = 0
        self.n_buy_signals = 0
        self.n_sell_signals = 0

    def _open_cache(self, cache_path: str | None) -> LLMResponseCache | _NullCache:
        """Öffnet den Response-Cache; nicht schreibbares Verzeichnis → No-OP-Cache."""
        path = cache_path or _DEFAULT_CACHE_PATH
        try:
            return LLMResponseCache(path)
        except OSError:
            logger.warning("LLM-Cache-Pfad %s nicht schreibbar, verwende No-OP-Cache", path)
            return _NullCache()

    def on_bar(self, candle: Candle) -> StrategySignal | None:
        """Neue Kerze; alle ``llm_every`` Bars LLM-Entscheidung (ab Warmup)."""
        self._window.append(candle)
        self._bars_seen += 1
        if self._bars_seen % self.llm_every != 0:
            return None
        if len(self._window) < self.min_candles:
            return None
        return self._evaluate(candle)

    def _evaluate(self, candle: Candle) -> StrategySignal | None:
        """Baute den Snapshot, fragt Cache/LLM ab, parst und mappt auf ein Signal."""
        summary = summarize_window(
            open=np.array([c.open for c in self._window], dtype=np.float64),
            high=np.array([c.high for c in self._window], dtype=np.float64),
            low=np.array([c.low for c in self._window], dtype=np.float64),
            close=np.array([c.close for c in self._window], dtype=np.float64),
            volume=np.array([c.volume for c in self._window], dtype=np.float64),
        )
        cache_key = LLMResponseCache.key(self.model_name, PROMPT_VERSION, summary)
        response = self.cache.get(cache_key)
        if response is None:
            self.llm_calls += 1
            try:
                response = self.client.complete(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": summary},
                    ]
                )
            except LLMError as exc:
                self._abstain(f"LLM-Client-Fehler ({exc.code}): {exc}")
                return None
            self.cache.put(cache_key, response)
        else:
            self.llm_cache_hits += 1
        try:
            payload = _parse_decision(response)
        except LLMError as exc:
            self._abstain(f"LLM-Antwort nicht verwertbar ({exc.code}): {exc}")
            return None
        return self._build_signal(payload, candle)

    def _abstain(self, message: str) -> None:
        """Loggt eine Fehlleistung und zählt sie (kein Signal, kein Crash)."""
        logger.warning("prompt-strategy: %s — abstain (kein Signal)", message)
        self.llm_failures += 1

    def _build_signal(self, payload: dict[str, Any], candle: Candle) -> StrategySignal | None:
        """Mappt die LLM-Entscheidung auf ein StrategySignal (long-only)."""
        decision = str(payload["decision"])
        confidence = float(payload["confidence"])
        if decision == "NONE":
            return None
        metadata: dict[str, Any] = {
            "decision": decision,
            "confidence": confidence,
            "hypothesis": str(payload["hypothesis"]),
            "up": float(payload["up"]),
            "down": float(payload["down"]),
            "range": float(payload["range"]),
            "prompt_version": PROMPT_VERSION,
            "instrument": self.instrument,
        }
        if decision == "BUY":
            self.n_buy_signals += 1
            return StrategySignal(
                action=SignalAction.BUY,
                symbol=candle.symbol,
                confidence=confidence,
                reason=f"prompt-llm: BUY (decision confidence={confidence:.3f})",
                position_size=self.trade_notional / self.initial_capital,
                timestamp=candle.timestamp,
                metadata=metadata,
            )
        self.n_sell_signals += 1
        return StrategySignal(
            action=SignalAction.SELL,
            symbol=candle.symbol,
            confidence=confidence,
            reason=f"prompt-llm: SELL (decision confidence={confidence:.3f})",
            position_size=0.0,
            timestamp=candle.timestamp,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Zusammenfassung der Strategie für Report/MLflow."""
        return {
            "name": "prompt-strategy",
            "instrument": self.instrument,
            "prompt_version": PROMPT_VERSION,
            "model": self.model_name,
            "llm_every": self.llm_every,
            "min_candles": self.min_candles,
            "llm_calls": self.llm_calls,
            "llm_cache_hits": self.llm_cache_hits,
            "llm_failures": self.llm_failures,
            "n_buy_signals": self.n_buy_signals,
            "n_sell_signals": self.n_sell_signals,
        }


def _parse_decision(text: str) -> dict[str, Any]:
    """Extrahiert das erste JSON-Objekt und validiert das Antwort-Schema.

    Umschließender Text wird toleriert; fehlendes/ungültiges/inkonsistentes
    JSON ist ein `LLMError` mit Code ``parse``.
    """
    start = text.find("{")
    if start < 0:
        raise LLMError("parse", "kein JSON-Objekt in der Antwort")
    try:
        payload, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise LLMError("parse", f"ungültiges JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise LLMError("parse", "JSON-Wurzel ist kein Objekt")
    _validate_decision(payload)
    return payload


def _validate_decision(payload: dict[str, Any]) -> None:
    """Prüft Entscheidung, Konfidenz, Wahrscheinlichkeits-Summe und Hypothese."""
    decision = payload.get("decision")
    if decision not in _DECISIONS:
        raise LLMError("parse", f"decision muss BUY/SELL/NONE sein, ist {decision!r}")
    confidence = _unit_float(payload.get("confidence"), "confidence")
    up = _unit_float(payload.get("up"), "up")
    down = _unit_float(payload.get("down"), "down")
    range_prob = _unit_float(payload.get("range"), "range")
    if abs(up + down + range_prob - 1.0) > _PROBABILITY_TOLERANCE:
        raise LLMError("parse", "up+down+range muss 1.0 sein (± 0.0001)")
    hypothesis = payload.get("hypothesis")
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        raise LLMError("parse", "hypothesis muss ein nicht-leerer Text sein")
    payload["confidence"] = confidence
    payload["up"] = up
    payload["down"] = down
    payload["range"] = range_prob


def _unit_float(value: object, name: str) -> float:
    """Wandelt einen Wert in einen Float aus [0, 1] um (bool wird abgelehnt)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LLMError("parse", f"{name} muss eine Zahl in [0,1] sein, ist {value!r}")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise LLMError("parse", f"{name} muss in [0,1] liegen, ist {number}")
    return number
