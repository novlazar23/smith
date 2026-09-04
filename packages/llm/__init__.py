"""LLM-Infrastruktur (Strategy-Zoo, Phase 2).

OpenAI-kompatibler LiteLLM-Client mit deterministischer Response-Cache
(JSONL, SHA-256-Keys), deterministischem OHLCV-Marktsnapshot und
einem einzigen, maschinell auswertbaren Fehlertyp (`LLMError`).
"""

from .cache import LLMResponseCache
from .client import LLMClient
from .errors import LLMError
from .market_summary import summarize_window

__all__ = [
    "LLMClient",
    "LLMError",
    "LLMResponseCache",
    "summarize_window",
]
