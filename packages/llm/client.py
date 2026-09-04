"""OpenAI-kompatibler LLM-Client über LiteLLM.

Der Client kapselt `litellm.completion` mit fester Timeout-Strategie und
mappt **alle** Ausnahmewege auf `LLMError` mit maschinellen Codes:

- ``timeout``: ``litellm.Timeout``
- ``network``: requests/urllib3- bzw. stdlib-Connectionfehler
- ``api``: sonstige LiteLLM-/API-Fehler und leere/malformed Completion-Responses
- ``unknown``: alle übrigen Exceptions

`from_env` liest die Konfiguration aus der Umgebung (Variablen:
``LITELLM_BASE_URL``, ``SMITH_LLM_MODEL``, ``LITELLM_API_KEY``) mit
Secret-Datei-Fallback unter ``/run/secrets/llm_api_key``.
"""

from __future__ import annotations

import os
from pathlib import Path

import litellm
from requests.exceptions import RequestException

from .errors import LLMError

#: Standard-Endpunkt (OpenAI-kompatible API).
DEFAULT_BASE_URL = "https://ymir-api.ifak.eu/v1"
#: Standard-Modell.
DEFAULT_MODEL = "local-fast"
#: Secret-Datei-Fallback für den API-Key (Pfad ist in Tests überschreibbar).
_SECRET_PATH = "/run/secrets/llm_api_key"
#: Klassennamen, die als Netzwerk-/Connection-Fehler gelten.
_NETWORK_ERROR_NAMES = frozenset(
    {"ConnectionError", "ConnectTimeout", "ReadTimeout", "ProxyError", "SSLError"}
)


class LLMClient:
    """Synchroner, OpenAI-kompatibler Chat-Completion-Client (LiteLLM)."""

    def __init__(self, *, base_url: str, api_key: str, model: str, timeout: float = 60.0) -> None:
        """Initialisiert den Client; leere Konfigurationswerte sind ein ``config``-Fehler."""
        if not base_url or not base_url.strip():
            raise LLMError("config", "base_url darf nicht leer sein")
        if not api_key or not api_key.strip():
            raise LLMError("config", "api_key darf nicht leer sein")
        if not model or not model.strip():
            raise LLMError("config", "model darf nicht leer sein")
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def complete(self, messages: list[dict[str, str]], *, temperature: float = 0.0) -> str:
        """Führt eine Chat-Completion aus und liefert den Antworttext.

        Alle Fehler werden auf `LLMError` gemappt; ein fehlender/leerer
        Antwortinhalt ist ein ``api``-Fehler ("empty completion").
        """
        try:
            response = litellm.completion(
                model=_litellm_model_name(self.model),
                messages=messages,
                base_url=self.base_url,
                api_key=self.api_key,
                temperature=temperature,
                timeout=self.timeout,
            )
        except Exception as exc:  # wird unten auf LLMError umgemappt
            raise _map_error(exc) from exc
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise LLMError("api", "malformed completion response") from exc
        if not content:
            raise LLMError("api", "empty completion")
        return content

    @classmethod
    def from_env(cls, model: str | None = None) -> LLMClient:
        """Erzeugt einen Client aus der Umgebung.

        Args:
            model: Modell-Override (z.B. von der CLI); ``None`` → Env-Default.

        Gelesene Variablen:
            - ``LITELLM_BASE_URL`` (Default: ``https://ymir-api.ifak.eu/v1``)
            - ``SMITH_LLM_MODEL`` (Default: ``local-fast``, durch ``model`` überschreibbar)
            - ``LITELLM_API_KEY`` — falls unset, wird die Secret-Datei
              ``/run/secrets/llm_api_key`` gelesen (Whitespace wird gestrippt).
        """
        base_url = os.environ.get("LITELLM_BASE_URL", DEFAULT_BASE_URL).strip()
        model = (model or os.environ.get("SMITH_LLM_MODEL", DEFAULT_MODEL)).strip()
        api_key = os.environ.get("LITELLM_API_KEY", "").strip()
        if not api_key:
            secret = _read_secret_file(_SECRET_PATH)
            if secret is None:
                raise LLMError(
                    "config",
                    "no API key (set LITELLM_API_KEY or /run/secrets/llm_api_key)",
                )
            api_key = secret
        return cls(base_url=base_url, api_key=api_key, model=model)


def _litellm_model_name(model: str) -> str:
    """Bereitet den Modellnamen für LiteLLM auf (``self.model`` bleibt unverändert).

    LiteLLM leitet den Provider aus dem Modellnamen ab; ein nackter Name wie
    ``local-fast`` mit eigener ``base_url`` wird verworfen ("LLM Provider NOT
    provided"). Für generische OpenAI-kompatible Endpunkte ist daher das
    ``openai/``-Präfix erforderlich. Bereits präfixte Namen (enthaltend ``/``,
    z.B. ``openai/gpt-4o-mini``) werden unverändert durchgereicht.
    """
    if "/" in model:
        return model
    return f"openai/{model}"


def _read_secret_file(path: str) -> str | None:
    """Liest eine Secret-Datei; liefert None, wenn sie fehlt oder leer ist."""
    secret_path = Path(path)
    if not secret_path.is_file():
        return None
    text = secret_path.read_text(encoding="utf-8").strip()
    return text or None


def _is_network_error(exc: BaseException) -> bool:
    """Erkennt requests/urllib3- und stdlib-Connectionfehler (Name-basiert)."""
    if isinstance(exc, RequestException):
        return True
    return any(cls.__name__ in _NETWORK_ERROR_NAMES for cls in type(exc).__mro__)


def _is_litellm_exception(exc: BaseException) -> bool:
    """Erkennt LiteLLM-Ausnahmen über den Modulnamen ihrer Klassenhierarchie."""
    return any(cls.__module__.startswith("litellm") for cls in type(exc).__mro__)


def _map_error(exc: BaseException) -> LLMError:
    """Mappt eine Exception auf einen `LLMError`-Code (in dieser Reihenfolge)."""
    if isinstance(exc, litellm.Timeout):
        return LLMError("timeout", f"{type(exc).__name__}: {exc}")
    if _is_network_error(exc):
        return LLMError("network", f"{type(exc).__name__}: {exc}")
    if _is_litellm_exception(exc):
        return LLMError("api", f"{type(exc).__name__}: {exc}")
    return LLMError("unknown", f"{type(exc).__name__}: {exc}")
