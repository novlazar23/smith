"""Fehlertypen der LLM-Infrastruktur.

Ein einziger Exception-Typ (`LLMError`) mit maschinell auswertbarem Code,
damit Strategien und Clients Fehlerarten (Timeout, Netzwerk, API, Config,
Parse) ohne String-Matching unterscheiden können.
"""

from __future__ import annotations

#: Erlaubte Fehlercodes für `LLMError`.
LLM_ERROR_CODES = frozenset({"timeout", "network", "api", "config", "parse", "unknown"})


class LLMError(Exception):
    """Fehler im LLM-Client, im Response-Cache oder in der Prompt-Strategie.

    Attributes:
        code: Maschineller Fehlercode aus `LLM_ERROR_CODES`
            (``timeout``/``network``/``api``/``config``/``parse``/``unknown``).
    """

    def __init__(self, code: str, message: str = "") -> None:
        """Erzeugt den Fehler; ein Code außerhalb der erlaubten Menge ist ein ValueError."""
        if code not in LLM_ERROR_CODES:
            raise ValueError(f"unbekannter LLM-Fehlercode: {code!r}")
        self._code = code
        super().__init__(message or code)

    @property
    def code(self) -> str:
        """Der maschinelle Fehlercode (``timeout``/``network``/``api``/``config``/``parse``/``unknown``)."""
        return self._code
