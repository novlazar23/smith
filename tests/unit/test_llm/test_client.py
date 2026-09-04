"""Tests für packages.llm.client (LiteLLM-Passthrough, Fehler-Mapping, from_env).

Es werden keine realen Netzwerk- oder LiteLLM-Aufrufe ausgeführt:
``packages.llm.client.litellm.completion`` wird monkeypatched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import litellm.exceptions
import pytest
from packages.llm.client import LLMClient
from packages.llm.errors import LLMError


class _Message:
    """Duck-Typ für die Message einer Completion-Antwort."""

    def __init__(self, content: str | None) -> None:
        self.content = content


class _Choice:
    """Duck-Typ für ein Choice einer Completion-Antwort."""

    def __init__(self, content: str | None) -> None:
        self.message = _Message(content)


class _Response:
    """Duck-Typ für den Completion-Response (choices[0].message.content)."""

    def __init__(self, content: str | None) -> None:
        self.choices = [_Choice(content)]


class _FakeCompletion:
    """Ersatz für litellm.completion: rekordiert kwargs, liefert oder raise."""

    def __init__(self, response: _Response | None = None, error: BaseException | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> _Response:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


def _make_client() -> LLMClient:
    """Test-Client mit festen Werten (keine Umgebung)."""
    return LLMClient(base_url="https://example.test/v1", api_key="sk-test", model="local-fast")


def _patch_completion(monkeypatch: pytest.MonkeyPatch, fake: _FakeCompletion) -> None:
    monkeypatch.setattr("packages.llm.client.litellm.completion", fake)


class TestComplete:
    """Aufrufverhalten und Passthrough der LiteLLM-Parameter."""

    def test_returns_content_and_passes_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeCompletion(response=_Response("hello"))
        _patch_completion(monkeypatch, fake)
        client = _make_client()

        result = client.complete([{"role": "user", "content": "hi"}], temperature=0.2)

        assert result == "hello"
        kwargs = fake.calls[0]
        # nackter Name → LiteLLM-Provider-Präfix (openai/ für OpenAI-kompatible Endpunkte)
        assert kwargs["model"] == "openai/local-fast"
        assert kwargs["base_url"] == "https://example.test/v1"
        assert kwargs["api_key"] == "sk-test"
        assert kwargs["temperature"] == 0.2
        assert kwargs["timeout"] == 60.0
        assert kwargs["messages"] == [{"role": "user", "content": "hi"}]

    def test_bare_model_name_is_prefixed_for_litellm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeCompletion(response=_Response("ok"))
        _patch_completion(monkeypatch, fake)
        client = _make_client()

        client.complete([{"role": "user", "content": "hi"}])

        assert client.model == "local-fast"  # Attribut bleibt der rohe Name
        assert fake.calls[0]["model"] == "openai/local-fast"

    def test_provider_prefixed_model_name_is_passed_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeCompletion(response=_Response("ok"))
        _patch_completion(monkeypatch, fake)
        client = LLMClient(
            base_url="https://example.test/v1", api_key="sk-test", model="openai/gpt-4o-mini"
        )

        client.complete([{"role": "user", "content": "hi"}])

        assert client.model == "openai/gpt-4o-mini"  # Attribut bleibt unverändert
        assert fake.calls[0]["model"] == "openai/gpt-4o-mini"

    def test_litellm_timeout_maps_to_timeout_code(self, monkeypatch: pytest.MonkeyPatch) -> None:
        error = litellm.exceptions.Timeout("timed out", "local-fast", "openai")
        fake = _FakeCompletion(error=error)
        _patch_completion(monkeypatch, fake)

        with pytest.raises(LLMError) as excinfo:
            _make_client().complete([{"role": "user", "content": "x"}])

        assert excinfo.value.code == "timeout"

    def test_connection_error_maps_to_network_code(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeCompletion(error=ConnectionError("refused"))
        _patch_completion(monkeypatch, fake)

        with pytest.raises(LLMError) as excinfo:
            _make_client().complete([{"role": "user", "content": "x"}])

        assert excinfo.value.code == "network"

    def test_other_litellm_error_maps_to_api_code(self, monkeypatch: pytest.MonkeyPatch) -> None:
        error = litellm.exceptions.BadRequestError("bad request", "local-fast", "openai")
        fake = _FakeCompletion(error=error)
        _patch_completion(monkeypatch, fake)

        with pytest.raises(LLMError) as excinfo:
            _make_client().complete([{"role": "user", "content": "x"}])

        assert excinfo.value.code == "api"

    def test_unexpected_error_maps_to_unknown_code(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeCompletion(error=RuntimeError("boom"))
        _patch_completion(monkeypatch, fake)

        with pytest.raises(LLMError) as excinfo:
            _make_client().complete([{"role": "user", "content": "x"}])

        assert excinfo.value.code == "unknown"

    @pytest.mark.parametrize("content", [None, ""])
    def test_empty_content_maps_to_api_code(
        self, monkeypatch: pytest.MonkeyPatch, content: str | None
    ) -> None:
        fake = _FakeCompletion(response=_Response(content))
        _patch_completion(monkeypatch, fake)

        with pytest.raises(LLMError) as excinfo:
            _make_client().complete([{"role": "user", "content": "x"}])

        assert excinfo.value.code == "api"
        assert "empty completion" in str(excinfo.value)


class TestConstructor:
    """Konfigurations-Validierung."""

    @pytest.mark.parametrize(
        ("base_url", "api_key", "model"),
        [
            ("", "sk", "m"),
            (" ", "sk", "m"),
            ("https://x", "", "m"),
            ("https://x", "sk", ""),
        ],
    )
    def test_empty_config_raises_config_error(
        self, base_url: str, api_key: str, model: str
    ) -> None:
        with pytest.raises(LLMError) as excinfo:
            LLMClient(base_url=base_url, api_key=api_key, model=model)
        assert excinfo.value.code == "config"

    def test_custom_timeout_is_stored(self) -> None:
        client = _make_client()
        assert client.timeout == 60.0
        client_5 = LLMClient(
            base_url="https://x", api_key="sk", model="m", timeout=5.0
        )
        assert client_5.timeout == 5.0


class TestFromEnv:
    """Umgebungs-Konfiguration (Variablen + Secret-Datei-Fallback)."""

    def test_reads_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LITELLM_BASE_URL", "https://env.test/v1")
        monkeypatch.setenv("SMITH_LLM_MODEL", "local-main")
        monkeypatch.setenv("LITELLM_API_KEY", "sk-env")

        client = LLMClient.from_env()

        assert client.base_url == "https://env.test/v1"
        assert client.model == "local-main"
        assert client.api_key == "sk-env"

    def test_defaults_without_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
        monkeypatch.delenv("SMITH_LLM_MODEL", raising=False)
        monkeypatch.setenv("LITELLM_API_KEY", "sk-env")

        client = LLMClient.from_env()

        assert client.base_url == "https://ymir-api.ifak.eu/v1"
        assert client.model == "local-fast"

    def test_secret_file_fallback_strips_whitespace(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        secret = tmp_path / "llm_api_key"
        secret.write_text("  sk-file \n", encoding="utf-8")
        monkeypatch.delenv("LITELLM_API_KEY", raising=False)
        monkeypatch.setattr("packages.llm.client._SECRET_PATH", str(secret))

        client = LLMClient.from_env()

        assert client.api_key == "sk-file"

    def test_missing_key_raises_config_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LITELLM_API_KEY", raising=False)
        monkeypatch.setattr("packages.llm.client._SECRET_PATH", "/nonexistent/llm_api_key")

        with pytest.raises(LLMError) as excinfo:
            LLMClient.from_env()

        assert excinfo.value.code == "config"
        assert "no API key" in str(excinfo.value)
