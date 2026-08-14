from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from trading_harness.llm.client import OpenAICompatibleClient


def _make_response(json_data: dict, status_code: int = 200) -> httpx.Response:
    """Create a mock httpx.Response."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.json.return_value = json_data
    response.raise_for_status.return_value = None
    return response


def _make_error_response(status_code: int, reason: str) -> httpx.Response:
    """Create a mock httpx.Response that raises on raise_for_status."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.json.return_value = {}
    error = httpx.HTTPStatusError(reason, request=MagicMock(), response=response)
    response.raise_for_status.side_effect = error
    return response


@pytest.mark.asyncio
async def test_chat_sends_correct_payload():
    """Client posts correct JSON payload to the chat completions endpoint."""
    json_data = {
        "choices": [{"message": {"content": json.dumps({
            "direction": "LONG",
            "confidence": 0.8,
            "reasoning": "trend up",
        })}}],
    }
    mock_response = _make_response(json_data)

    mock_post = AsyncMock(return_value=mock_response)

    mock_client = MagicMock()
    mock_client.post = mock_post
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("trading_harness.llm.client.httpx.AsyncClient", return_value=mock_client):
        llm = OpenAICompatibleClient(base_url="http://localhost:8000/v1", api_key="test-key")
        result = await llm.chat(
            model="local-main",
            messages=[
                {"role": "system", "content": "You are a trader"},
                {"role": "user", "content": "Analyze this"},
            ],
            temperature=0.3,
        )

    assert result["choices"][0]["message"]["content"] == json.dumps({
        "direction": "LONG",
        "confidence": 0.8,
        "reasoning": "trend up",
    })

    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert call_args[0][0] == "http://localhost:8000/v1/chat/completions"
    payload = call_args[1]["json"]
    assert payload["model"] == "local-main"
    assert payload["temperature"] == 0.3
    assert len(payload["messages"]) == 2


@pytest.mark.asyncio
async def test_chat_handles_http_error():
    """Client raises on HTTP error responses (4xx/5xx)."""
    mock_response = _make_error_response(429, "rate limited")

    mock_post = AsyncMock(return_value=mock_response)

    mock_client = MagicMock()
    mock_client.post = mock_post
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("trading_harness.llm.client.httpx.AsyncClient", return_value=mock_client):
        llm = OpenAICompatibleClient(base_url="http://localhost:8000/v1", api_key="test-key")
        with pytest.raises(httpx.HTTPStatusError):
            await llm.chat(
                model="local-main",
                messages=[{"role": "user", "content": "test"}],
            )


@pytest.mark.asyncio
async def test_chat_default_temperature():
    """Client uses temperature=0.2 when not specified."""
    json_data = {"choices": [{"message": {"content": "{}"}}]}
    mock_response = _make_response(json_data)

    mock_post = AsyncMock(return_value=mock_response)

    mock_client = MagicMock()
    mock_client.post = mock_post
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("trading_harness.llm.client.httpx.AsyncClient", return_value=mock_client):
        llm = OpenAICompatibleClient(base_url="http://localhost:8000/v1", api_key="test-key")
        await llm.chat(
            model="local-main",
            messages=[{"role": "user", "content": "test"}],
        )

    payload = mock_post.call_args[1]["json"]
    assert payload["temperature"] == 0.2


@pytest.mark.asyncio
async def test_base_url_strips_trailing_slash():
    """Client strips trailing slash from base_url."""
    json_data = {"choices": [{"message": {"content": "{}"}}]}
    mock_response = _make_response(json_data)

    mock_post = AsyncMock(return_value=mock_response)

    mock_client = MagicMock()
    mock_client.post = mock_post
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("trading_harness.llm.client.httpx.AsyncClient", return_value=mock_client):
        llm = OpenAICompatibleClient(base_url="http://localhost:8000/v1/", api_key="test-key")
        await llm.chat(
            model="local-main",
            messages=[{"role": "user", "content": "test"}],
        )

    assert mock_post.call_args[0][0] == "http://localhost:8000/v1/chat/completions"


@pytest.mark.asyncio
async def test_chat_sends_api_key_header():
    """Client includes Bearer token in Authorization header."""
    json_data = {"choices": [{"message": {"content": "{}"}}]}
    mock_response = _make_response(json_data)

    mock_post = AsyncMock(return_value=mock_response)

    mock_client = MagicMock()
    mock_client.post = mock_post
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("trading_harness.llm.client.httpx.AsyncClient", return_value=mock_client):
        llm = OpenAICompatibleClient(base_url="http://localhost:8000/v1", api_key="secret-key-123")
        await llm.chat(
            model="local-main",
            messages=[{"role": "user", "content": "test"}],
        )

    headers = mock_post.call_args[1]["headers"]
    assert headers["Authorization"] == "Bearer secret-key-123"