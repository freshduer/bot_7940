"""Tests for OpenAICompatProvider handling custom/direct endpoints."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from nanobot.providers.openai_compat_provider import OpenAICompatProvider


def test_custom_provider_parse_handles_empty_choices() -> None:
    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI"):
        provider = OpenAICompatProvider()
    response = SimpleNamespace(choices=[])

    result = provider._parse(response)

    assert result.finish_reason == "error"
    assert "empty choices" in result.content


def test_custom_provider_parse_accepts_plain_string_response() -> None:
    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI"):
        provider = OpenAICompatProvider()

    result = provider._parse("hello from backend")

    assert result.finish_reason == "stop"
    assert result.content == "hello from backend"


def test_custom_provider_parse_accepts_dict_response() -> None:
    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI"):
        provider = OpenAICompatProvider()

    result = provider._parse({
        "choices": [{
            "message": {"content": "hello from dict"},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
        },
    })

    assert result.finish_reason == "stop"
    assert result.content == "hello from dict"
    assert result.usage["total_tokens"] == 3


def test_custom_provider_parse_chunks_accepts_plain_text_chunks() -> None:
    result = OpenAICompatProvider._parse_chunks(["hello ", "world"])

    assert result.finish_reason == "stop"
    assert result.content == "hello world"


def test_api_key_auth_uses_header_and_default_query() -> None:
    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI") as mock_cls:
        OpenAICompatProvider(
            api_key="secret-key",
            api_base="https://example.com/api/v0/rest/deployments/gpt-5-mini",
            default_query={"api-version": "2024-12-01-preview"},
            auth_style="api_key",
        )

    mock_cls.assert_called_once()
    call_kw = mock_cls.call_args.kwargs
    assert call_kw["api_key"] == ""
    assert call_kw["default_query"] == {"api-version": "2024-12-01-preview"}
    headers = call_kw["default_headers"]
    assert headers["api-key"] == "secret-key"


def test_api_key_auth_requires_key() -> None:
    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI", MagicMock()):
        try:
            OpenAICompatProvider(api_key="", auth_style="api_key")
        except ValueError as e:
            assert "api_key" in str(e).lower()
        else:
            raise AssertionError("expected ValueError")
