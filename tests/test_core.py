"""Pure-logic tests for ai_chat.core and ai_chat.config: no network, no real API keys."""

import asyncio
from unittest.mock import patch

import anthropic
import httpx
import pytest

from ai_chat import config
from ai_chat.config import AppConfig
from ai_chat.core import (
    ChatConfig,
    ChatMessage,
    _require_env,
    ask,
    build_messages,
    run_chat_loop,
)


@pytest.fixture
def sample_messages():
    """A minimal one-turn conversation: a single user message."""
    return [ChatMessage(role="user", content="hi")]


@pytest.fixture
def test_config():
    """An AppConfig with obviously fake, non-default values.

    If any of these leak into output or a request, the test shows it at once;
    ``.invalid`` hosts never resolve, so a stray network call goes nowhere.
    """
    return AppConfig(
        default_provider="fake-provider",
        claude_model="fake-claude-model",
        groq_base_url="http://groq.invalid/v1",
        groq_model="fake-groq-model",
        ollama_base_url="http://ollama.invalid/v1",
        ollama_model="fake-ollama-model",
        log_level="WARNING",
    )


def test_build_messages_puts_system_prompt_first(sample_messages):
    cfg = ChatConfig(system_prompt="Be brief.")

    result = build_messages(cfg, sample_messages)

    assert result == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "hi"},
    ]


def test_build_messages_trims_history_but_keeps_system_prompt():
    cfg = ChatConfig(system_prompt="Be brief.", max_history=2)
    history = [ChatMessage(role="user", content=f"m{i}") for i in range(5)]

    result = build_messages(cfg, history)

    assert [m["content"] for m in result] == ["Be brief.", "m3", "m4"]
    assert result[0]["role"] == "system"


def test_ask_unknown_provider_returns_error_string_instead_of_raising(test_config):
    reply = ask("hello", test_config, provider="bogus-name")

    assert reply == "bogus-name: Error | Unknown provider: bogus-name"


# --- ask() provider branches, mocked ------------------------------------------
# Patch the SDK class where core.py looks it up ("ai_chat.core.<name>"), not
# where it is defined: core.py already holds its own reference to it.
# ask() turns every exception into an error string, so a mis-wired mock shows
# up as "<provider>: Error | ..." and fails the equality assert below.


@patch("ai_chat.core.anthropic.Anthropic")
def test_ask_returns_claude_reply(mock_anthropic, test_config, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")  # passes the guard; never sent
    mock_client = mock_anthropic.return_value
    mock_client.messages.create.return_value.content[0].text = "Hello from Claude"

    reply = ask("hi", test_config, provider="claude")

    assert reply == "Hello from Claude"
    mock_client.messages.create.assert_called_once()
    assert mock_client.messages.create.call_args.kwargs["model"] == test_config.claude_model


@patch("ai_chat.core.OpenAI")
def test_ask_returns_groq_reply(mock_openai, test_config):
    mock_client = mock_openai.return_value
    mock_client.chat.completions.create.return_value.choices[0].message.content = "Hello from Groq"

    reply = ask("hi", test_config, provider="groq")

    assert reply == "Hello from Groq"


@patch("ai_chat.core.OpenAI")
def test_ask_returns_ollama_reply(mock_openai, test_config):
    mock_client = mock_openai.return_value
    mock_client.chat.completions.create.return_value.choices[0].message.content = "Hello from Ollama"

    reply = ask("hi", test_config, provider="ollama")

    assert reply == "Hello from Ollama"


@patch("ai_chat.core.OpenAI")
def test_ask_groq_sends_model_and_base_url_from_config(mock_openai, test_config, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-groq-key")
    mock_client = mock_openai.return_value
    mock_client.chat.completions.create.return_value.choices[0].message.content = "ok"

    ask("hi", test_config, provider="groq")

    mock_openai.assert_called_once_with(
        base_url=test_config.groq_base_url, api_key="fake-groq-key"
    )
    mock_client.chat.completions.create.assert_called_once_with(
        model=test_config.groq_model,
        messages=[{"role": "user", "content": "hi"}],
    )


# --- ask() error paths ----------------------------------------------------------
# ask() has a single `except Exception` that formats every failure as
# "<provider>: Error | <str(exception)>", so the exact string is the contract.


@patch("ai_chat.core.anthropic.Anthropic")
def test_ask_handles_claude_api_error(mock_anthropic, test_config, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    # A real SDK error object: APIStatusError subclasses need the HTTP response
    # they came from; str(error) is the message we pass in.
    response = httpx.Response(
        401, request=httpx.Request("POST", "https://api.anthropic.invalid/v1/messages")
    )
    mock_anthropic.return_value.messages.create.side_effect = anthropic.AuthenticationError(
        "invalid x-api-key", response=response, body=None
    )

    reply = ask("hi", test_config, provider="claude")

    assert reply == "claude: Error | invalid x-api-key"


@patch("ai_chat.core.OpenAI")
def test_ask_handles_connection_failure(mock_openai, test_config):
    mock_openai.return_value.chat.completions.create.side_effect = ConnectionError(
        "Connection refused"
    )

    reply = ask("hi", test_config, provider="ollama")

    assert reply == "ollama: Error | Connection refused"


def test_run_chat_loop_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider: bogus-name"):
        asyncio.run(run_chat_loop("bogus-name"))


def test_require_env_returns_values_in_order(monkeypatch):
    monkeypatch.setenv("AI_CHAT_TEST_A", "one")
    monkeypatch.setenv("AI_CHAT_TEST_B", "two")

    assert _require_env("AI_CHAT_TEST_B", "AI_CHAT_TEST_A") == ["two", "one"]


def test_require_env_names_every_missing_variable(monkeypatch):
    monkeypatch.delenv("AI_CHAT_TEST_X", raising=False)
    monkeypatch.delenv("AI_CHAT_TEST_Y", raising=False)

    with pytest.raises(ValueError, match="AI_CHAT_TEST_X, AI_CHAT_TEST_Y not set"):
        _require_env("AI_CHAT_TEST_X", "AI_CHAT_TEST_Y")


def test_load_or_create_config_creates_default_file_in_home(monkeypatch, tmp_path):
    monkeypatch.setattr(config.Path, "home", lambda: tmp_path)

    prompt = config.load_or_create_config()

    assert prompt == config.DEFAULT_SYSTEM_PROMPT
    assert (tmp_path / ".ai-chat" / "config.txt").exists()


def test_setup_logging_rejects_a_typo_in_log_level():
    with pytest.raises(ValueError, match="Invalid log_level: 'DEBG'"):
        config.setup_logging(AppConfig(log_level="DEBG"))


def test_setup_logging_rejects_names_that_are_not_levels():
    with pytest.raises(ValueError, match="Invalid log_level"):
        config.setup_logging(AppConfig(log_level="getLogger"))


def test_app_config_from_env_reads_the_four_provider_variables(monkeypatch):
    monkeypatch.setenv("GROQ_BASE_URL", "http://groq.test/v1")
    monkeypatch.setenv("GROQ_MODEL_NAME", "groq-model")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.test/v1")
    monkeypatch.setenv("OLLAMA_MODEL_NAME", "ollama-model")

    cfg = AppConfig.from_env()

    assert (cfg.groq_base_url, cfg.groq_model) == ("http://groq.test/v1", "groq-model")
    assert (cfg.ollama_base_url, cfg.ollama_model) == ("http://ollama.test/v1", "ollama-model")
