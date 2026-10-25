"""Pure-logic tests for ai_chat.core and ai_chat.config: no network, no real API keys."""

import asyncio

import pytest

from ai_chat import config
from ai_chat.core import (
    ChatConfig,
    ChatMessage,
    _require_env,
    ask,
    build_messages,
    run_chat_loop,
)


def test_build_messages_puts_system_prompt_first():
    cfg = ChatConfig(system_prompt="Be brief.")
    history = [ChatMessage(role="user", content="hi")]

    result = build_messages(cfg, history)

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


def test_ask_unknown_provider_returns_error_string_instead_of_raising():
    reply = ask("hello", provider="bogus-name")

    assert reply == "bogus-name: Error | Unknown provider: bogus-name"


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
