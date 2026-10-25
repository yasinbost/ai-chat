"""Core chat logic for ai-chat: message types, provider requests and the chat loop.

Two providers are supported, each with its own request function:

* ``send_request`` talks to a local Ollama server over HTTP.
* ``send_claude_request`` talks to Claude through the Anthropic SDK.
* ``send_openai_compatible_request`` talks to any OpenAI-compatible API
  (Groq, Ollama's ``/v1`` endpoint).

``ask`` is a one-call helper that picks the provider by name.

The two differ in how they take the system prompt: Ollama expects it as a
``role="system"`` message inside ``messages``, Anthropic expects it in the
separate ``system=`` parameter.
"""

import asyncio
import os
from dataclasses import asdict, dataclass

import anthropic
import httpx
from dotenv import load_dotenv
from openai import OpenAI

from ai_chat.config import DEFAULT_SYSTEM_PROMPT, load_or_create_config


@dataclass
class ChatConfig:
    url: str = "http://localhost:11434/api/chat"
    model: str = "llama3.2"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_history: int = 10


@dataclass
class ChatMessage:
    role: str
    content: str


async def send_request(chatbot: ChatConfig, messages: list[ChatMessage]) -> str:
    """Send a conversation to a local Ollama server and return the reply text.

    The system prompt is not a separate argument here: Ollama reads it from a
    ``role="system"`` message at the start of ``messages``.

    Args:
        chatbot: Connection settings; ``url`` and ``model`` are used.
        messages: The conversation so far, oldest first, including the
            system message if one is wanted.

    Returns:
        The assistant's reply text.

    Raises:
        httpx.ConnectError: If the Ollama server is not reachable.
        httpx.ReadTimeout: If no reply arrives within 60 seconds.
        httpx.HTTPStatusError: If Ollama answers with an error status, for
            example 404 for a model that is not installed.
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            chatbot.url,
            json={
                "model": chatbot.model,
                "messages": [asdict(m) for m in messages],
                "stream": False,
            },
        )
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"]


def send_claude_request(messages: list[dict[str, str]], system_prompt: str) -> str:
    """Send a conversation to Claude and return the reply text.

    Unlike Ollama, Anthropic does not accept a ``role="system"`` message, so
    the system prompt is passed separately through ``system=``.

    Args:
        messages: The conversation so far, oldest first, as
            ``{"role": ..., "content": ...}`` dicts. Only ``"user"`` and
            ``"assistant"`` roles are allowed, and the first should be ``"user"``.
        system_prompt: Instruction that sets Claude's behaviour for the whole
            conversation.

    Returns:
        The text of Claude's reply.

    Raises:
        ValueError: If the ``ANTHROPIC_API_KEY`` environment variable is not set.
        anthropic.AuthenticationError: If the key is set but invalid.
        anthropic.BadRequestError: If ``messages`` is malformed, for example
            if it contains a ``"system"`` role.
        anthropic.APIConnectionError: If the Anthropic API cannot be reached.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY not set. Check your .env file.")
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    response = client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=1024,
        system=system_prompt,
        messages=messages,
    )
    return response.content[0].text


def send_openai_compatible_request(
    messages: list[dict[str, str]], base_url: str, api_key: str, model: str
) -> str:
    """Send a conversation to an OpenAI-compatible API and return the reply text.

    Groq and Ollama's ``/v1`` endpoint both speak the OpenAI chat format, so one
    function serves both; only ``base_url``, ``api_key`` and ``model`` differ.

    Args:
        messages: The conversation so far, oldest first, as
            ``{"role": ..., "content": ...}`` dicts. A ``"system"`` role is allowed.
        base_url: Root URL of the API, for example ``https://api.groq.com/openai/v1``.
        api_key: The provider's key. Ollama ignores it, but the SDK requires one.
        model: Model identifier as the provider names it.

    Returns:
        The text of the reply.

    Raises:
        openai.AuthenticationError: If the key is invalid.
        openai.NotFoundError: If the model does not exist at the provider.
        openai.APIConnectionError: If ``base_url`` cannot be reached.
    """
    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content


def ask(prompt: str, provider: str = "claude") -> str:
    """Send a prompt to an LLM provider and return the reply.

    Args:
        prompt: The user message to send.
        provider: One of "claude", "groq", "ollama". Defaults to "claude".

    Returns:
        The model's reply text, or an error message string on failure.
    """
    messages = [{"role": "user", "content": prompt}]
    try:
        if provider == "claude":
            return send_claude_request(
                messages, system_prompt="You are a helpful assistant."
            )
        elif provider == "groq":
            return send_openai_compatible_request(
                messages,
                base_url=os.getenv("GROQ_BASE_URL"),
                api_key=os.getenv("GROQ_API_KEY"),
                model=os.getenv("GROQ_MODEL_NAME"),
            )
        elif provider == "ollama":
            return send_openai_compatible_request(
                messages,
                base_url=os.getenv("OLLAMA_BASE_URL"),
                api_key="ollama",
                model=os.getenv("OLLAMA_MODEL_NAME"),
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")
    except Exception as e:  # noqa: BLE001 - ask() reports every failure as a string by design
        return f"{provider}: Error | {e}"


def build_messages(config: ChatConfig, history: list[ChatMessage]) -> list[dict[str, str]]:
    """
    config: carries system_prompt and max_history
    history: every ChatMessage so far
    returns: the dict list Ollama expects
    """
    system_message = ChatMessage(role="system", content=config.system_prompt)
    trimmed_history = history[-config.max_history:]
    combined = [system_message] + trimmed_history
    return [{"role": m.role, "content": m.content} for m in combined]


async def run_chat(chatbot: ChatConfig) -> None:
    """Runs the interactive chat loop until the user types 'quit'."""
    messages: list[ChatMessage] = [
        ChatMessage(role="system", content=chatbot.system_prompt)
    ]
    print("Chatbot ready. Type 'quit' to exit.")

    while True:
        user_input: str = input("You: ")
        if user_input.lower() == "quit":
            break

        messages.append(ChatMessage(role="user", content=user_input))
        reply: str = await send_request(chatbot, messages)
        print(f"Bot: {reply}")
        messages.append(ChatMessage(role="assistant", content=reply))


PROVIDERS = ("claude", "groq", "ollama")


def _require_env(*names: str) -> list[str]:
    """Return the values of the given environment variables, or raise if one is unset."""
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        raise ValueError(f"{', '.join(missing)} not set. Check your .env file.")
    return [os.environ[n] for n in names]


def _reply_for(provider: str, history: list[dict[str, str]], system_prompt: str) -> str:
    """Get one reply for ``provider`` (claude or groq) from the conversation so far."""
    if provider == "claude":
        return send_claude_request(history, system_prompt)
    base_url, api_key, model = _require_env(
        "GROQ_BASE_URL", "GROQ_API_KEY", "GROQ_MODEL_NAME"
    )
    messages = [{"role": "system", "content": system_prompt}] + history
    return send_openai_compatible_request(messages, base_url, api_key, model)


async def run_chat_loop(
    provider: str, system_prompt: str | None = None, model: str | None = None
) -> None:
    """Run the interactive chat loop against the chosen provider until 'quit'.

    Args:
        provider: One of ``PROVIDERS``.
        system_prompt: Instruction for the bot; defaults to the config file.
        model: Ollama model name. Ignored by claude and groq, whose models come
            from code and ``.env``.

    Raises:
        ValueError: If ``provider`` is unknown or a required variable is unset.
        anthropic.AuthenticationError: If the Claude key is invalid.
        openai.AuthenticationError: If the Groq key is invalid.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}")
    system_prompt = system_prompt or load_or_create_config()
    if provider == "ollama":
        await run_chat(ChatConfig(model=model or ChatConfig.model, system_prompt=system_prompt))
        return

    max_history = ChatConfig.max_history
    history: list[dict[str, str]] = []
    print(f"Chatbot ready ({provider}). Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        history.append({"role": "user", "content": user_input})
        trimmed = history[-max_history:]
        while trimmed and trimmed[0]["role"] != "user":  # APIs expect a user turn first
            trimmed = trimmed[1:]
        reply = await asyncio.to_thread(_reply_for, provider, trimmed, system_prompt)
        print(f"Bot: {reply}")
        history.append({"role": "assistant", "content": reply})


async def main():
    load_dotenv()
    await run_chat(ChatConfig(system_prompt=load_or_create_config()))


if __name__ == "__main__":
    asyncio.run(main())
