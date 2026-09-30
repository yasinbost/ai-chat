import asyncio
from dataclasses import dataclass, asdict

import httpx

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
    """
    messages: a list of ChatMessage objects
    returns: the model's reply text (str)
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
    data = response.json()
    return data["message"]["content"]


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


async def main():
    chatbot = ChatConfig(system_prompt=load_or_create_config())
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


if __name__ == "__main__":
    asyncio.run(main())
