from dataclasses import dataclass, asdict

import requests


@dataclass
class ChatConfig:
    url: str = "http://localhost:11434/api/chat"
    model: str = "llama3.2"
    system_prompt: str = "You are a helpful assistant."


@dataclass
class ChatMessage:
    role: str
    content: str


def send_request(chatbot: ChatConfig, messages: list[ChatMessage]) -> str:
    """
    messages: a list of ChatMessage objects
    returns: the model's reply text (str)
    """
    response = requests.post(
        chatbot.url,
        json={
            "model": chatbot.model,
            "messages": [asdict(m) for m in messages],
            "stream": False,
        },
    )
    data = response.json()
    return data["message"]["content"]


def main():
    chatbot = ChatConfig()
    messages: list[ChatMessage] = [
        ChatMessage(role="system", content=chatbot.system_prompt)
    ]
    print("Chatbot ready. Type 'quit' to exit.")

    while True:
        user_input: str = input("You: ")
        if user_input.lower() == "quit":
            break

        messages.append(ChatMessage(role="user", content=user_input))
        reply: str = send_request(chatbot, messages)
        print(f"Bot: {reply}")
        messages.append(ChatMessage(role="assistant", content=reply))


if __name__ == "__main__":
    main()