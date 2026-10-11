"""Ayni soruyu system prompt'suz ve system prompt'lu gonderip tonu karsilastirir.

Calistirmadan once anahtari ortama yukle:  set -a; source .env; set +a
"""

import anthropic

MODEL = "claude-sonnet-5-5"
QUESTION = "What is a variable?"
SYSTEM = "You are a grumpy teacher who gives short, blunt answers."

client = anthropic.Anthropic()  # ANTHROPIC_API_KEY'i ortamdan okur


def ask(system: str | None) -> str:
    kwargs = {"system": system} if system else {}
    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": QUESTION}],
        **kwargs,
    )
    return response.content[0].text


print("=== system prompt YOK ===")
print(ask(None))
print("\n=== system prompt VAR (grumpy teacher) ===")
print(ask(SYSTEM))
