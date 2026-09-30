import typer

from ai_chat.config import DEFAULT_SYSTEM_PROMPT
from ai_chat.core import ChatConfig

app = typer.Typer(help="A simple command-line chatbot.")


@app.callback()
def callback():
    """Keeps `chat` a named subcommand even while it is the only one."""


@app.command()
def chat(
    model: str = typer.Option(ChatConfig.model, help="Ollama model to talk to."),
    system_prompt: str = typer.Option(
        DEFAULT_SYSTEM_PROMPT, help="Instruction that sets the bot's behaviour."
    ),
):
    """Start a chat session."""
    print(f"model: {model}")
    print(f"system_prompt: {system_prompt}")


if __name__ == "__main__":
    app()
