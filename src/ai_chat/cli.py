import asyncio

import httpx
import typer

from ai_chat.config import load_or_create_config
from ai_chat.core import ChatConfig, run_chat

app = typer.Typer(help="A simple command-line chatbot.")


@app.callback()
def callback():
    """Keeps `chat` a named subcommand even while it is the only one."""


@app.command()
def chat(
    model: str = typer.Option(ChatConfig.model, help="Ollama model to talk to."),
    system_prompt: str = typer.Option(
        None, help="Instruction that sets the bot's behaviour (default: config file)."
    ),
):
    """Start a chat session."""
    config = ChatConfig(
        model=model,
        system_prompt=system_prompt or load_or_create_config(),
    )
    try:
        asyncio.run(run_chat(config))
    except httpx.ConnectError:
        typer.echo(
            f"Could not connect to Ollama at {config.url}.\n"
            "Is it running? Start it with:  ollama serve",
            err=True,
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
