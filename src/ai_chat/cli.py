import asyncio

import anthropic
import httpx
import openai
import typer
from dotenv import load_dotenv

from ai_chat.core import PROVIDERS, ChatConfig, run_chat_loop

app = typer.Typer(help="A simple command-line chatbot.")


@app.callback()
def callback():
    """Keeps `chat` a named subcommand even while it is the only one."""


@app.command()
def chat(
    provider: str = typer.Option("claude", help="claude, groq, or ollama"),
    system_prompt: str = typer.Option(
        None, help="Instruction that sets the bot's behaviour (default: config file)."
    ),
    model: str = typer.Option(ChatConfig.model, help="Ollama model (ollama provider only)."),
):
    """Start a chat session."""
    load_dotenv()
    if provider not in PROVIDERS:
        raise typer.BadParameter(
            f"'{provider}' is not one of: {', '.join(PROVIDERS)}", param_hint="--provider"
        )
    try:
        asyncio.run(run_chat_loop(provider, system_prompt, model))
    except httpx.ConnectError:
        typer.echo(
            f"Could not connect to Ollama at {ChatConfig.url}.\n"
            "Is it running? Start it with:  ollama serve",
            err=True,
        )
        raise typer.Exit(code=1)
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        hint = (
            f"Is the model '{model}' installed? Try:  ollama pull {model}"
            if status == 404
            else "The server returned an error."
        )
        typer.echo(f"Ollama answered with HTTP {status}.\n{hint}", err=True)
        raise typer.Exit(code=1)
    except (ValueError, anthropic.APIError, openai.APIError) as e:
        typer.echo(f"{provider}: {e}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
