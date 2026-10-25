# ai-chat

## What

A command-line chatbot that holds a multi-turn conversation with an LLM. You pick the provider per run: Claude, Groq, or a local model through Ollama.

## Why

I'm learning AI engineering and wanted a project small enough to understand end to end. The goal was not a feature-rich product but a codebase where I can explain every decision: packaging, how chat APIs work, async I/O, and a proper CLI.

## How

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env        # then fill in the API keys you need
ai-chat chat --provider groq   # or: claude, ollama
```

Type `quit` to exit. Requires Python 3.10+; `--provider ollama` also needs [Ollama](https://ollama.com/download) running locally.

## Learned

**Week 01 — Python building blocks**
- Dataclasses give messages a fixed shape (`role`, `content`), so typos are caught by the editor instead of at runtime.

**Week 02 — Packaging**
- A venv, `pyproject.toml`, and a `src/` layout make the project installable and reproducible; `pip install -e .` keeps it editable.
- Verifying in a fresh venv catches the "works on my machine" trap.

**Week 03 — How chat LLMs work**
- Models are stateless: the whole conversation is re-sent on every request, so history must be trimmed to fit the context window.
- A system prompt shapes behaviour, and each API places it differently (Claude takes a separate `system=` parameter).

**Week 04 — Async and CLI**
- `httpx.AsyncClient` with `asyncio.run` for non-blocking calls; blocking SDK calls go through `asyncio.to_thread`.
- Typer turns functions into commands; failures should print a short message to stderr and exit 1, not a traceback.

**Week 05 — Providers and secrets**
- Groq and Ollama speak the OpenAI chat format, so one function serves both; Claude needs its own.
- API keys live in `.env` (git-ignored) with a committed `.env.example`; read config at startup, never at import time.
