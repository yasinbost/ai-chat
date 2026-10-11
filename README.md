# ai-chat

A small command-line chatbot that talks to an LLM, built as an installable Python package. Pick the provider per run: [Claude](https://www.anthropic.com) (default), [Groq](https://groq.com), or a local model through [Ollama](https://ollama.com).

## Install

Requirements: Python 3.10+. Then, from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

`-e` installs the package in editable mode, so code changes take effect without reinstalling.

### API keys

Copy the example file and fill in the keys for the providers you want to use:

```bash
cp .env.example .env
```

| Variable | Used by | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | `claude` | Key from console.anthropic.com |
| `GROQ_API_KEY` | `groq` | Free key from console.groq.com |
| `GROQ_BASE_URL` | `groq` | `https://api.groq.com/openai/v1` |
| `GROQ_MODEL_NAME` | `groq` | A model name from your Groq console |
| `OLLAMA_BASE_URL`, `OLLAMA_MODEL_NAME` | `ask()` helper only | Not needed for `chat --provider ollama` |

`.env` is listed in `.gitignore`; only `.env.example` (placeholders) is committed. The file is loaded when the command starts, and variables already set in your shell take priority.

### Ollama (optional)

Only needed for `--provider ollama`. Install [Ollama](https://ollama.com/download) and pull a model:

```bash
ollama pull llama3.2
```

## Usage

```bash
ai-chat --help
```

```
 Usage: ai-chat [OPTIONS] COMMAND [ARGS]...

 A simple command-line chatbot.

╭─ Commands ───────────────────────────────────────────────────────────────────╮
│ chat  Start a chat session.                                                  │
╰──────────────────────────────────────────────────────────────────────────────╯
```

Start a chat (type `quit` to exit):

```bash
ai-chat chat --provider ollama --system-prompt "You are a concise assistant."
```

```
Chatbot ready. Type 'quit' to exit.
You: What is the capital of France? Answer in one short sentence.
Bot: The capital of France is Paris.
You: And its population, roughly?
Bot: The population of Paris is around 2.1 million within the city limits, but the larger metropolitan area has a population of over 12 million.
You: quit
```

The second question only makes sense because the full conversation is sent with every request; the model itself keeps no memory.

The same command with the other providers:

```bash
ai-chat chat                     # claude (default)
ai-chat chat --provider groq
```

### Options

| Option | Default | Description |
|---|---|---|
| `--provider` | `claude` | `claude`, `groq`, or `ollama` |
| `--system-prompt` | contents of the config file | Instruction that sets the bot's behaviour |
| `--model` | `llama3.2` | Ollama model; ignored by `claude` and `groq` |

The Claude model is set in `core.py` (`claude-haiku-5-5`); the Groq model comes from `GROQ_MODEL_NAME`.

### Configuration

If `--system-prompt` is omitted, the system prompt is read from `~/.ai-chat/config.txt`. The file is created with a default prompt on first run, so you can edit it without touching the code.

### Errors

Failures print a short message to stderr and exit with code 1 instead of a traceback.

An unknown provider:

```
Invalid value for --provider: 'bogus-name' is not one of: claude, groq, ollama
```

A missing or invalid key, for example:

```
groq: Error code: 401 - {'error': {'message': 'Invalid API Key', ...}}
```

An unset variable is reported as `<NAME> not set. Check your .env file.` A placeholder such as `your-key-here` counts as set, so the provider itself rejects it with a 401.

Ollama problems:

```
Could not connect to Ollama at http://localhost:11434/api/chat.
Is it running? Start it with:  ollama serve
```

```
Ollama answered with HTTP 404.
Is the model 'yok-boyle-model' installed? Try:  ollama pull yok-boyle-model
```

## How it works

```
src/ai_chat/
├── core.py     # message types, one request function per provider, chat loops, ask()
├── config.py   # loads or creates ~/.ai-chat/config.txt
└── cli.py      # Typer command line interface
scripts/
└── system_prompt_compare.py   # same question with and without a system prompt
```

- Ollama is called on its native `/api/chat` endpoint with `httpx.AsyncClient`.
- Claude is called through the `anthropic` SDK. Its API does not accept a `system` role inside `messages`, so the system prompt goes in the separate `system=` parameter.
- Groq (and Ollama's `/v1` endpoint) speak the OpenAI chat format, so one function using the `openai` SDK serves both.
- The Claude and Groq SDK calls are blocking; the chat loop runs them with `asyncio.to_thread` so the event loop is not blocked.
- `ask(prompt, provider)` is a one-shot helper for scripts. It returns errors as `"<provider>: Error | ..."` strings instead of raising.
- Messages are typed with dataclasses (`system`, `user`, `assistant` roles).
- Configuration and `.env` are read at startup, never at import time, so importing the package has no side effects.
- The package uses a `src/` layout and is declared in `pyproject.toml`, with dependencies pinned in `requirements.txt`.

## Why this exists

I built this while learning modern Python and AI Engineering fundamentals, one small step at a time: virtual environments and packaging, how LLM chat APIs work (roles, stateless requests, context limits), async I/O, and building a proper CLI. The goal was not a feature-rich product but a codebase where I can explain every decision.

## Status and next steps

This is a learning project and still in progress. Known gaps:

- Conversation history is trimmed to `max_history` for `claude` and `groq`, but not yet for `ollama`, so very long Ollama chats will grow the request size.
- Timeouts and other unexpected failures still raise a traceback.
- The Claude model name is hard-coded.
- The `claude` and `groq` paths have only been run against placeholder keys so far; the error handling is verified, a real reply is not.
- No automated tests yet.
- Responses are returned all at once rather than streamed.
