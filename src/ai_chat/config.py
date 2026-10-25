"""Reads, and on first run creates, the system prompt file ``~/.ai-chat/config.txt``."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


@dataclass
class AppConfig:
    """Runtime configuration for ai-chat, read once at startup.

    Build it with ``AppConfig.from_env()`` after ``.env`` has been loaded, or
    construct it directly (for example in a test) without touching the environment.
    """

    default_provider: str = "claude"
    groq_base_url: str = ""
    groq_model: str = ""
    ollama_base_url: str = ""
    ollama_model: str = ""
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Build an AppConfig by reading the current environment."""
        return cls(
            groq_base_url=os.getenv("GROQ_BASE_URL", ""),
            groq_model=os.getenv("GROQ_MODEL_NAME", ""),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", ""),
            ollama_model=os.getenv("OLLAMA_MODEL_NAME", ""),
        )


def setup_logging(config: AppConfig) -> None:
    """Configure the root logger from an AppConfig.

    Raises:
        ValueError: If ``config.log_level`` is not a real logging level name.
            A typo fails loudly instead of silently falling back to a default.
    """
    level = getattr(logging, config.log_level.upper(), None)
    if not isinstance(level, int):  # also rejects names like "getLogger"
        # a bad config *value* (not a wrong argument type), so ValueError fits
        raise ValueError(  # noqa: TRY004
            f"Invalid log_level: '{config.log_level}'. "
            "Use DEBUG, INFO, WARNING, ERROR, or CRITICAL."
        )
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")
    # Third-party libraries log far more than we want to read at DEBUG.
    # httpx2/httpcore2 are the logger names newer httpx/httpcore releases use.
    for noisy in ("httpx", "httpx2", "httpcore", "httpcore2", "anthropic", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_config_path() -> Path:
    """
    returns: where the config file lives (~/.ai-chat/config.txt).
    Only builds the path; it does not create anything.
    """
    return Path.home() / ".ai-chat" / "config.txt"


def load_or_create_config() -> str:
    """
    Creates the config file with the default system prompt if it is missing.
    returns: the system prompt stored in the file (str)
    """
    path = get_config_path()

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(DEFAULT_SYSTEM_PROMPT)

    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()
