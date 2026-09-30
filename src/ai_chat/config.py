from pathlib import Path

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


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
