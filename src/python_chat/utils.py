import os
from importlib.metadata import PackageNotFoundError, version


def get_environment() -> str:
    deployment_hint = os.getenv("CHATBOT_ENV", "").strip().lower()
    return deployment_hint or "production"


def get_project_version() -> str:
    """Return the installed package version, or ``"unknown"`` if unavailable."""
    try:
        return version("python-chat")
    except PackageNotFoundError:
        pass

    return "unknown"
