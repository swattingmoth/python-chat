import os
from importlib.metadata import PackageNotFoundError, version


def get_environment() -> str:
    deployment_hint = os.getenv("CHATBOT_ENV", "").strip().lower()
    return deployment_hint or "production"


def get_project_version() -> str:
    """Read the package version from pyproject.toml, searching upward from start_path."""
    try:
        return version("python-chat")
    except PackageNotFoundError:
        pass

    return "unknown"
