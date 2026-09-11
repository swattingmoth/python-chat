import os
import tomllib
from pathlib import Path


def get_environment() -> str:
    deployment_hint = os.getenv("CHATBOT_ENV", "").strip().lower()
    return deployment_hint or "production"


def get_project_version(start_path: Path | None = None) -> str:
    """Read the package version from pyproject.toml, searching upward from start_path."""
    search_root = (start_path or Path(__file__)).resolve()
    for parent in search_root.parents:
        pyproject_path = parent / "pyproject.toml"
        if pyproject_path.is_file():
            with pyproject_path.open("rb") as pyproject_file:
                data = tomllib.load(pyproject_file)
            version = data.get("project", {}).get("version")
            if isinstance(version, str) and version:
                return version
            break

    return "unknown"
