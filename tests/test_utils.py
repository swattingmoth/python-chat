from __future__ import annotations

from pathlib import Path

import pytest

from python_chat.utils import get_environment, get_project_version


def test_get_environment_defaults_to_production_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CHATBOT_ENV", raising=False)

    assert get_environment() == "production"


def test_get_environment_returns_lowercased_configured_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHATBOT_ENV", "Development")

    assert get_environment() == "development"


def test_get_project_version_reads_version_from_pyproject_toml() -> None:
    version = get_project_version()

    assert version != "unknown"
    assert version[0].isdigit()


def test_get_project_version_returns_unknown_when_pyproject_missing(
    tmp_path: Path,
) -> None:
    isolated_path = tmp_path / "python_chat" / "utils.py"

    assert get_project_version(start_path=isolated_path) == "unknown"
