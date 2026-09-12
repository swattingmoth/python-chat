from __future__ import annotations

from importlib.metadata import PackageNotFoundError

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


def test_get_project_version_returns_unknown_when_package_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_package_not_found(_: str) -> str:
        raise PackageNotFoundError

    monkeypatch.setattr("python_chat.utils.version", _raise_package_not_found)

    assert get_project_version() == "unknown"
