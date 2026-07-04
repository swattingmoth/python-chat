"""Tests for api.py module.

Covers the Models constants and init_api initialization (with full mocking
of environment and xAI Client construction).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from python_chat.api import Models, init_api


def test_models_constants_have_expected_values() -> None:
    """The Models class exposes the three model identifiers used by the app."""
    assert Models.IMAGES == "grok-imagine-image"
    assert Models.QUESTIONS == "grok-4-1-fast-reasoning"
    assert Models.COMPLEX_QUESTIONS == "grok-4.3"


@patch("python_chat.api.Client")
@patch("python_chat.api.os.getenv")
@patch("python_chat.api.load_dotenv")
def test_init_api_loads_env_and_constructs_xai_client(
    mock_load: MagicMock, mock_getenv: MagicMock, mock_client: MagicMock
) -> None:
    """init_api calls load_dotenv, reads the named env var, and returns configured xAI Client."""
    mock_getenv.return_value = "sk-test-123"
    fake_client = MagicMock()
    mock_client.return_value = fake_client

    result = init_api("XAI_API_KEY", "https://api.x.ai/v1")

    mock_load.assert_called_once_with()
    mock_getenv.assert_called_once_with("XAI_API_KEY")
    mock_client.assert_called_once_with(api_key="sk-test-123")
    assert result is fake_client


@patch("python_chat.api.Client")
@patch("python_chat.api.os.getenv")
@patch("python_chat.api.load_dotenv")
def test_init_api_allows_none_api_key_for_testing(
    mock_load: MagicMock, mock_getenv: MagicMock, mock_client: MagicMock
) -> None:
    """Handles missing key gracefully (xAI Client accepts None and falls back to env)."""
    mock_getenv.return_value = None
    fake_client = MagicMock()
    mock_client.return_value = fake_client

    result = init_api("MISSING_KEY", "https://example.com")

    mock_client.assert_called_once_with(api_key=None)
    assert result is fake_client
