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
    assert Models.COMPLEX_QUESTIONS == "grok-4.5"


@patch("python_chat.api.Client")
@patch("python_chat.api.load_env_file")
def test_init_api_loads_env_and_constructs_xai_client(
    mock_load_env: MagicMock,
    mock_client: MagicMock,
) -> None:
    """init_api calls env loader and returns configured xAI Client."""
    fake_client = MagicMock()
    mock_client.return_value = fake_client

    result = init_api("sk-test-123", "https://api.x.ai/v1")

    mock_load_env.assert_called_once_with()
    mock_client.assert_called_once_with(api_key="sk-test-123")
    assert result is fake_client
