"""Tests for images.py module.

Heavy use of mocks/patches because generate_image performs real API calls,
UUID filename generation, and filesystem writes. generate_image_tool
additionally depends on the ModelContext singleton.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Generator
from unittest.mock import MagicMock, mock_open, patch

import pytest

from python_chat.api import Models
from python_chat.context import ModelContext
from python_chat.images import generate_image, generate_image_tool
from python_chat.tools import ToolResult


@pytest.fixture(autouse=True)
def reset_and_setup_context() -> Generator[None, None, None]:
    """Provide a fully mocked ModelContext for image tool tests."""
    ModelContext.reset()
    fake_client = MagicMock()
    # The image gen response shape expected by new xai code: .image -> bytes directly
    # (the autouse fixture client is not actually invoked for image.sample in tool tests
    # because they patch generate_image; this just satisfies ModelContext.create)
    fake_client.image.sample.return_value = SimpleNamespace(
        image=b"fake-png-bytes-here"
    )

    ctx = ModelContext.create(fake_client, "/tmp/test-images")
    # ensure model starts as QUESTIONS but tool will switch
    yield
    ModelContext.reset()


def test_generate_image_calls_api_writes_file_and_returns_path_bytes() -> None:
    """Core happy path: builds prompt, calls client.image.sample, gets .image bytes, writes uuid.png, returns tuple."""
    fake_client = MagicMock()
    fake_client.image.sample.return_value = SimpleNamespace(image=b"\x89PNG...")

    with (
        patch("python_chat.images.uuid.uuid4") as mock_uuid,
        patch(
            "python_chat.images.os.path.join", return_value="/tmp/test-images/fake.png"
        ) as mock_join,
        patch("builtins.open", mock_open()) as mock_file,
    ):
        mock_uuid.return_value = "test-uuid-1234"

        path, data = generate_image(
            prompt="a cat in hat",
            model=Models.IMAGES,
            client=fake_client,
            image_path="/tmp/test-images",
        )

        # API called with wrapped system prompt + user prompt
        call_kwargs = fake_client.image.sample.call_args.kwargs
        assert "Generate an image based on the request" in call_kwargs["prompt"]
        assert "a cat in hat" in call_kwargs["prompt"]
        assert call_kwargs["model"] == Models.IMAGES
        assert call_kwargs["image_format"] == "base64"

        mock_join.assert_called_once()
        mock_file.assert_called_once_with("/tmp/test-images/fake.png", "wb")
        mock_file().write.assert_called_once_with(b"\x89PNG...")

        assert path == "/tmp/test-images/fake.png"
        assert data == b"\x89PNG..."


def test_generate_image_tool_uses_context_switches_model_and_returns_toolresult() -> (
    None
):
    """The tool wrapper obtains context, temporarily uses IMAGES model, calls generate, wraps in ToolResult."""
    # The autouse fixture already created a context with a client that returns b64
    # We just need to ensure use_model was used and result shape is correct.

    with patch("python_chat.images.generate_image") as mock_gen:
        mock_gen.return_value = ("/tmp/f.png", b"imgdata")

        result = generate_image_tool("draw a tree")

        assert isinstance(result, ToolResult)
        assert result.content_for_model == "Generated an image."
        assert result.content == b"imgdata"
        assert result.content_type == "image"

        # Verify it switched model during call
        mock_gen.assert_called_once()
        args, _ = mock_gen.call_args
        # The call inside tool is positional: generate_image(prompt, model_context.model_name, client, path)
        # Inside the with use_model(IMAGES) the .model_name has been switched.
        assert args[1] == Models.IMAGES


def test_generate_image_tool_propagates_context_errors() -> None:
    """If no context initialized, current() raises (tool does not swallow)."""
    ModelContext.reset()  # force uninitialized

    with pytest.raises(Exception, match="has not been initialized"):
        generate_image_tool("anything")
