"""Tests for images.py module.

Heavy use of mocks/patches because generate_image performs real API calls,
UUID filename generation, and filesystem writes. generate_image_tool
additionally depends on the ModelContext singleton.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, Generator
from unittest.mock import MagicMock, patch

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

    ModelContext.create(fake_client, "/tmp/test-images")
    # ensure model starts as QUESTIONS but tool will switch
    yield
    ModelContext.reset()


def test_generate_image_calls_api_writes_file_and_returns_path_bytes() -> None:
    """Core happy path: builds prompt, calls client.image.sample, gets .image bytes, writes uuid.png, returns tuple."""
    fake_client = MagicMock()
    fake_client.image.sample.return_value = SimpleNamespace(
        image=b"\x89PNG...", cost_usd=0.05
    )

    with (
        patch("python_chat.images.uuid.uuid4") as mock_uuid,
        patch(
            "python_chat.images.ensure_local_image_copy",
            return_value="/tmp/test-images/fake.png",
        ) as mock_local_copy,
    ):
        mock_uuid.return_value = "test-uuid-1234"

        os.environ["CHATBOT_ENV"] = "development"
        path, cost = generate_image(
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

        mock_local_copy.assert_called_once_with(
            "/tmp/test-images", "test-uuid-1234.png", b"\x89PNG..."
        )

        assert path == "/tmp/test-images/fake.png"
        assert cost == 0.05


def test_generate_image_tool_uses_context_switches_model_and_returns_toolresult() -> (
    None
):
    """The image tool wrapper uses context and returns a ToolResult from generate_image output."""
    model_context = ModelContext.current()
    assert model_context.model_name == Models.QUESTIONS

    with patch(
        "python_chat.images.generate_image",
        return_value=("/tmp/test-images/generated.png", 0.0),
    ) as mock_generate_image:
        result = generate_image_tool("draw a tree", "tool-call-1")

    assert isinstance(result, ToolResult)
    assert result.content_for_model == "Generated image generated.png"
    assert result.content == "/tmp/test-images/generated.png"
    assert result.content_type == "image"
    assert result.tool_call_id == "tool-call-1"
    assert result.metadata == {"image_reference": "/tmp/test-images/generated.png"}

    mock_generate_image.assert_called_once_with(
        "draw a tree",
        Models.IMAGES,
        model_context.client,
        model_context.image_path,
    )
    assert ModelContext.current().model_name == Models.QUESTIONS


def test_generate_image_tool_returns_tool_cost() -> None:

    with patch(
        "python_chat.images.generate_image",
        return_value=("/tmp/test-images/generated.png", 0.05),
    ):
        result = generate_image_tool("draw a tree", "tool-call-1")

    assert result.cost == 0.05


def test_generate_image_tool_handles_missing_image_path() -> None:
    """When generate_image returns no path, the tool reports failure instead of a ToolResult with an image."""
    with patch(
        "python_chat.images.generate_image",
        return_value=(None, 0.02),
    ) as mock_generate_image:
        result = generate_image_tool("draw a tree", "tool-call-3")

    mock_generate_image.assert_called_once()
    assert isinstance(result, ToolResult)
    assert result.content_for_model == "Failed to generate image"
    assert result.content is None
    assert result.content_type == "text"
    assert result.tool_call_id == "tool-call-3"
    assert result.cost == 0.02
    assert result.metadata == {}


def test_generate_image_tool_propagates_context_errors() -> None:
    """If no context initialized, current() raises (tool does not swallow)."""
    ModelContext.reset()  # force uninitialized

    with pytest.raises(Exception, match="has not been initialized"):
        generate_image_tool("anything", "tool-call-2")


def test_generate_image_uploads_and_uses_public_url(monkeypatch: Any) -> None:
    fake_client = MagicMock()
    fake_client.image.sample.return_value = SimpleNamespace(image=b"png", cost_usd=0.0)

    upload_bytes = MagicMock()
    get_public_url = MagicMock(return_value="https://example.test/image.png")
    persistence_client = SimpleNamespace(
        upload_bytes=upload_bytes,
        get_public_url=get_public_url,
    )

    ModelContext.reset()
    ModelContext.create(MagicMock(), "c:/tmp", persistence_client=persistence_client)  # type: ignore[arg-type]

    monkeypatch.setattr(
        "python_chat.images.ensure_local_image_copy", lambda *_: "c:/tmp/local.png"
    )

    image_file, cost = generate_image(
        prompt="tree",
        model=Models.IMAGES,
        client=fake_client,
        image_path="c:/tmp",
        upload_to_storage=True,
    )

    assert image_file == "https://example.test/image.png"
    assert cost == 0.0
    upload_bytes.assert_called_once()
    get_public_url.assert_called_once()


def test_generate_image_sets_tool_cost(monkeypatch: Any) -> None:
    fake_client = MagicMock()
    fake_client.image.sample.return_value = SimpleNamespace(image=b"png", cost_usd=0.05)

    monkeypatch.setattr(
        "python_chat.images.ensure_local_image_copy",
        lambda *_: "c:/tmp/local.png",
    )

    _, estimated_cost = generate_image(
        prompt="tree",
        model=Models.IMAGES,
        client=fake_client,
        image_path="c:/tmp",
        upload_to_storage=False,
    )

    assert estimated_cost == 0.05
