import logging
import os
import uuid
from datetime import datetime, timezone

from xai_sdk import Client

from python_chat.api import Models
from python_chat.context import ModelContext
from python_chat.persistence import ensure_local_image_copy
from python_chat.tools import ToolResult
from python_chat.utils import get_environment

logger = logging.getLogger(__name__)


def generate_image(
    prompt: str,
    model: str,
    client: Client,
    image_path: str,
    *,
    upload_to_storage: bool = True,
) -> tuple[str | None, bytes, float]:
    """Generate an image using the xAI image API and save it locally.

    A safety system prompt is prepended to the user prompt to guide generation.
    The image is requested in base64 format for direct byte access without
    additional network fetches. The result is saved as a .png file using a UUID
    filename in the provided image_path directory.

    Args:
        prompt: The user's image generation request.
        model: The image model identifier (e.g. Models.IMAGES).
        client: The xAI SDK Client instance.
        image_path: Directory where the generated image file will be written.

    Returns:
        A tuple of (saved_file_path, raw_image_bytes, image_cost).
    """
    image_system_prompt = (
        "Generate an image based on the request below. The generated image must not "
        "include any nudity, suggestive content, or graphic violence. If requested, "
        "acts of affection (e.g. hugging, kissing) and display of weapons (e.g swords, "
        "guns, knives) is acceptable. If the requested picture does not meet the "
        "guidelines generate a picture of a peaceful landscape instead. Always follow "
        "the guidelines and never generate content that violates them.\n\nRequest:"
    )
    image_response = client.image.sample(
        prompt=f"{image_system_prompt}\n\n{prompt}",
        model=model,
        image_format="base64",
    )
    image_data: bytes = image_response.image
    image_name = f"{uuid.uuid4()}.png"
    image_cost = image_response.cost_usd or 0.0
    image_file = None
    if get_environment() == "development":
        image_file = ensure_local_image_copy(image_path, image_name, image_data)
        logger.info(f"Generated image saved to {image_file}")

    model_context = ModelContext.current()
    if upload_to_storage and model_context.persistence_client:
        ymd = datetime.now(timezone.utc).strftime("%Y/%m/%d")
        remote_path = f"{ymd}/{image_name}"
        try:
            model_context.persistence_client.upload_bytes(
                bucket="images",
                path=remote_path,
                data=image_data,
                content_type="image/png",
            )
            public_url = model_context.persistence_client.get_public_url(
                bucket="images",
                path=remote_path,
            )
            if public_url:
                image_file = public_url
            else:
                image_file = remote_path

        except Exception as exc:
            logger.warning("Image upload to storage failed: %s", exc)

    if not image_file:
        logger.warning(
            "Image file path is None. Image was not saved locally or uploaded to storage."
        )

    logger.info(f"Generated image file: {image_file}, cost: {image_cost}")

    return (image_file, image_cost)


def generate_image_tool(prompt: str, tool_call_id: str) -> "ToolResult":
    """Tool function to generate an image."""
    model_context = ModelContext.current()
    image_file, image_data, image_cost = generate_image(
        prompt,
        Models.IMAGES,
        model_context.client,
        model_context.image_path,
    )
    # uncomment below to test with a static image instead of generating a new one each time.
    # with open(
    #     r"C:\Users\jordan-dev\model_output\images\5e6305fa-e54d-496b-b742-6f49ba5f1e45.png",
    #     "rb",
    # ) as f:
    #     image_data = f.read()

    content_for_model = "Generated image"
    if image_file:
        content_for_model += f" {os.path.basename(image_file)}"

    return ToolResult(
        content_for_model=content_for_model,
        content=image_data,
        content_type="image",
        tool_call_id=tool_call_id,
        cost=image_cost,
        metadata={"image_reference": image_file},
    )
