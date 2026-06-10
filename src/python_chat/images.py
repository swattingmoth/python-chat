import base64
import os
import uuid

from xai_sdk import Client

from python_chat.api import Models
from python_chat.context import ModelContext
from python_chat.tools import ToolResult


def generate_image(
    prompt: str, model: str, client: Client, image_path: str
) -> tuple[str, bytes]:
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
        A tuple of (saved_file_path, raw_image_bytes).
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
    image_file = os.path.join(image_path, f"{uuid.uuid4()}.png")
    with open(image_file, "wb") as f:
        f.write(image_data)

    print(f"Generated image saved to {image_file}")

    return (image_file, image_data)


def generate_image_tool(prompt: str, tool_call_id: str) -> "ToolResult":
    """Tool function to generate an image."""
    model_context = ModelContext.current()
    with model_context.use_model(Models.IMAGES):
        _, image_data = generate_image(
            prompt,
            model_context.model_name,
            model_context.client,
            model_context.image_path,
        )
    # uncomment below to test with a static image instead of generating a new one each time.
    # with open(
    #     r"C:\Users\jordan-dev\model_output\images\5e6305fa-e54d-496b-b742-6f49ba5f1e45.png",
    #     "rb",
    # ) as f:
    #     image_data = f.read()

    return ToolResult(
        content_for_model="Generated an image.",
        content=image_data,
        content_type="image",
        tool_call_id=tool_call_id,
    )
