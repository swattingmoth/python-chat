import os
import uuid

from openai import OpenAI, omit
import base64

from python_chat.api import Models
from python_chat.context import ModelContext
from python_chat.tools import ToolResult


def generate_image(
    prompt: str, model: str, client: OpenAI, image_path: str
) -> tuple[str, bytes]:
    """Example function to generate an image using the API."""
    image_system_prompt = "Generate an image based on the request below. The generaged image must not include any nudity, suggestive content, or graphic violence. If requested, acts of affection (e.g. hugging, kissing) and display of weapons (e.g swords, guns, knives) is acceptable. If the requested picture does not meet the guidelines generate a picture of a peaceful landscape instead. Always follow the guidelines and never generate content that violates them.\n\nRequest:"
    image_response = client.images.generate(
        model=model,
        prompt=f"{image_system_prompt}\n\n{prompt}",
        size=omit,
        n=1,
        response_format="b64_json",
    )
    image_base64 = image_response.data[0].b64_json  # type: ignore[index]
    image_data = base64.b64decode(image_base64)  # type: ignore[arg-type]
    image_file = os.path.join(image_path, f"{uuid.uuid4()}.png")
    with open(image_file, "wb") as f:
        f.write(image_data)

    print(f"Generated image saved to {image_file}")

    return (image_file, image_data)


def generate_image_tool(prompt: str) -> "ToolResult":
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
    # with open(r"C:\Users\jordan-dev\model_output\images\5e6305fa-e54d-496b-b742-6f49ba5f1e45.png", "rb") as f:
    #     image_data = f.read()

    return ToolResult(
        content_for_model="Generated an image.",
        content=image_data,
        content_type="image",
    )
