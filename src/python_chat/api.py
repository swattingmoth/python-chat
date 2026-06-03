import os

from dotenv import load_dotenv
from xai_sdk import Client


class Models:
    """Model identifiers for the xAI Grok API."""

    IMAGES = "grok-imagine-image"
    QUESTIONS = "grok-4-1-fast-reasoning"
    COMPLEX_QUESTIONS = "grok-4.3"


def init_api(api_key_name: str, api_url: str | None = None) -> Client:
    """Initialize the xAI Client.

    The xai-sdk Client reads XAI_API_KEY from the environment by default and
    connects to the production xAI endpoint. The api_url parameter is accepted
    for API compatibility with callers but is not used (the SDK manages the
    gRPC endpoint internally).

    Args:
        api_key_name (str): Environment variable name for the API key.
        api_url (str | None): Ignored; kept for backward compatibility.

    Returns:
        Client: Configured xAI SDK client instance.
    """
    load_dotenv()
    api_key = os.getenv(api_key_name)
    return Client(api_key=api_key)
