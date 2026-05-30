import os

from dotenv import load_dotenv
from openai import OpenAI


class Models:
    IMAGES = "grok-imagine-image"
    QUESTIONS = "grok-4-1-fast-reasoning"
    COMPLEX_QUESTIONS = "grok-4.3"


def init_api(api_key_name: str, api_url: str) -> OpenAI:
    """Initialize the OpenAI client using environment configuration.

    Args:
        api_key_name (str): Environment variable name for the API key.
        api_url (str): Base URL for the OpenAI-compatible API.

    Returns:
        OpenAI: Configured OpenAI client instance.
    """
    load_dotenv()
    api_key = os.getenv(api_key_name)
    return OpenAI(api_key=api_key, base_url=api_url)
