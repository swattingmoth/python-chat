import os


def get_environment() -> str:
    deployment_hint = os.getenv("CHATBOT_ENV", "").strip().lower()
    return deployment_hint or "production"
