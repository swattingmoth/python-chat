FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml README.md ./
COPY src ./src

RUN uv sync --no-dev

ENV CHATBOT_ENV=production
ENV PORT=7860

EXPOSE 7860

CMD ["sh", "-c", "uv run uvicorn python_chat.server:app --host 0.0.0.0 --port ${PORT:-7860}"]
