FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CHATBOT_ENV=production
ENV PORT=7860

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install --no-cache-dir uv

# Copy configuration files first to utilize Docker build cache layer
COPY pyproject.toml uv.lock README.md ./

# Copy the source tree before syncing so the project itself can be installed into .venv
COPY src ./src

# Sync dependencies without dev tools; this installs the project into the virtual environment
RUN uv sync --no-dev

# Explicitly add the uv virtual environment binaries to the system PATH
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 7860

CMD ["python", "-m", "python_chat.server"]
