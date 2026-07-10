# python-chat

A production-grade LLM-powered chatbot package using the native xAI Python SDK (`xai-sdk`), with support for streaming chat, client-side and server-side tools, image generation, and a Gradio web UI (targeting Supabase hosting).

**Status**: Fully implemented, strictly typed (`mypy --strict`), comprehensively tested, and ready to run/deploy.

## Features

- **xAI SDK native integration** (not OpenAI compat layer): chat streaming, tool calling (client + server-side like `web_search`), image generation via `client.image.sample`
- **Simplified tool loop**: leverages xai-sdk's complete `tool_calls` on final `Response` (no manual delta accumulation)
- **Client-side tools**: register any Python callable; auto schema from signature + docstring (Google style)
- **Gradio UI**: mode switcher (Question / Complex Question / Generate Image), streaming chat, image display
- **Modern Python packaging** with `uv` (fast, reliable) and `pyproject.toml`
- **src/ layout** for the `python_chat` package (import as `from python_chat import ...`)
- **Testing**: `pytest` (every public API has tests)
- **Type checking**: `mypy --strict` (zero errors)
- **Hello example**: `hello_world()` + test (preserved)
- **Editor support**: VS Code settings for pytest + mypy
- **Copilot instructions**: `.github/copilot-instructions.md` for consistent AI assistance
- **Git ready**: `.gitignore` tailored for Python/uv/mypy

## Project Structure

```
python-chat/
├── .github/
│   └── copilot-instructions.md
├── .vscode/
│   └── settings.json
├── src/
│   └── python_chat/
│       ├── __init__.py
│       ├── api.py            # Client init (xai_sdk.Client)
│       ├── app.py            # Gradio UI wiring
│       ├── chat.py           # Core ChatInterface + streaming + tool loop
│       ├── chatbot.py        # Entrypoint (__main__): init + register + launch
│       ├── context.py        # ModelContext singleton (client + tools + model modes)
│       ├── hello.py          # hello_world() example
│       ├── images.py         # Image gen tool (uses xai image.sample)
│       ├── tools.py          # Tools registry + schema gen + execution + ToolResult
│       └── py.typed
├── tests/
│   ├── test_api.py
│   ├── test_chat.py
│   ├── test_context.py
│   ├── test_hello.py
│   ├── test_images.py
│   └── test_tools.py
├── .gitignore
├── pyproject.toml
├── README.md
└── uv.lock (after first sync)
```

## Getting Started

### Prerequisites

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/) (v0.4+ recommended)
- An xAI API key (set as `XAI_API_KEY` in `.env`)
- `IMAGE_FOLDER` env var (directory for generated images, e.g. `c:/temp/images`)

### Setup (one-time)

```bash
# Clone or cd into the project
cd python-chat

# Create virtualenv + install project + dev dependencies (pytest, mypy)
uv sync --group dev

# Activate the env (uv will suggest or use `uv run`)
# On Windows PowerShell:
. .venv\Scripts\Activate.ps1
```

This creates `.venv/` and `uv.lock`.

### Running Tests

```bash
# Using uv (recommended, no manual activate needed)
uv run pytest

# Or with activated venv
pytest

# Verbose / specific
uv run pytest -v tests/test_hello.py
```

All tests should pass ✅

### Type Checking and Linting

```bash
uv run mypy src
# or simply
uv run mypy
```
Should report no errors with strict mode.

```bash
uv run ruff check .
```
Should report no errors

### Running Code / Python Shell

```bash
uv run python
>>> from python_chat.hello import hello_world
>>> hello_world()
'Hello World'
```

Or for package:

```bash
uv run python -c "from python_chat import hello_world; print(hello_world())"
```

### Running the Chatbot (Gradio UI)

1. Create a `.env` file in the project root:

   ```
   XAI_API_KEY=your_xai_key_here
   IMAGE_FOLDER=c:/temp/images   # or any writable dir; will be created if needed
   SUPABASE_URL=https://<project-ref>.supabase.co
   SUPABASE_ANON_KEY=your_anon_key
   SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
   ```

2. Launch:

   ```bash
   uv run python -m python_chat.chatbot
   ```

   Or directly:

   ```bash
   uv run python src/python_chat/chatbot.py
   ```

The UI lets you choose modes:
- **Question**: basic chat (no extra tools)
- **Complex Question**: enables server-side tools like web search (xAI handles inside the request)
- **Generate Image**: registers the image gen tool; model can call it, result displays below chat

Images are saved as UUID-named PNGs under `IMAGE_FOLDER`.
If Supabase env vars are configured, image bytes are also uploaded to the `images`
storage bucket and detailed chat logs are batched to `chat-logs` as JSONL.

### Running the FastAPI Host (Auth-Protected Gradio)

Use this mode when deploying behind Supabase bearer-token auth (recommended for Hugging Face Docker Spaces).

1. Ensure the same `.env` values as above are set.
2. Start the API server:

    ```bash
    uv run uvicorn python_chat.server:app --host 0.0.0.0 --port 7860
    ```

3. Access the Gradio app via `/app` and provide an `Authorization: Bearer <token>` header.

Behavioral notes:
- `XAI_API_KEY` is resolved from Supabase Vault using service-role execution only.
- Chat session/message/tool persistence uses user-scoped execution (anon key + bearer token).
- If a user-scoped token is missing for chat persistence, writes fail closed (no automatic service-role escalation).

### Docker (Hugging Face Space)

Build and run locally:

```bash
docker build -t python-chat .
docker run --rm -p 7860:7860 \
   -e PORT=7860 \
   -e CHATBOT_ENV=production \
   -e SUPABASE_URL=https://<project-ref>.supabase.co \
   -e SUPABASE_ANON_KEY=<anon-key> \
   -e SUPABASE_SERVICE_ROLE_KEY=<service-role-key> \
   -e IMAGE_FOLDER=/tmp/images \
   python-chat
```

Health checks:
- `/healthz` returns process liveness.
- `/readyz` confirms model context initialization.

## Architecture & Extending

- **Core flow**: `ChatInterface.chat()` builds messages (via `_to_xai_message`), calls the injected `completer` (default uses `client.chat.create(...).stream()`), accumulates content, then inspects `last_response.tool_calls` (filtered by `get_tool_call_type == "client_side_tool"`), executes via `Tools.handle_tool_calls`, appends `tool` role entries to history for UI, and loops for follow-ups. Server tools (e.g. `web_search()`) are included in the initial `tools=` list and executed by xAI (no client loop for them).
- **Registering tools**: Use `ModelContext.current().register_tool(my_func, "Description of what it does")`. The 2nd positional is the description (for call-site ergonomics); use `name=...` kwarg to override the tool name exposed to the model. Schemas are derived via `inspect.signature` + docstring parsing.
- **Image tool**: Special-cased in `chat()` (yields `(history, image_bytes)` for Gradio); registered dynamically from the dropdown in `app.py`.
- **History**: Maintained as `list[dict[str, Any]]` (role/content/tool_call_id/tool_calls) to satisfy both Gradio Chatbot and test assertions. Internal xai messages are (re)built each turn.
- **Testing**: All public functions/classes have tests. Use dependency injection of `completer` / `tool_handler` in `ChatInterface` and `ModelContext` to avoid real API calls. Fakes in `test_chat.py` yield `(response, chunk)` pairs to match the `ChatCompleter` protocol.
- **Type safety**: `mypy --strict` enforced. Third-party `xai_sdk` lacks types, so `pyproject.toml` has `[[tool.mypy.overrides]]` ignore for it.
- To add new tools or modes: implement in a module, register in `chatbot.py` or `app.py`, add test mirroring `src/`, ensure `uv run pytest -q && uv run mypy src` pass.

### TODO

- Chat session is not updated to compute and store title
- Session end detection/cleanup — Add TTL sweeper for stale sessions
- Documentation updates — Update README architecture section

See `.github/copilot-instructions.md` for full development rules.

## Development Commands

| Command                          | Description                                      |
|----------------------------------|--------------------------------------------------|
| `uv sync --group dev`            | Install/update all deps + editable package       |
| `uv run pytest -q`               | Run the test suite (quiet)                       |
| `uv run mypy src`                | Strict static type check (must be clean)         |
| `uv run pytest && uv run mypy src` | Verify before commit (per copilot-instructions) |
| `uv add <pkg>`                   | Add runtime dependency                           |
| `uv add --group dev <pkg>`       | Add dev-only tool (pytest, mypy, etc.)           |
| `uv run python -m python_chat.chatbot` | Launch the Gradio chatbot UI              |
| `uv run uvicorn python_chat.server:app --host 0.0.0.0 --port 7860` | Launch the auth-protected FastAPI host |

## VS Code Integration

The `.vscode/settings.json` configures:

- Pytest as default test runner
- Mypy as type checker
- Python interpreter from `.venv`
- Format on save, etc.

Open the folder in VS Code and select the interpreter from `.venv\Scripts\python.exe` if prompted.

## License

This project is unlicensed (or add your license here). For personal/educational use in building LLM chatbots.

---

*Converted to native xai-sdk + full chatbot implementation. See copilot-instructions.md for contribution rules.*
