# python-chat

A Python project skeleton for developing an LLM-powered chatbot.

**Status**: Foundational setup complete. The actual chatbot implementation is **NOT** included (per requirements). This provides a clean, typed, tested base ready for adding LLM features (e.g., using OpenAI, LangChain, Gradio UI, etc.).

## Features of the Skeleton

- **Modern Python packaging** with `uv` (fast, reliable) and `pyproject.toml`
- **src/ layout** for the `python_chat` package
- **Testing**: `pytest` with configuration
- **Type checking**: `mypy` in strict mode
- **Placeholder code**: `hello_world()` function + passing test
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
│       ├── hello.py          # Placeholder: hello_world()
│       └── py.typed
├── tests/
│   └── test_hello.py
├── .gitignore
├── pyproject.toml
├── README.md
└── uv.lock (after first sync)
```

## Getting Started

### Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) (v0.4+ recommended)

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

### Type Checking

```bash
uv run mypy src
# or simply
uv run mypy
```

Should report no errors with strict mode.

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

## Adding the LLM Chatbot Later

When ready to implement:

1. Add runtime dependencies: `uv add openai langchain-core gradio ...`
2. Create modules under `src/python_chat/` (e.g., `llm.py`, `chatbot.py`)
3. Add corresponding tests in `tests/`
4. Update `copilot-instructions.md` as needed
5. For Gradio web app (Supabase deploy): add a `examples/app.py` or `src/python_chat/app.py` that imports your chatbot logic and launches `gr.Interface(...)`. Use `uv run python -m python_chat.app` or add `[project.scripts]`.

**Never** put actual LLM keys or full chatbot code in the initial skeleton.

## Development Commands

| Command                    | Description                          |
|----------------------------|--------------------------------------|
| `uv sync --group dev`      | Install/update all deps + editable   |
| `uv run pytest`            | Run the test suite                   |
| `uv run mypy`              | Strict static type check             |
| `uv run ruff check .`      | (Future) Linting                     |
| `uv add <pkg>`             | Add runtime dependency               |
| `uv add --group dev <pkg>` | Add dev-only tool                    |
| `uv run python -m pytest --cov` | (Future) with coverage          |

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

*Generated as part of initial project scaffolding. Happy coding!*
