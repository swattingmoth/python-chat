# GitHub Copilot Instructions for python-chat

This file provides persistent guidance for GitHub Copilot (and other AI coding assistants) when working in this repository.

## Project Overview
- **Purpose**: Production-grade LLM chatbot package (`python-chat`) using native xai-sdk (chat + tools + images), Gradio UI, strict typing and tests. (Originally started as skeleton.)
- **Deployment target**: Gradio-based web application, to be hosted on Supabase
- **Package name**: `python_chat` (importable as `from python_chat import ...`)
- **Python**: >= 3.13
- **Tooling**: `uv` (primary), `pytest`, `mypy` (strict)

## Core Rules (Always Follow)
1. **Type Safety First**
   - All new code **must** have complete type hints.
   - `mypy --strict` (or `uv run mypy`) must pass with zero errors before committing.
   - Use `-> None`, `list[str]`, `dict[str, Any]`, etc. Avoid `Any` unless absolutely necessary.

2. **Testing Discipline**
   - Every public function / class must have at least one pytest test.
   - Tests live in `tests/` mirroring `src/python_chat/`.
   - Use `pytest` 
   - Run `uv run pytest` and ensure green before suggesting changes.
   - Prefer descriptive test names: `test_xxx_does_yyy_when_zzz`.

4. **Project Structure Discipline**
   - Source code **always** under `src/python_chat/`.
   - Never put Python source in root, `app.py` at root, or `chatbot/` at root.
   - New modules: `src/python_chat/<feature>.py`
   - Tests: `tests/test_<feature>.py`

5. **Dependency Management**
   - Use `uv add <package>` for runtime deps.
   - Use `uv add --group dev <package>` for linters, formatters, test tools.
   - Never edit `uv.lock` manually.
   - Update `pyproject.toml` only via `uv` commands or careful manual edits that keep the format valid.

6. **Code Style & Quality**
   - Follow PEP 8 + modern practices (black/ruff compatible).
   - Keep functions small and focused.
   - Document public APIs with Google style docstrings.
   - Use `py.typed` marker (already present).
   - Look at existing code to infer styling/conventions to use

7. **Copilot Behavior**
   - Propose changes via clear explanations + diffs.
   - Always suggest running `uv run pytest` and `uv run mypy` after edits.
   - If user pastes error from mypy/pytest, prioritize fixing types/tests.
   - Prefer editing existing files over creating new ones unless structure requires it.

## Useful Commands (for Copilot to suggest)
```bash
uv sync --group dev          # after changing pyproject.toml
uv run pytest -q
uv run mypy src
uv run python -c "from python_chat import hello_world; print(hello_world())"
```

## File-Specific Notes
- `pyproject.toml`: Keep mypy strict=true, pytest config, dependency-groups.
- `src/python_chat/__init__.py`: Maintain clean public API exports.
- `tests/`: Must remain fast; mark slow tests if LLM calls are added later.
- `.github/copilot-instructions.md`: Update this file when project rules evolve.

## Tone & Communication
- Be concise and technical.
- Always verify that suggested code would pass `mypy` and `pytest`.
- If something is ambiguous, ask the user for clarification rather than guessing (especially around LLM architecture).