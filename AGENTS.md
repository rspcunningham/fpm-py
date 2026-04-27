# Repository Guidelines

## Project Structure & Module Organization

This repository packages `ptych`, a Python library for Fourier ptychographic reconstruction. Core reconstruction code lives in `src/ptych/core/`. Data handling / loading is in `src/ptych/data`. Data is provided from a custom repository hosted on the lab team's NextCloud.

Top-level demo scripts are `synthetic_demo.py` and `reconstruction_demo.py`. Documentation and
assets live in `README.md`, `INFO_JSON_SCHEMA.md`, `docs/`, and `demo_images/`.

## Build, Test, and Development Commands

- `uv sync --dev`: install runtime and development dependencies from `pyproject.toml` and
  `uv.lock`.
- `uv run ruff format`: format Python files.
- `uv run ruff check`: run lint checks.
- `uv run ty check`: run static type checks.
- `uv run python -m compileall src synthetic_demo.py reconstruction_demo.py`: verify files compile.

## Coding Style & Naming Conventions

Use descriptive but concise names. Always use snake_case except for classes which should be PascalCase.
Avoid defensive programming -- assume the user is educated on the codebase internals. Prefer to verify data structure at lint-time and trust repo internals.
Absolutely zero backwards compatibility. Warn the user if something will break, but assume that the demo scripts in the repo root are the only callers of this library.
Prefer modern pytorch and ML best practices over custom routines always. Rely on builtin features where sensible. Do not create functionbloat -- clear inline code is better than needless abstraction.

## Testing Guidelines

This is a research/prototype codebase. Ignore testing unless specifically asked for.
