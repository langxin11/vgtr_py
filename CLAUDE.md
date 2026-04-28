# Development Workflow

**Always use `uv run`, not python**.

```sh
# 1. Make changes.

# 2. Format and lint.
uv run ruff format
uv run ruff check --fix

# 3. Run tests.
uv run pytest tests/                      # Full suite
uv run pytest tests/<test_file>.py        # Specific file
uv run pytest tests/ -k "<test_name>"     # Filter by name
```

# Style guidelines

- Line length limit is 100 columns (configured in `pyproject.toml`).
- Avoid local imports unless strictly necessary (e.g. circular imports).
- Tests should follow these principles:
  - Use functions and fixtures; do not use test classes.
  - Favor targeted, efficient tests over exhaustive edge-case coverage.
  - Prefer running individual tests rather than the full test suite to improve iteration speed.

# Commits and PRs

- Put `Fixes #<number>` at the end of the commit message body, not in the title.
- PR body should be plain, concise prose. No section headers, checklists, or structured templates. Describe the problem, what the change does, and any non-obvious tradeoffs.
- PR and commit messages are rendered on GitHub, so don't hard-wrap them at 100 columns. Let each sentence flow on one line.
