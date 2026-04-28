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

# Architecture overview

Unified runtime lives in `src/vgtr_py/runtime/`:

```
state.py     → RuntimeState (env-major, num_envs first dim)
kernels.py   → step_state(), compute_forces(), integrate() — single source of truth
session.py   → RuntimeSession (public facade, has step/step_one/step_batch)
alloc.py     → make_state() / reset_state()
observe.py   → build_observation()
reward.py    → compute_reward() / build_info()
project.py   → project_anchor_targets()
```

Gym adapters in `adapters/`:
```
gym_env.py      → VGTRGymEnv (single-env RL, wraps RuntimeSession)
vector_env.py   → VectorVGTREnv (batched RL)
```

Control mode is explicit (`"direct"` or `"projection"`), not inferred from
`projection_anchor_indices`.

**Legacy archive**: `legacy/engine_workspace.py` — old workspace-mutating engine (kept for reference).

UI (`ui.py`) holds a `RuntimeSession` directly for both single and batch modes.

# Style guidelines

- Line length limit is 100 columns (configured in `pyproject.toml`).
- Avoid local imports unless strictly necessary (e.g. circular imports).
- Tests should follow these principles:
  - Use functions and fixtures; do not use test classes.
  - Favor targeted, efficient tests over exhaustive edge-case coverage.

# Commits and PRs

- Put `Fixes #<number>` at the end of the commit message body, not in the title.
- PR body should be plain, concise prose. No section headers, checklists, or
  structured templates. Describe the problem, what the change does, and any
  non-obvious tradeoffs.
- PR and commit messages are rendered on GitHub, so don't hard-wrap them at 100
  columns. Let each sentence flow on one line.
