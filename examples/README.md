# Python Examples

These examples demonstrate the runtime API from small headless scripts to Viser-based
frontends. Run commands from the repository root.

## Headless Runtime

```bash
uv run python examples/sim_minimal.py
```

Shows the core path:

```text
Workspace JSON -> VGTRModel -> VGTRData -> Simulator.step()
```

## Projection Control

```bash
uv run python examples/projection_minimal.py
```

Projects desired anchor positions to control targets, then advances the simulator.

## Gym-Style Environment

```bash
uv run python examples/env_minimal.py
```

Creates `VGTREnv`, calls `reset()`, and performs a few `step()` calls.

## Batched Environment

```bash
uv run python examples/vector_env_minimal.py
```

Creates `VectorVGTREnv` with multiple CPU-vectorized environments and steps a
batched action array.

## Runtime Viser Loop

```bash
uv run python examples/runtime_viser_loop.py
uv run python examples/runtime_viser_loop.py --example configs/example.json --port 8081
```

Shows the full runtime/rendering path:

```text
Workspace JSON -> VGTRModel -> VGTRData -> Simulator.step() -> SceneRenderer/Viser
```

Open the printed Viser URL in a browser.

## Batched Runtime Viser Loop

```bash
uv run python examples/runtime_viser_batch_loop.py
uv run python examples/runtime_viser_batch_loop.py --num-envs 9 --hide-others --selected-env 0
```

Renders multiple `VectorVGTREnv` instances in one Viser scene using batched mesh
handles, grid offsets, and selected-env controls.

## Full UI

```bash
uv run python examples/ui_minimal.py
```

Launches the full Viser editor and simulator UI at `http://127.0.0.1:8080`.
