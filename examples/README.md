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

## Full UI

```bash
uv run python examples/ui_minimal.py
```

Launches the full Viser editor and simulator UI at `http://127.0.0.1:8080`.
