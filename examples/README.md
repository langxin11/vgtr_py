# Python Examples

These examples demonstrate the runtime API from small headless scripts to
Viser-based frontends. The recommended entrypoints are the headless runtime and
batched environment examples. Run commands from the repository root.

## Headless Runtime

```bash
uv run python examples/sim_minimal.py
```

Shows the core path:

```text
Workspace JSON -> VGTRModel -> RuntimeSession.step_batch()
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

Creates `VGTRGymEnv`, calls `reset()`, and performs a few `step()` calls.

## Batched Environment

```bash
uv run python examples/vector_env_minimal.py
```

Creates `VGTRVectorEnv` with multiple CPU-vectorized environments and steps a
batched action array.

## Runtime Viser Loop (Single Env)

```bash
uv run python examples/runtime_viser_loop.py
uv run python examples/runtime_viser_loop.py --example configs/example.json --port 8081
```

Single-environment runtime/rendering path:

```text
Workspace JSON -> RuntimeSession -> SceneRenderer/Viser
```

Open the printed Viser URL in a browser.

## Batched Runtime Viser Loop

```bash
uv run python examples/runtime_viser_batch_loop.py
uv run python examples/runtime_viser_batch_loop.py --num-envs 9 --hide-others --selected-env 0
uv run python examples/runtime_viser_batch_loop.py --mode fixed --control-mode direct
uv run python examples/runtime_viser_batch_loop.py --example configs/crawling-bot.json --control-mode direct
```

Multiple `RuntimeSession` envs in one Viser scene using batched mesh handles.
Supports `direct` and `projection` control modes, plus `sine`/`fixed` actuation.
The default `GeoTrussRover.json` path uses `direct` mode and drives all control
groups with phase-shifted sine waves. Use `projection` mode when you want to
drive projection-target anchor positions instead of control groups.

## Manual Control Viser Loop

```bash
uv run python examples/runtime_viser_loop_manual_control.py
```

Single-environment loop with three actuation modes: GUI sliders (interactive),
sine wave (open-loop gait), and script playback (workspace script matrix).

## Full UI

```bash
uv run python examples/ui_minimal.py
```

Launches the full Viser editor and simulator UI at `http://127.0.0.1:8080`.

Opens in a browser with the full Editor + Physics tabs, including batch mode
controls (Environment folder in the Physics tab) and single-env editing tools.
