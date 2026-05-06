"""支持多种作动控制模式的运行时仿真循环。

演示三种向仿真器输入控制量的方式：

1. GUI 滑块     — Viser 控制组滑块（交互式）。
2. 正弦波       — 程序化开环正弦步态。
3. 脚本回播     — 使用工作区内置脚本矩阵。

Usage:
    uv run python examples/runtime_viser_loop_manual_control.py
    uv run python examples/runtime_viser_loop_manual_control.py --example configs/crawling-bot.json
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
import viser

from vgtr_py.commands import load_workspace_from_paths
from vgtr_py.rendering import SceneRenderer
from vgtr_py.runtime import RuntimeSession

app = typer.Typer(add_completion=False)


@app.command()
def main(
    example: Annotated[
        Path,
        typer.Option(help="Workspace JSON to load."),
    ] = Path("configs/crawling-bot.json"),
    config: Annotated[
        Path | None,
        typer.Option(help="Optional simulation config JSON. Defaults to built-in config."),
    ] = None,
    host: Annotated[
        str,
        typer.Option(help="Viser host."),
    ] = "0.0.0.0",
    port: Annotated[
        int,
        typer.Option(help="Viser port."),
    ] = 8080,
    render_hz: Annotated[
        float,
        typer.Option(help="Renderer refresh rate."),
    ] = 60.0,
    steps_per_frame: Annotated[
        int,
        typer.Option(help="Simulation steps to advance before each render."),
    ] = 20,
    mode: Annotated[
        str,
        typer.Option(
            help="slider = Viser GUI sliders; sine = autonomous gait; script = replay workspace script."
        ),
    ] = "slider",
) -> None:
    """启动支持多种作动模式的运行时仿真查看器。"""
    # ------------------------------------------------------------------
    # 1. Load workspace and compile simulation model / data.
    # ------------------------------------------------------------------
    workspace = load_workspace_from_paths(
        config_path=config,
        example_path=example,
    )
    session = RuntimeSession.from_workspace(workspace, control_mode="control_group")

    # ------------------------------------------------------------------
    # 2. Viser server + scene renderer.
    # ------------------------------------------------------------------
    server = viser.ViserServer(host=host, port=port)
    renderer = SceneRenderer(server=server)
    renderer.setup_scene()
    renderer.render(workspace, anchor_pos=session.state.qpos[0])

    url_host = "localhost" if host in {"0.0.0.0", "::"} else host
    print(f"Viewer: http://{url_host}:{port}")
    print(f"Control mode: {mode}")
    print("Press Ctrl+C to stop.\n")

    # ------------------------------------------------------------------
    # 3. Mode-specific setup.
    # ------------------------------------------------------------------
    num_cg = session.model.control_group_count
    slider_handles: list[viser.GuiSliderHandle[float]] = []

    if mode == "slider" and num_cg > 0:
        # Create one Viser slider per control group.
        with server.gui.add_folder("Manual Actuation"):
            for i in range(num_cg):
                slider = server.gui.add_slider(
                    label=f"CG {i}",
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    initial_value=float(session.state.ctrl_target[0, i]),
                )
                slider_handles.append(slider)
        print(f"Created {num_cg} GUI slider(s). Drag them to actuate.")

    elif mode == "sine":
        server.gui.add_markdown("Running **autonomous sine-wave gait**. No manual input needed.")
        print("Sine-wave frequencies (rad/step):")
        # Give each control group a distinct frequency for interesting motion.
        frequencies = [0.05 * (i + 1) for i in range(num_cg)] if num_cg else []
        for i, f in enumerate(frequencies):
            print(f"  CG {i}: {f:.3f}")

    elif mode == "script":
        if session.model.script.size == 0:
            print("WARNING: Workspace has no script. Falling back to default targets.")
        server.gui.add_markdown("Replaying workspace **script** matrix.")

    # ------------------------------------------------------------------
    # 4. Simulation loop.
    # ------------------------------------------------------------------
    frame_dt = 1.0 / render_hz
    step_idx = 0

    try:
        while True:
            frame_start = time.perf_counter()

            for _ in range(steps_per_frame):
                # ---- decide control input for this step ----------------
                if mode == "slider":
                    # Read slider values and pack into action array.
                    action = np.array([s.value for s in slider_handles], dtype=np.float64)
                    session.step_batch(action)

                elif mode == "sine":
                    # Open-loop sinusoidal pattern.
                    action = np.array(
                        [0.5 + 0.5 * math.sin(0.05 * (i + 1) * step_idx) for i in range(num_cg)],
                        dtype=np.float64,
                    )
                    session.step_batch(action)
                    step_idx += 1

                elif mode == "script":
                    session.advance_script_targets()
                    session.step_batch(None)

                else:
                    session.step_batch(None)

            renderer.render(workspace, anchor_pos=session.state.qpos[0])

            elapsed = time.perf_counter() - frame_start
            if elapsed < frame_dt:
                time.sleep(frame_dt - elapsed)
    except KeyboardInterrupt:
        print("Stopped runtime viewer.")


if __name__ == "__main__":
    app()
