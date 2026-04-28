"""最小化带 Viser 渲染的运行时仿真循环。

演示流程：
    Workspace JSON -> RuntimeSession.step_batch() -> SceneRenderer/Viser

Usage:
    uv run python examples/runtime_viser_loop.py
    uv run python examples/runtime_viser_loop.py --example configs/example.json --port 8081
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

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
    no_script: Annotated[
        bool,
        typer.Option(help="Disable control script playback from the workspace file."),
    ] = False,
) -> None:
    """启动最小化运行时仿真查看器。"""
    # ------------------------------------------------------------------
    # 1. Workspace: authoring/runtime input state loaded from JSON.
    # ------------------------------------------------------------------
    workspace = load_workspace_from_paths(
        config_path=config,
        example_path=example,
    )

    # ------------------------------------------------------------------
    # 2. Model/Data: compile immutable simulation constants and allocate mutable state.
    # ------------------------------------------------------------------
    session = RuntimeSession.from_workspace(workspace, control_mode="direct")

    # ------------------------------------------------------------------
    # 4. Viser frontend: render simulated anchor positions from data.qpos.
    # ------------------------------------------------------------------
    server = viser.ViserServer(host=host, port=port)
    renderer = SceneRenderer(server=server)
    renderer.setup_scene()
    renderer.render(workspace, anchor_pos=session.state.qpos[0])

    url_host = "localhost" if host in {"0.0.0.0", "::"} else host
    print(f"Serving Viser viewer at http://{url_host}:{port}")
    print(f"Script playback: {'OFF' if no_script else 'ON'}  |  Steps/frame: {steps_per_frame}")
    print("Press Ctrl+C to stop.\n")

    frame_dt = 1.0 / render_hz
    try:
        while True:
            frame_start = time.perf_counter()

            for _ in range(steps_per_frame):
                if not no_script:
                    session.advance_script_targets()
                session.step_batch(None)

            renderer.render(workspace, anchor_pos=session.state.qpos[0])

            elapsed = time.perf_counter() - frame_start
            if elapsed < frame_dt:
                time.sleep(frame_dt - elapsed)
    except KeyboardInterrupt:
        print("Stopped runtime viewer.")


if __name__ == "__main__":
    app()
