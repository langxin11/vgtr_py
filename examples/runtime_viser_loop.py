"""最小化带 Viser 渲染的运行时仿真循环。

演示流程：
    Workspace JSON -> VGTRModel -> VGTRData -> Simulator.step() -> SceneRenderer/Viser

Usage:
    uv run python examples/runtime_viser_loop.py
    uv run python examples/runtime_viser_loop.py --example configs/example.json --port 8081
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import viser

from vgtr_py.commands import load_workspace_from_paths
from vgtr_py.data import make_data
from vgtr_py.model import compile_workspace
from vgtr_py.rendering import SceneRenderer
from vgtr_py.sim import Simulator, advance_script_targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Minimal runtime pipeline: Workspace -> VGTRModel -> VGTRData -> "
            "Simulator step loop -> Viser renderer."
        )
    )
    parser.add_argument(
        "--example",
        type=Path,
        default=Path("configs/crawling-bot.json"),
        help="Workspace JSON to load.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional simulation config JSON. Defaults to built-in config.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Viser host.")
    parser.add_argument("--port", type=int, default=8080, help="Viser port.")
    parser.add_argument("--render-hz", type=float, default=60.0, help="Renderer refresh rate.")
    parser.add_argument(
        "--steps-per-frame",
        type=int,
        default=20,
        help="Simulation steps to advance before each render.",
    )
    parser.add_argument(
        "--no-script",
        action="store_true",
        help="Disable control script playback from the workspace file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Workspace: authoring/runtime input state loaded from JSON.
    workspace = load_workspace_from_paths(
        config_path=args.config,
        example_path=args.example,
    )

    # 2. Model/Data: compile immutable simulation constants and allocate mutable state.
    model = compile_workspace(workspace)
    data = make_data(model)

    # 3. Simulator: stateless stepper operating on model + data.
    simulator = Simulator()

    # 4. Viser frontend: render simulated anchor positions from data.qpos.
    server = viser.ViserServer(host=args.host, port=args.port)
    renderer = SceneRenderer(server=server)
    renderer.setup_scene()
    renderer.render(workspace, anchor_pos=data.qpos)

    url_host = "localhost" if args.host in {"0.0.0.0", "::"} else args.host
    print(f"Serving Viser viewer at http://{url_host}:{args.port}")
    print("Press Ctrl+C to stop.")

    frame_dt = 1.0 / args.render_hz
    try:
        while True:
            frame_start = time.perf_counter()

            for _ in range(args.steps_per_frame):
                if not args.no_script:
                    advance_script_targets(model, data)
                simulator.step(model, data)

            renderer.render(workspace, anchor_pos=data.qpos)

            elapsed = time.perf_counter() - frame_start
            if elapsed < frame_dt:
                time.sleep(frame_dt - elapsed)
    except KeyboardInterrupt:
        print("Stopped runtime viewer.")


if __name__ == "__main__":
    main()
