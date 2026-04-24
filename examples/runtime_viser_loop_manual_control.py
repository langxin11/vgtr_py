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

import argparse
import math
import time
from pathlib import Path

import numpy as np
import viser

from vgtr_py.commands import load_workspace_from_paths
from vgtr_py.data import make_data
from vgtr_py.model import compile_workspace
from vgtr_py.rendering import SceneRenderer
from vgtr_py.sim import Simulator, advance_script_targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Runtime loop with manual / scripted / wave actuation."
    )
    parser.add_argument(
        "--example", type=Path, default=Path("configs/crawling-bot.json")
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--render-hz", type=float, default=60.0)
    parser.add_argument("--steps-per-frame", type=int, default=20)
    parser.add_argument(
        "--mode",
        choices=["slider", "sine", "script"],
        default="slider",
        help=(
            "slider = Viser GUI sliders (default); "
            "sine = autonomous sinusoidal gait; "
            "script = replay workspace script matrix."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load workspace and compile simulation model / data.
    # ------------------------------------------------------------------
    workspace = load_workspace_from_paths(
        config_path=args.config,
        example_path=args.example,
    )
    model = compile_workspace(workspace)
    data = make_data(model)
    simulator = Simulator()

    # ------------------------------------------------------------------
    # 2. Viser server + scene renderer.
    # ------------------------------------------------------------------
    server = viser.ViserServer(host=args.host, port=args.port)
    renderer = SceneRenderer(server=server)
    renderer.setup_scene()
    renderer.render(workspace, anchor_pos=data.qpos)

    url_host = "localhost" if args.host in {"0.0.0.0", "::"} else args.host
    print(f"Viewer: http://{url_host}:{args.port}")
    print(f"Control mode: {args.mode}")
    print("Press Ctrl+C to stop.\n")

    # ------------------------------------------------------------------
    # 3. Mode-specific setup.
    # ------------------------------------------------------------------
    num_cg = model.control_group_count
    slider_handles: list[viser.GuiSliderHandle[float]] = []

    if args.mode == "slider" and num_cg > 0:
        # Create one Viser slider per control group.
        with server.gui.add_folder("Manual Actuation"):
            for i in range(num_cg):
                slider = server.gui.add_slider(
                    label=f"CG {i}",
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    initial_value=float(data.ctrl_target[i]),
                )
                slider_handles.append(slider)
        print(f"Created {num_cg} GUI slider(s). Drag them to actuate.")

    elif args.mode == "sine":
        server.gui.add_markdown(
            "Running **autonomous sine-wave gait**. No manual input needed."
        )
        print("Sine-wave frequencies (rad/step):")
        # Give each control group a distinct frequency for interesting motion.
        frequencies = [0.05 * (i + 1) for i in range(num_cg)] if num_cg else []
        for i, f in enumerate(frequencies):
            print(f"  CG {i}: {f:.3f}")

    elif args.mode == "script":
        if model.script.size == 0:
            print("WARNING: Workspace has no script. Falling back to default targets.")
        server.gui.add_markdown("Replaying workspace **script** matrix.")

    # ------------------------------------------------------------------
    # 4. Simulation loop.
    # ------------------------------------------------------------------
    frame_dt = 1.0 / args.render_hz
    step_idx = 0

    try:
        while True:
            frame_start = time.perf_counter()

            for _ in range(args.steps_per_frame):
                # ---- decide control input for this step ----------------
                if args.mode == "slider":
                    # Read slider values and pack into action array.
                    action = np.array([s.value for s in slider_handles], dtype=np.float64)
                    simulator.step(model, data, action=action)

                elif args.mode == "sine":
                    # Open-loop sinusoidal pattern.
                    action = np.array(
                        [
                            0.5 + 0.5 * math.sin(0.05 * (i + 1) * step_idx)
                            for i in range(num_cg)
                        ],
                        dtype=np.float64,
                    )
                    simulator.step(model, data, action=action)
                    step_idx += 1

                elif args.mode == "script":
                    advance_script_targets(model, data)
                    simulator.step(model, data)

                else:
                    simulator.step(model, data)

            renderer.render(workspace, anchor_pos=data.qpos)

            elapsed = time.perf_counter() - frame_start
            if elapsed < frame_dt:
                time.sleep(frame_dt - elapsed)
    except KeyboardInterrupt:
        print("Stopped runtime viewer.")


if __name__ == "__main__":
    main()
