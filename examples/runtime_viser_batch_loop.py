"""最小化批量环境 Viser 渲染循环。

演示流程：
    Workspace JSON -> VectorVGTREnv -> SceneRenderer.render_batch() -> Viser

Usage:
    uv run python examples/runtime_viser_batch_loop.py
    uv run python examples/runtime_viser_batch_loop.py --num-envs 9 --hide-others --selected-env 0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import viser

from vgtr_py.commands import load_workspace_from_paths
from vgtr_py.rendering import SceneRenderer
from vgtr_py.vector_env import VectorVGTREnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Minimal batch runtime pipeline: Workspace -> VectorVGTREnv -> "
            "batched Viser renderer."
        )
    )
    parser.add_argument(
        "--example",
        type=Path,
        default=Path("configs/GeoTrussRover.json"),
        help="Workspace JSON to load.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional simulation config JSON. Defaults to built-in config.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Viser host.")
    parser.add_argument("--port", type=int, default=8081, help="Viser port.")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of vectorized envs.")
    parser.add_argument("--selected-env", type=int, default=0, help="Highlighted env index.")
    parser.add_argument(
        "--hide-others",
        action="store_true",
        help="Only render the selected env.",
    )
    parser.add_argument("--spacing", type=float, default=3.0, help="Grid spacing between envs.")
    parser.add_argument("--render-hz", type=float, default=60.0, help="Renderer refresh rate.")
    parser.add_argument(
        "--steps-per-frame",
        type=int,
        default=20,
        help="Vector env steps to advance before each render.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = load_workspace_from_paths(
        config_path=args.config,
        example_path=args.example,
    )
    env = VectorVGTREnv.from_workspace(
        workspace,
        num_envs=args.num_envs,
        max_steps=10_000_000,
    )
    env.reset(seed=42)

    server = viser.ViserServer(host=args.host, port=args.port)
    renderer = SceneRenderer(server=server)
    renderer.setup_scene()
    renderer.render_batch(
        workspace,
        batch_anchor_pos=env.data.qpos,
        selected_env=args.selected_env,
        show_only_selected=args.hide_others,
        spacing=args.spacing,
    )

    url_host = "localhost" if args.host in {"0.0.0.0", "::"} else args.host
    print(f"Serving batch Viser viewer at http://{url_host}:{args.port}")
    print("Press Ctrl+C to stop.")

    action = _make_batched_action(env)
    frame_dt = 1.0 / args.render_hz
    try:
        while True:
            frame_start = time.perf_counter()

            for _ in range(args.steps_per_frame):
                env.step(action)

            renderer.render_batch(
                workspace,
                batch_anchor_pos=env.data.qpos,
                selected_env=args.selected_env,
                show_only_selected=args.hide_others,
                spacing=args.spacing,
            )

            elapsed = time.perf_counter() - frame_start
            if elapsed < frame_dt:
                time.sleep(frame_dt - elapsed)
    except KeyboardInterrupt:
        print("Stopped batch runtime viewer.")


def _make_batched_action(env: VectorVGTREnv) -> np.ndarray:
    """为每个 env 生成略有差异的固定动作。"""
    actions = np.zeros(env.action_space.shape, dtype=np.float32)
    if env.model.projection_anchor_indices.size:
        rest_targets = env.model.anchor_rest_pos[env.model.projection_anchor_indices].reshape(-1)
        actions[:] = rest_targets[None, :]
        x_offsets = np.linspace(-0.2, 0.2, env.data.num_envs, dtype=np.float32)
        actions[:, 0] += x_offsets
    elif env.model.control_group_count:
        targets = np.linspace(0.0, 1.0, env.data.num_envs, dtype=np.float32)
        actions[:, 0] = targets
    return actions


if __name__ == "__main__":
    main()
