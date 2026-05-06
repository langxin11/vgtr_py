"""最小化批量环境 Viser 渲染循环。

演示流程：
    Workspace JSON -> RuntimeSession(num_envs=N) -> SceneRenderer.render_batch() -> Viser

支持两种作动模式：
    fixed — 恒定动作（用于观察稳态差异）。
    sine  — 正弦波开环步态（默认，每个 env 频率错开，直观对比动态响应）。

Usage:
    uv run python examples/runtime_viser_batch_loop.py
    uv run python examples/runtime_viser_batch_loop.py --num-envs 9 --hide-others --selected-env 0
    uv run python examples/runtime_viser_batch_loop.py --mode fixed --steps-per-frame 20
    uv run python examples/runtime_viser_batch_loop.py --example configs/crawling-bot.json --control-mode direct
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


def _make_fixed_action(session: RuntimeSession) -> np.ndarray:
    """为每个 env 生成略有差异的固定动作。"""
    actions = np.zeros((session.num_envs, session.action_dim), dtype=np.float32)
    if session.control_mode == "projection":
        rest_targets = session.model.anchor_rest_pos[
            session.model.projection_anchor_indices
        ].reshape(-1)
        actions[:] = rest_targets[None, :]
        x_offsets = np.linspace(-0.2, 0.2, session.num_envs, dtype=np.float32)
        actions[:, 0] += x_offsets
    elif session.model.active_rod_indices.size:
        targets = np.linspace(0.0, 1.0, session.num_envs, dtype=np.float32)
        actions[:, 0] = targets
    return actions


def _make_sine_action(session: RuntimeSession, step_idx: int, dt: float = 0.02) -> np.ndarray:
    """生成随时间变化的正弦波动作，每个 env 频率错开。

    Args:
        env: 向量化环境实例。
        step_idx: 当前物理步序号。
        dt: 单步时间间隔（秒），仅用于计算相位。

    Returns:
        动作数组，shape 与 ``env.action_space`` 一致。
    """
    actions = np.zeros((session.num_envs, session.action_dim), dtype=np.float32)
    t = step_idx * dt

    if session.control_mode == "projection":
        rest_targets = session.model.anchor_rest_pos[
            session.model.projection_anchor_indices
        ].reshape(-1)
        for i in range(session.num_envs):
            freq = 0.08 + i * 0.03
            actions[i, :] = rest_targets
            actions[i, 0] += 0.2 * math.sin(2.0 * math.pi * freq * t)
    elif session.model.active_rod_indices.size:
        active_count = int(session.model.active_rod_indices.shape[0])
        phase_offset = (2.0 * math.pi) / max(active_count, 1)
        for i in range(session.num_envs):
            freq = 0.05 + i * 0.02
            for j in range(active_count):
                phase = j * phase_offset
                actions[i, j] = 0.5 + 0.5 * math.sin(2.0 * math.pi * freq * t + phase)
    return actions


@app.command()
def main(
    example: Annotated[
        Path,
        typer.Option(help="Workspace JSON to load."),
    ] = Path("configs/GeoTrussRover.json"),
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
    ] = 8081,
    num_envs: Annotated[
        int,
        typer.Option(help="Number of vectorized envs."),
    ] = 4,
    selected_env: Annotated[
        int,
        typer.Option(help="Highlighted env index."),
    ] = 0,
    hide_others: Annotated[
        bool,
        typer.Option(help="Only render the selected env."),
    ] = False,
    spacing: Annotated[
        float,
        typer.Option(help="Grid spacing between envs."),
    ] = 3.0,
    render_hz: Annotated[
        float,
        typer.Option(help="Renderer refresh rate."),
    ] = 60.0,
    steps_per_frame: Annotated[
        int,
        typer.Option(help="Vector env steps to advance before each render."),
    ] = 20,
    mode: Annotated[
        str,
        typer.Option(help="Actuation mode: fixed = static targets; sine = sinusoidal gait."),
    ] = "sine",
    control_mode: Annotated[
        str,
        typer.Option(help="Control semantics: direct, control_group, or projection."),
    ] = "direct",
) -> None:
    """启动批量环境实时渲染查看器。"""
    # ------------------------------------------------------------------
    # 1. Load workspace and create vectorized environment.
    # ------------------------------------------------------------------
    workspace = load_workspace_from_paths(
        config_path=config,
        example_path=example,
    )
    if control_mode == "direct" and not np.any(workspace.topology.rod_type == 1):
        raise typer.BadParameter(
            "direct control mode requires at least one active rod in the workspace."
        )
    if control_mode == "control_group" and workspace.script.num_channels <= 0:
        raise typer.BadParameter(
            "control_group mode requires at least one control group in the workspace."
        )
    projection_target_count = int(np.count_nonzero(workspace.topology.anchor_projection_target))
    if control_mode == "projection" and projection_target_count <= 0:
        raise typer.BadParameter(
            "projection control mode requires at least one projection_target site in the workspace."
        )
    session = RuntimeSession.from_workspace(
        workspace,
        num_envs=num_envs,
        max_steps=10_000_000,
        control_mode=control_mode,
    )
    session.reset(seed=42)

    # ------------------------------------------------------------------
    # 2. Viser server + scene renderer.
    # ------------------------------------------------------------------
    server = viser.ViserServer(host=host, port=port)
    renderer = SceneRenderer(server=server)
    renderer.setup_scene()
    renderer.render_batch(
        workspace,
        batch_qpos=session.state.qpos,
        selected_env=selected_env,
        show_only_selected=hide_others,
        spacing=spacing,
        track_selected=True,
    )

    url_host = "localhost" if host in {"0.0.0.0", "::"} else host
    print(f"Serving batch Viser viewer at http://{url_host}:{port}")
    print(f"Mode: {mode}  |  Envs: {num_envs}  |  Steps/frame: {steps_per_frame}")
    print(f"Control mode: {control_mode}  |  Physics dt: {session.model.config.h:.4f}s")
    print("Press Ctrl+C to stop.\n")

    # ------------------------------------------------------------------
    # 3. Pre-compute static action if in fixed mode.
    # ------------------------------------------------------------------
    fixed_action = _make_fixed_action(session) if mode == "fixed" else None

    frame_dt = 1.0 / render_hz
    step_idx = 0

    try:
        while True:
            frame_start = time.perf_counter()

            for _ in range(steps_per_frame):
                if mode == "fixed" and fixed_action is not None:
                    action = fixed_action
                else:
                    action = _make_sine_action(session, step_idx, dt=session.model.config.h)
                session.step_batch(action)
                step_idx += 1

            renderer.render_batch(
                workspace,
                batch_qpos=session.state.qpos,
                selected_env=selected_env,
                show_only_selected=hide_others,
                spacing=spacing,
                track_selected=True,
            )

            elapsed = time.perf_counter() - frame_start
            if elapsed < frame_dt:
                time.sleep(frame_dt - elapsed)
    except KeyboardInterrupt:
        print("Stopped batch runtime viewer.")


if __name__ == "__main__":
    app()
