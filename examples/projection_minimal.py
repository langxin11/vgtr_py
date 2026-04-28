"""最小化投影控制示例。

演示流程：
    Workspace JSON -> RuntimeSession(control_mode=\"projection\") -> step_batch()

Usage:
    uv run python examples/projection_minimal.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vgtr_py.commands import load_workspace_from_paths
from vgtr_py.runtime import RuntimeSession, project_anchor_targets


def main() -> None:
    workspace = load_workspace_from_paths(
        config_path=None,
        example_path=Path("configs/example.json"),
    )
    session = RuntimeSession.from_workspace(workspace, control_mode="projection")

    if session.model.projection_anchor_indices.size == 0:
        raise RuntimeError("This example requires at least one projection target anchor.")

    current_targets = session.state.qpos[0, session.model.projection_anchor_indices].copy()
    desired_targets = current_targets.copy()
    desired_targets[:, 0] -= 0.2

    projected_action = desired_targets.reshape(-1)
    ctrl_target = project_anchor_targets(session.model, session.state, projected_action)[0]
    print("Projected control target:", ctrl_target.tolist())

    for step_index in range(19):
        session.step_batch(projected_action)
        print(
            f"step={step_index + 1} "
            f"ctrl={session.state.ctrl[0].tolist()} "
            f"rod_target_length={session.state.rod_target_length[0].tolist()}"
        )

    print("Projected anchor positions:")
    print(session.state.qpos[0, session.model.projection_anchor_indices])


if __name__ == "__main__":
    main()
