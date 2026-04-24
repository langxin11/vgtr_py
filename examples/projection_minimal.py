"""最小化投影控制示例。

演示流程：
    Workspace JSON -> VGTRModel/VGTRData -> project_anchor_targets() -> Simulator.step()

Usage:
    uv run python examples/projection_minimal.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vgtr_py.commands import load_workspace_from_paths
from vgtr_py.data import make_data
from vgtr_py.model import compile_workspace
from vgtr_py.projection import project_anchor_targets
from vgtr_py.sim import Simulator


def main() -> None:
    workspace = load_workspace_from_paths(
        config_path=None,
        example_path=Path("configs/example.json"),
    )
    model = compile_workspace(workspace)
    data = make_data(model)
    simulator = Simulator()

    if model.projection_anchor_indices.size == 0:
        raise RuntimeError("This example requires at least one projection target anchor.")

    current_targets = data.qpos[model.projection_anchor_indices].copy()
    desired_targets = current_targets.copy()
    desired_targets[:, 0] -= 0.2

    ctrl_target = project_anchor_targets(model, data, desired_targets)
    print("Projected control target:", ctrl_target.tolist())

    for step_index in range(20):
        simulator.step(model, data, ctrl_target)
        print(
            f"step={step_index + 1} "
            f"ctrl={data.ctrl.tolist()} "
            f"rod_target_length={data.rod_target_length.tolist()}"
        )

    print("Projected anchor positions:")
    print(data.qpos[model.projection_anchor_indices])


if __name__ == "__main__":
    main()
