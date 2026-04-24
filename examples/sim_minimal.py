"""最小化无头仿真器示例。

演示流程：
    Workspace JSON -> VGTRModel -> VGTRData -> Simulator.step()

Usage:
    uv run python examples/sim_minimal.py
"""

from __future__ import annotations

from pathlib import Path

from vgtr_py.commands import load_workspace_from_paths
from vgtr_py.data import make_data
from vgtr_py.model import compile_workspace
from vgtr_py.sim import Simulator


def main() -> None:
    workspace = load_workspace_from_paths(
        config_path=None,
        example_path=Path("configs/example.json"),
    )
    model = compile_workspace(workspace)
    data = make_data(model)
    simulator = Simulator()

    print("Initial qpos:")
    print(data.qpos)

    for step_index in range(10):
        simulator.step(model, data)
        print(
            f"step={step_index + 1} "
            f"rod_length={data.rod_length.tolist()} "
            f"rod_force={data.rod_axial_force.tolist()}"
        )

    print("Final qpos:")
    print(data.qpos)


if __name__ == "__main__":
    main()
