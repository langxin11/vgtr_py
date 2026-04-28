"""最小化无头仿真器示例。

演示流程：
    Workspace JSON -> VGTRModel -> RuntimeSession.step_batch()

Usage:
    uv run python examples/sim_minimal.py
"""

from __future__ import annotations

from pathlib import Path

from vgtr_py.commands import load_workspace_from_paths
from vgtr_py.runtime import RuntimeSession


def main() -> None:
    workspace = load_workspace_from_paths(
        config_path=None,
        example_path=Path("configs/example.json"),
    )
    session = RuntimeSession.from_workspace(workspace, control_mode="direct")

    print("Initial qpos:")
    print(session.state.qpos[0])

    for step_index in range(10):
        session.step_batch()
        print(
            f"step={step_index + 1} "
            f"rod_length={session.state.rod_length[0].tolist()} "
            f"rod_force={session.state.rod_axial_force[0].tolist()}"
        )

    print("Final qpos:")
    print(session.state.qpos[0])


if __name__ == "__main__":
    main()
