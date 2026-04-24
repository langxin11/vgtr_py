"""最小化批量环境示例。

演示流程：
    Workspace JSON -> VectorVGTREnv -> batched reset()/step()

Usage:
    uv run python examples/vector_env_minimal.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vgtr_py.commands import load_workspace_from_paths
from vgtr_py.vector_env import VectorVGTREnv


def main() -> None:
    workspace = load_workspace_from_paths(
        config_path=None,
        example_path=Path("configs/example.json"),
    )
    env = VectorVGTREnv.from_workspace(workspace, num_envs=4, max_steps=5)

    obs, info = env.reset(seed=42)
    print("batched observation shape:", obs.shape)
    print("initial tracking_error:", info["tracking_error"].tolist())

    actions = np.zeros(env.action_space.shape, dtype=np.float32)
    actions[:, 0] = np.linspace(0.8, 1.1, env.data.num_envs, dtype=np.float32)

    for step_index in range(5):
        obs, reward, terminated, truncated, info = env.step(actions)
        print(
            f"step={step_index + 1} "
            f"reward={reward.round(6).tolist()} "
            f"tracking_error={info['tracking_error'].round(6).tolist()} "
            f"rod_length={info['rod_length'].round(6).tolist()}"
        )
        if np.any(terminated | truncated):
            break


if __name__ == "__main__":
    main()
