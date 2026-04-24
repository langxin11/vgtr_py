"""最小化 Gymnasium 风格环境示例。

演示流程：
    Workspace JSON -> VGTREnv -> reset()/step()

Usage:
    uv run python examples/env_minimal.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vgtr_py.commands import load_workspace_from_paths
from vgtr_py.env import VGTREnv


def main() -> None:
    workspace = load_workspace_from_paths(
        config_path=None,
        example_path=Path("configs/example.json"),
    )
    env = VGTREnv.from_workspace(workspace, max_steps=5)

    obs, info = env.reset(seed=42)
    print("observation shape:", obs.shape)
    print("initial tracking_error:", info["tracking_error"])

    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[0] = 0.8

    for step_index in range(5):
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"step={step_index + 1} "
            f"reward={reward:.6f} "
            f"tracking_error={info['tracking_error']:.6f} "
            f"rod_length={info['rod_length'].tolist()}"
        )
        if terminated or truncated:
            break


if __name__ == "__main__":
    main()
