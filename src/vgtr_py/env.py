"""Gymnasium 风格的 VGTR 强化学习环境。

封装 Simulator、VGTRModel 与 VGTRData，提供标准 RL 接口：
    reset() -> obs, info
    step(action) -> obs, reward, terminated, truncated, info
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import VGTRData, make_data, reset_data
from .gym_compat import gym, spaces
from .model import VGTRModel, compile_workspace
from .projection import project_anchor_targets
from .sim import Simulator
from .workspace import Workspace


@dataclass(slots=True)
class VGTREnv(gym.Env):
    """基于 VGTRModel + VGTRData + Simulator 的 RL 环境。

    支持两种动作模式：
    1. 节点目标模式（projection_anchor_indices 非空）：动作空间为关键锚点的目标位置。
    2. 直接控制模式：动作空间为控制组目标值。

    Attributes:
        model: 只读静态模型。
        data: 动态运行时状态。
        simulator: 无状态仿真步进器。
        max_steps: 单回合最大步数。
    """

    model: VGTRModel
    data: VGTRData
    simulator: Simulator
    max_steps: int = 1000

    metadata = {"render_modes": []}

    def __post_init__(self) -> None:
        action_dim = int(self.model.projection_anchor_indices.shape[0] * 3)
        if action_dim == 0:
            action_dim = max(int(self.model.control_group_count), 1)
        action_low = np.full(action_dim, -10.0, dtype=np.float32)
        action_high = np.full(action_dim, 10.0, dtype=np.float32)
        obs_template = _observation(self.model, self.data).astype(np.float32)
        self.action_space = spaces.Box(
            low=action_low,
            high=action_high,
            shape=action_low.shape,
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.full(obs_template.shape, -np.inf, dtype=np.float32),
            high=np.full(obs_template.shape, np.inf, dtype=np.float32),
            shape=obs_template.shape,
            dtype=np.float32,
        )

    @classmethod
    def from_workspace(cls, workspace: Workspace, *, max_steps: int = 1000) -> "VGTREnv":
        """从工作区编译并构建环境。

        Args:
            workspace: 编辑态工作区。
            max_steps: 单回合最大步数。

        Returns:
            构建好的 VGTREnv 实例。
        """
        model = compile_workspace(workspace)
        data = make_data(model)
        simulator = Simulator()
        return cls(model=model, data=data, simulator=simulator, max_steps=max_steps)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
        """重置环境到初始状态。

        Args:
            seed: 随机种子。
            options: 额外选项（当前未使用）。

        Returns:
            observation: 初始观测向量。
            info: 初始信息字典。
        """
        del options
        reset_data(self.model, self.data, seed=seed)
        return _observation(self.model, self.data), _info(self.model, self.data)

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, np.ndarray | float]]:
        """执行一步仿真。

        Args:
            action: 动作向量。若启用投影层，为关键锚点目标位置；否则为控制组目标值。

        Returns:
            observation: 下一步观测。
            reward: 即时奖励。
            terminated: 是否达到终止状态（当前恒为 False）。
            truncated: 是否超过最大步数。
            info: 额外信息字典。
        """
        if self.model.projection_anchor_indices.size:
            ctrl_target = project_anchor_targets(self.model, self.data, action)
        else:
            ctrl_target = np.asarray(action, dtype=np.float64).reshape(-1)
        self.simulator.step(self.model, self.data, ctrl_target)

        observation = _observation(self.model, self.data)
        reward = _reward(self.model, self.data, action)
        terminated = False
        truncated = self.data.step_count >= self.max_steps
        info = _info(self.model, self.data)
        return observation, reward, terminated, truncated, info


def _observation(model: VGTRModel, data: VGTRData) -> np.ndarray:
    """拼接观测向量。

    观测包含：锚点位置、速度、杆长、目标杆长、轴向力、应变、堵转标志、
    控制量、控制目标、地面接触标志。
    """
    parts = [
        data.qpos.reshape(-1),
        data.qvel.reshape(-1),
        data.rod_length,
        data.rod_target_length,
        data.rod_axial_force,
        data.rod_strain,
        data.rod_stalled.astype(np.float64),
        data.ctrl,
        data.ctrl_target,
        data.contact_mask.astype(np.float64),
    ]
    return np.concatenate(parts, dtype=np.float64) if parts else np.zeros(0, dtype=np.float64)


def _reward(model: VGTRModel, data: VGTRData, action: np.ndarray) -> float:
    """计算即时奖励。

    奖励为负值，惩罚项包括：
    - 跟踪误差（当前位置与目标位置的偏差）
    - 堵转杆数量
    - 被动杆受力（鼓励被动杆保持刚性）
    - 结构内应力（应变均值）
    """
    tracking_error = 0.0
    if model.projection_anchor_indices.size:
        target = np.asarray(action, dtype=np.float64).reshape(model.projection_anchor_indices.shape[0], 3)
        current = data.qpos[model.projection_anchor_indices]
        tracking_error = float(np.linalg.norm(current - target, axis=1).mean())

    overload_penalty = float(np.count_nonzero(data.rod_stalled))
    passive_force_penalty = 0.0
    passive_mask = model.rod_type == 0
    if np.any(passive_mask):
        passive_force_penalty = float(np.mean(np.abs(data.rod_axial_force[passive_mask])))
    internal_stress_penalty = float(np.mean(np.abs(data.rod_strain))) if data.rod_strain.size else 0.0
    return -(tracking_error + 0.1 * overload_penalty + 0.01 * passive_force_penalty + internal_stress_penalty)


def _info(model: VGTRModel, data: VGTRData) -> dict[str, np.ndarray | float]:
    """构建信息字典。

    Returns:
        包含杆长、轴力、堵转标志、接触标志、跟踪误差的字典。
    """
    tracking_error = 0.0
    if model.projection_anchor_indices.size:
        current = data.qpos[model.projection_anchor_indices]
        tracking_error = float(np.linalg.norm(current - model.anchor_rest_pos[model.projection_anchor_indices]))
    return {
        "rod_length": data.rod_length.copy(),
        "rod_axial_force": data.rod_axial_force.copy(),
        "rod_stalled": data.rod_stalled.copy(),
        "contact_mask": data.contact_mask.copy(),
        "tracking_error": tracking_error,
    }
