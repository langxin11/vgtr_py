"""批量 Gym 风格环境接口。

``VectorVGTREnv`` 面向 RL rollout 和性能实验，提供 ``(num_envs, action_dim)`` 的
批量动作输入，并返回批量观测、奖励、终止标志和 info。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .batch import BatchSimulator, BatchVGTRData, make_batch_data, reset_batch_data
from .gym_compat import gym, spaces
from .model import VGTRModel, compile_workspace
from .workspace import ROD_TYPE_ACTIVE, Workspace


@dataclass(slots=True)
class VectorVGTREnv(gym.Env):
    """基于 ``BatchVGTRData`` 的 CPU 向量化环境。"""

    model: VGTRModel
    data: BatchVGTRData
    simulator: BatchSimulator
    max_steps: int = 1000

    metadata = {"render_modes": []}

    def __post_init__(self) -> None:
        action_dim = int(self.model.projection_anchor_indices.shape[0] * 3)
        if action_dim == 0:
            action_dim = max(int(self.model.control_group_count), 1)
        action_shape = (self.data.num_envs, action_dim)
        obs_template = _batch_observation(self.model, self.data).astype(np.float32)
        self.single_action_space = spaces.Box(
            low=np.full(action_dim, -10.0, dtype=np.float32),
            high=np.full(action_dim, 10.0, dtype=np.float32),
            shape=(action_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.full(action_shape, -10.0, dtype=np.float32),
            high=np.full(action_shape, 10.0, dtype=np.float32),
            shape=action_shape,
            dtype=np.float32,
        )
        self.single_observation_space = spaces.Box(
            low=np.full(obs_template.shape[1:], -np.inf, dtype=np.float32),
            high=np.full(obs_template.shape[1:], np.inf, dtype=np.float32),
            shape=obs_template.shape[1:],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.full(obs_template.shape, -np.inf, dtype=np.float32),
            high=np.full(obs_template.shape, np.inf, dtype=np.float32),
            shape=obs_template.shape,
            dtype=np.float32,
        )

    @classmethod
    def from_workspace(
        cls,
        workspace: Workspace,
        *,
        num_envs: int,
        max_steps: int = 1000,
    ) -> "VectorVGTREnv":
        """从工作区编译并构建批量环境。"""
        model = compile_workspace(workspace)
        data = make_batch_data(model, num_envs)
        simulator = BatchSimulator()
        return cls(model=model, data=data, simulator=simulator, max_steps=max_steps)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """重置所有环境。"""
        del options
        reset_batch_data(self.model, self.data, seed=seed)
        return _batch_observation(self.model, self.data), _batch_info(self.model, self.data)

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """执行一批环境的单步仿真。"""
        if self.model.projection_anchor_indices.size:
            ctrl_target = _project_anchor_targets_batch(self.model, self.data, action)
        elif self.model.control_group_count:
            ctrl_target = _reshape_batch_action(
                action,
                self.data.num_envs,
                self.model.control_group_count,
            )
        else:
            ctrl_target = None

        self.simulator.step(self.model, self.data, ctrl_target)

        observation = _batch_observation(self.model, self.data)
        reward = _batch_reward(self.model, self.data, action)
        terminated = np.zeros(self.data.num_envs, dtype=np.bool_)
        truncated = self.data.step_count >= self.max_steps
        info = _batch_info(self.model, self.data)
        return observation, reward, terminated, truncated, info


def _reshape_batch_action(action: np.ndarray, num_envs: int, action_dim: int) -> np.ndarray:
    action_array = np.asarray(action, dtype=np.float64)
    if action_array.ndim == 1:
        if action_array.shape[0] != action_dim:
            raise ValueError(
                f"expected action with shape ({action_dim},) or ({num_envs}, {action_dim}), "
                f"got {action_array.shape}"
            )
        action_array = np.broadcast_to(action_array, (num_envs, action_dim))
    if action_array.shape != (num_envs, action_dim):
        raise ValueError(
            f"expected action with shape ({num_envs}, {action_dim}), got {action_array.shape}"
        )
    return action_array


def _project_anchor_targets_batch(
    model: VGTRModel,
    data: BatchVGTRData,
    target_positions: np.ndarray,
) -> np.ndarray:
    if model.projection_anchor_indices.size == 0 or model.control_group_count == 0:
        return np.zeros((data.num_envs, model.control_group_count), dtype=np.float64)

    target = _reshape_batch_action(
        target_positions,
        data.num_envs,
        int(model.projection_anchor_indices.shape[0] * 3),
    ).reshape(data.num_envs, model.projection_anchor_indices.shape[0], 3)
    projected_qpos = data.qpos.copy()
    projected_qpos[:, model.projection_anchor_indices, :] = target

    control_target = data.ctrl_target.copy()
    contributions = np.zeros((data.num_envs, model.control_group_count), dtype=np.float64)
    counts = np.zeros(model.control_group_count, dtype=np.int32)

    for rod_index in np.flatnonzero(model.rod_type == ROD_TYPE_ACTIVE):
        anchor_pair = model.rod_anchors[rod_index]
        desired_length = np.linalg.norm(
            projected_qpos[:, anchor_pair[1], :] - projected_qpos[:, anchor_pair[0], :],
            axis=1,
        )
        min_length, max_length = model.rod_length_limits[rod_index]
        if max_length <= min_length + 1e-9:
            normalized = np.zeros(data.num_envs, dtype=np.float64)
        else:
            normalized = (max_length - desired_length) / (max_length - min_length)
        group_index = int(model.rod_control_group[rod_index])
        contributions[:, group_index] += np.clip(normalized, 0.0, 1.0)
        counts[group_index] += 1

    mask = counts > 0
    control_target[:, mask] = contributions[:, mask] / counts[mask][None, :]
    np.clip(control_target, 0.0, 1.0, out=control_target)
    return control_target


def _batch_observation(model: VGTRModel, data: BatchVGTRData) -> np.ndarray:
    parts = [
        data.qpos.reshape(data.num_envs, -1),
        data.qvel.reshape(data.num_envs, -1),
        data.rod_length,
        data.rod_target_length,
        data.rod_axial_force,
        data.rod_strain,
        data.rod_stalled.astype(np.float64),
        data.ctrl,
        data.ctrl_target,
        data.contact_mask.astype(np.float64),
    ]
    return np.concatenate(parts, axis=1, dtype=np.float64) if parts else np.zeros((data.num_envs, 0))


def _batch_reward(model: VGTRModel, data: BatchVGTRData, action: np.ndarray) -> np.ndarray:
    tracking_error = np.zeros(data.num_envs, dtype=np.float64)
    if model.projection_anchor_indices.size:
        target = _reshape_batch_action(
            action,
            data.num_envs,
            int(model.projection_anchor_indices.shape[0] * 3),
        ).reshape(data.num_envs, model.projection_anchor_indices.shape[0], 3)
        current = data.qpos[:, model.projection_anchor_indices, :]
        tracking_error = np.linalg.norm(current - target, axis=2).mean(axis=1)

    overload_penalty = np.count_nonzero(data.rod_stalled, axis=1).astype(np.float64)
    passive_mask = model.rod_type == 0
    if np.any(passive_mask):
        passive_force_penalty = np.mean(np.abs(data.rod_axial_force[:, passive_mask]), axis=1)
    else:
        passive_force_penalty = np.zeros(data.num_envs, dtype=np.float64)
    if data.rod_strain.size:
        internal_stress_penalty = np.mean(np.abs(data.rod_strain), axis=1)
    else:
        internal_stress_penalty = np.zeros(data.num_envs, dtype=np.float64)
    return -(
        tracking_error
        + 0.1 * overload_penalty
        + 0.01 * passive_force_penalty
        + internal_stress_penalty
    )


def _batch_info(model: VGTRModel, data: BatchVGTRData) -> dict[str, np.ndarray]:
    tracking_error = np.zeros(data.num_envs, dtype=np.float64)
    if model.projection_anchor_indices.size:
        current = data.qpos[:, model.projection_anchor_indices, :]
        rest = model.anchor_rest_pos[model.projection_anchor_indices][None, :, :]
        tracking_error = np.linalg.norm(current - rest, axis=(1, 2))
    return {
        "rod_length": data.rod_length.copy(),
        "rod_axial_force": data.rod_axial_force.copy(),
        "rod_stalled": data.rod_stalled.copy(),
        "contact_mask": data.contact_mask.copy(),
        "tracking_error": tracking_error,
    }
