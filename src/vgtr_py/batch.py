"""批量运行时数据与 CPU 向量化仿真器。

本模块承接文档中的 Batched rollout 路线：同一份 ``VGTRModel`` 可驱动多个
相互独立的 ``BatchVGTRData`` 实例，用于无 UI 的并行采样与 RL 训练前置验证。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .model import VGTRModel
from .workspace import ROD_TYPE_ACTIVE, ROD_TYPE_ELASTIC, ROD_TYPE_PASSIVE

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


@dataclass(slots=True)
class BatchVGTRData:
    """多个同构仿真实例的可变运行时状态。

    所有环境共享同一份 ``VGTRModel``，batch 维度固定放在第 0 维。
    """

    qpos: FloatArray
    qvel: FloatArray
    forces: FloatArray
    ctrl: FloatArray
    ctrl_target: FloatArray
    rod_length: FloatArray
    rod_target_length: FloatArray
    rod_target_override: FloatArray
    rod_axial_force: FloatArray
    rod_strain: FloatArray
    rod_stalled: BoolArray
    contact_mask: BoolArray
    time: FloatArray
    step_count: IntArray
    i_action: IntArray
    i_action_prev: IntArray

    @property
    def num_envs(self) -> int:
        """返回批量环境数量。"""
        return int(self.qpos.shape[0])


def make_batch_data(model: VGTRModel, num_envs: int, *, seed: int | None = None) -> BatchVGTRData:
    """按模型默认值分配并初始化批量运行时数据。"""
    if num_envs <= 0:
        raise ValueError(f"num_envs must be positive, got {num_envs}")

    data = BatchVGTRData(
        qpos=np.broadcast_to(model.anchor_rest_pos, (num_envs, model.anchor_count, 3)).copy(),
        qvel=np.zeros((num_envs, model.anchor_count, 3), dtype=np.float64),
        forces=np.zeros((num_envs, model.anchor_count, 3), dtype=np.float64),
        ctrl=np.broadcast_to(
            model.control_group_default_target,
            (num_envs, model.control_group_count),
        ).copy(),
        ctrl_target=np.broadcast_to(
            model.control_group_default_target,
            (num_envs, model.control_group_count),
        ).copy(),
        rod_length=np.zeros((num_envs, model.rod_count), dtype=np.float64),
        rod_target_length=np.zeros((num_envs, model.rod_count), dtype=np.float64),
        rod_target_override=np.full((num_envs, model.rod_count), np.nan, dtype=np.float64),
        rod_axial_force=np.zeros((num_envs, model.rod_count), dtype=np.float64),
        rod_strain=np.zeros((num_envs, model.rod_count), dtype=np.float64),
        rod_stalled=np.zeros((num_envs, model.rod_count), dtype=np.bool_),
        contact_mask=np.zeros((num_envs, model.anchor_count), dtype=np.bool_),
        time=np.zeros(num_envs, dtype=np.float64),
        step_count=np.zeros(num_envs, dtype=np.int64),
        i_action=np.zeros(num_envs, dtype=np.int64),
        i_action_prev=np.zeros(num_envs, dtype=np.int64),
    )
    reset_batch_data(model, data, seed=seed)
    return data


def reset_batch_data(model: VGTRModel, data: BatchVGTRData, *, seed: int | None = None) -> None:
    """将批量运行时数据重置为模型静止状态。"""
    del seed
    num_envs = data.num_envs
    data.qpos[...] = model.anchor_rest_pos[None, :, :]
    data.qvel.fill(0.0)
    data.forces.fill(0.0)
    data.ctrl[...] = model.control_group_default_target[None, :]
    data.ctrl_target[...] = model.control_group_default_target[None, :]
    data.rod_length[...] = _batch_rod_lengths(data.qpos, model.rod_anchors)
    data.rod_target_length[...] = model.rod_rest_length[None, :]
    data.rod_target_override.fill(np.nan)
    data.rod_axial_force.fill(0.0)
    data.rod_strain.fill(0.0)
    data.rod_stalled.fill(False)
    data.contact_mask.fill(False)
    data.time[...] = np.zeros(num_envs, dtype=np.float64)
    data.step_count[...] = np.zeros(num_envs, dtype=np.int64)
    data.i_action.fill(0)
    data.i_action_prev.fill(0)


class BatchSimulator:
    """推进 ``BatchVGTRData`` 的 CPU 向量化步进器。"""

    def step(
        self,
        model: VGTRModel,
        data: BatchVGTRData,
        action: np.ndarray | None = None,
    ) -> None:
        """执行一批同构环境的单步仿真。"""
        if action is not None and model.control_group_count:
            action_array = np.asarray(action, dtype=np.float64)
            if action_array.ndim == 1:
                if action_array.shape[0] != model.control_group_count:
                    raise ValueError(
                        "expected action with shape "
                        f"({model.control_group_count},) or "
                        f"({data.num_envs}, {model.control_group_count}), got {action_array.shape}"
                    )
                action_array = np.broadcast_to(action_array, data.ctrl_target.shape)
            if action_array.shape != data.ctrl_target.shape:
                raise ValueError(
                    f"expected action with shape {data.ctrl_target.shape}, got {action_array.shape}"
                )
            data.ctrl_target[...] = np.clip(action_array, 0.0, 1.0)

        _update_batch_control_values(model, data)
        _update_batch_forces(model, data)
        _integrate_batch(model, data)
        data.time += model.config.h
        data.step_count += 1


def _update_batch_control_values(model: VGTRModel, data: BatchVGTRData) -> None:
    if data.ctrl.size == 0:
        return
    delta = np.clip(
        data.ctrl_target - data.ctrl,
        -model.config.contraction_percent_rate,
        model.config.contraction_percent_rate,
    )
    data.ctrl += delta
    np.clip(data.ctrl, 0.0, 1.0, out=data.ctrl)


def _update_batch_forces(model: VGTRModel, data: BatchVGTRData) -> None:
    data.forces.fill(0.0)
    data.rod_stalled.fill(False)

    if model.rod_count:
        start = data.qpos[:, model.rod_anchors[:, 0], :]
        end = data.qpos[:, model.rod_anchors[:, 1], :]
        rod_vec = end - start
        rod_lengths = np.linalg.norm(rod_vec, axis=2)
        safe_lengths = np.where(rod_lengths > 1e-12, rod_lengths, 1.0)
        unit = rod_vec / safe_lengths[:, :, None]

        target_lengths = np.broadcast_to(
            model.rod_rest_length,
            (data.num_envs, model.rod_count),
        ).copy()
        active_mask = model.rod_type == ROD_TYPE_ACTIVE
        elastic_mask = model.rod_type == ROD_TYPE_ELASTIC
        passive_mask = model.rod_type == ROD_TYPE_PASSIVE

        if np.any(active_mask) and data.ctrl.size:
            active_groups = model.rod_control_group[active_mask]
            active_limits = model.rod_length_limits[active_mask]
            target_lengths[:, active_mask] = active_limits[:, 0][None, :] + (
                active_limits[:, 1] - active_limits[:, 0]
            )[None, :] * data.ctrl[:, active_groups]

        if data.rod_target_override.shape == (data.num_envs, model.rod_count):
            override_mask = np.isfinite(data.rod_target_override) & active_mask[None, :]
            if np.any(override_mask):
                clipped_override = np.clip(
                    data.rod_target_override,
                    model.rod_length_limits[:, 0][None, :],
                    model.rod_length_limits[:, 1][None, :],
                )
                target_lengths = np.where(override_mask, clipped_override, target_lengths)

        if np.any(elastic_mask):
            target_lengths[:, elastic_mask] = np.clip(
                model.rod_rest_length[elastic_mask],
                model.rod_length_limits[elastic_mask, 0],
                model.rod_length_limits[elastic_mask, 1],
            )[None, :]
        if np.any(passive_mask):
            target_lengths[:, passive_mask] = np.clip(
                model.rod_rest_length[passive_mask],
                model.rod_length_limits[passive_mask, 0],
                model.rod_length_limits[passive_mask, 1],
            )[None, :]

        raw_force = model.config.k * (rod_lengths - target_lengths)
        clipped_force = np.clip(
            raw_force,
            model.rod_force_limits[:, 0][None, :],
            model.rod_force_limits[:, 1][None, :],
        )
        stalled_by_force = np.abs(raw_force - clipped_force) > 1e-9
        stalled_by_length = (rod_lengths < model.rod_length_limits[:, 0][None, :] - 1e-9) | (
            rod_lengths > model.rod_length_limits[:, 1][None, :] + 1e-9
        )
        data.rod_stalled[...] = active_mask[None, :] & (stalled_by_force | stalled_by_length)

        rod_forces = unit * clipped_force[:, :, None]
        env_indices = np.arange(data.num_envs)[:, None]
        np.add.at(data.forces, (env_indices, model.rod_anchors[:, 0]), rod_forces)
        np.add.at(data.forces, (env_indices, model.rod_anchors[:, 1]), -rod_forces)

        data.rod_length[...] = rod_lengths
        data.rod_target_length[...] = target_lengths
        data.rod_axial_force[...] = clipped_force
        safe_target = np.where(np.abs(target_lengths) > 1e-9, target_lengths, 1.0)
        data.rod_strain[...] = (rod_lengths - target_lengths) / safe_target

    if model.config.gravity:
        data.forces[:, :, 2] -= model.anchor_mass[None, :] * model.config.gravity_factor

    z_pos = data.qpos[:, :, 2]
    vz = data.qvel[:, :, 2]
    data.contact_mask[...] = z_pos < 0.0
    spring_force = -model.config.ground_k * z_pos
    damping_force = -model.config.ground_d * vz
    normal_force = np.where(data.contact_mask, np.maximum(0.0, spring_force + damping_force), 0.0)
    data.forces[:, :, 2] += normal_force

    v_xy = data.qvel[:, :, :2]
    speed_xy = np.linalg.norm(v_xy, axis=2)
    fric_max = model.config.friction_factor * normal_force
    fric_mag = np.minimum(fric_max, 100.0 * speed_xy)
    safe_speed = np.where(speed_xy > 1e-6, speed_xy, 1.0)
    data.forces[:, :, :2] += -v_xy * (fric_mag / safe_speed)[:, :, None]


def _integrate_batch(model: VGTRModel, data: BatchVGTRData) -> None:
    movable_indices = np.flatnonzero(~model.anchor_fixed)
    if movable_indices.size == 0:
        return

    safe_mass = np.where(model.anchor_mass > 1e-9, model.anchor_mass, 1.0)
    data.qvel[:, movable_indices, :] = data.qvel[:, movable_indices, :] + (
        data.forces[:, movable_indices, :] / safe_mass[movable_indices][None, :, None]
    ) * model.config.h
    data.qvel[:, movable_indices, :] *= model.config.damping_ratio

    speed = np.linalg.norm(data.qvel[:, movable_indices, :], axis=2)
    scale = np.where(speed > 10.0, 10.0 / np.where(speed > 1e-12, speed, 1.0), 1.0)
    data.qvel[:, movable_indices, :] *= scale[:, :, None]

    data.qpos[:, movable_indices, :] += data.qvel[:, movable_indices, :] * model.config.h
    below_ground = data.qpos[:, :, 2] < 0.0
    data.qvel[:, :, 2] = np.where(below_ground, 0.0, data.qvel[:, :, 2])
    data.qpos[:, :, 2] = np.where(below_ground, 0.0, data.qpos[:, :, 2])


def _batch_rod_lengths(qpos: FloatArray, rod_anchors: NDArray[np.int32]) -> FloatArray:
    if rod_anchors.size == 0:
        return np.zeros((qpos.shape[0], 0), dtype=np.float64)
    diff = qpos[:, rod_anchors[:, 1], :] - qpos[:, rod_anchors[:, 0], :]
    return np.linalg.norm(diff, axis=2)
