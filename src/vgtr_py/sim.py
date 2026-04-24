"""基于 VGTRModel 与 VGTRData 的运行时仿真器。

Simulator 为无状态步进器：所有状态保存在 VGTRData 中，
步进时根据 VGTRModel 的静态参数计算力并积分更新状态。
"""

from __future__ import annotations

import numpy as np

from .data import VGTRData
from .model import VGTRModel
from .workspace import ROD_TYPE_ACTIVE, ROD_TYPE_ELASTIC, ROD_TYPE_PASSIVE


class Simulator:
    """推进单个 VGTRData 实例的无状态步进器。

    典型调用链：
        simulator.step(model, data)           # 脚本/默认值驱动
        simulator.step(model, data, action)   # 外部策略/控制驱动
    """

    def step(self, model: VGTRModel, data: VGTRData, action: np.ndarray | None = None) -> None:
        """执行单步仿真。

        流程：
        1. 若传入 action，更新控制目标 ctrl_target；
        2. 平滑逼近控制值（受 contraction_percent_rate 限制）；
        3. 计算杆组受力（含堵转检测）；
        4. 显式欧拉积分更新速度与位置；
        5. 处理地面碰撞与摩擦。

        Args:
            model: 只读静态模型。
            data: 可变运行时状态。
            action: 可选的控制组目标值，shape (control_group_count,)，值域 [0, 1]。

        Raises:
            ValueError: action 维度与控制组数量不匹配时抛出。
        """
        if action is not None:
            action_array = np.asarray(action, dtype=np.float64).reshape(-1)
            if action_array.shape[0] != model.control_group_count:
                raise ValueError(
                    f"expected action with shape ({model.control_group_count},), got {action_array.shape}"
                )
            data.ctrl_target[:] = np.clip(action_array, 0.0, 1.0)

        _update_control_values(model, data)
        _update_forces(model, data)
        _integrate(model, data)
        data.time += model.config.h
        data.step_count += 1
        if data.record_frames:
            if data.frames is None:
                data.frames = []
            data.frames.append(data.qpos.tolist())


def advance_script_targets(model: VGTRModel, data: VGTRData) -> None:
    """按运行时步计数器推进脚本控制目标。

    根据 data.step_count 计算当前动作索引 i_action，
    从 model.script 中读取对应列并写入 data.ctrl_target。
    """
    if model.script.size == 0 or model.num_actions <= 0 or model.control_group_count == 0:
        return

    threshold = ((data.i_action + 1) % model.num_actions) * model.config.num_steps_action
    if data.step_count <= threshold:
        return

    data.i_action_prev = data.i_action
    data.i_action = int(data.step_count / model.config.num_steps_action) % model.num_actions
    data.ctrl_target[:] = np.clip(model.script[:, data.i_action], 0.0, 1.0)


def _update_control_values(model: VGTRModel, data: VGTRData) -> None:
    """平滑更新当前控制值向目标值逼近。

    控制值变化速率受 model.config.contraction_percent_rate 限制，
    模拟真实执行器的响应延迟。
    """
    if data.ctrl.size == 0:
        return
    delta = np.clip(
        data.ctrl_target - data.ctrl,
        -model.config.contraction_percent_rate,
        model.config.contraction_percent_rate,
    )
    data.ctrl += delta
    np.clip(data.ctrl, 0.0, 1.0, out=data.ctrl)


def _update_forces(model: VGTRModel, data: VGTRData) -> None:
    """计算全场受力。

    包括：
    - 杆组轴向弹簧力（含主动/弹性/被动三种类型的目标长度处理）
    - 超程/超力导致的堵转检测与截断
    - 重力
    - 地面惩罚力（弹簧-阻尼）
    - 库仑摩擦
    """
    data.forces.fill(0.0)
    data.rod_stalled.fill(False)

    if model.rod_count:
        start = data.qpos[model.rod_anchors[:, 0]]
        end = data.qpos[model.rod_anchors[:, 1]]
        rod_vec = end - start
        rod_lengths = np.linalg.norm(rod_vec, axis=1)
        safe_lengths = np.where(rod_lengths > 1e-12, rod_lengths, 1.0)
        unit = rod_vec / safe_lengths[:, None]

        target_lengths = model.rod_rest_length.copy()
        active_mask = model.rod_type == ROD_TYPE_ACTIVE
        elastic_mask = model.rod_type == ROD_TYPE_ELASTIC
        passive_mask = model.rod_type == ROD_TYPE_PASSIVE

        if np.any(active_mask) and data.ctrl.size:
            active_groups = model.rod_control_group[active_mask]
            active_limits = model.rod_length_limits[active_mask]
            # ctrl = 0 对应 min_length，ctrl = 1 对应 max_length
            target_lengths[active_mask] = active_limits[:, 0] + (
                active_limits[:, 1] - active_limits[:, 0]
            ) * data.ctrl[active_groups]
        if data.rod_target_override.shape[0] == model.rod_count:
            override_mask = np.isfinite(data.rod_target_override) & active_mask
            if np.any(override_mask):
                target_lengths[override_mask] = np.clip(
                    data.rod_target_override[override_mask],
                    model.rod_length_limits[override_mask, 0],
                    model.rod_length_limits[override_mask, 1],
                )
        if np.any(elastic_mask):
            target_lengths[elastic_mask] = np.clip(
                model.rod_rest_length[elastic_mask],
                model.rod_length_limits[elastic_mask, 0],
                model.rod_length_limits[elastic_mask, 1],
            )
        if np.any(passive_mask):
            target_lengths[passive_mask] = np.clip(
                model.rod_rest_length[passive_mask],
                model.rod_length_limits[passive_mask, 0],
                model.rod_length_limits[passive_mask, 1],
            )

        raw_force = model.config.k * (rod_lengths - target_lengths)
        clipped_force = np.clip(
            raw_force,
            model.rod_force_limits[:, 0],
            model.rod_force_limits[:, 1],
        )
        stalled_by_force = np.abs(raw_force - clipped_force) > 1e-9
        stalled_by_length = (rod_lengths < model.rod_length_limits[:, 0] - 1e-9) | (
            rod_lengths > model.rod_length_limits[:, 1] + 1e-9
        )
        data.rod_stalled[:] = active_mask & (stalled_by_force | stalled_by_length)

        rod_forces = unit * clipped_force[:, None]
        np.add.at(data.forces, model.rod_anchors[:, 0], rod_forces)
        np.add.at(data.forces, model.rod_anchors[:, 1], -rod_forces)

        data.rod_length = rod_lengths
        data.rod_target_length = target_lengths
        data.rod_axial_force = clipped_force
        safe_target = np.where(np.abs(target_lengths) > 1e-9, target_lengths, 1.0)
        data.rod_strain = (rod_lengths - target_lengths) / safe_target

    if model.config.gravity:
        data.forces[:, 2] -= model.anchor_mass * model.config.gravity_factor

    z_pos = data.qpos[:, 2]
    vz = data.qvel[:, 2]
    data.contact_mask = z_pos < 0.0
    if np.any(data.contact_mask):
        spring_force = -model.config.ground_k * z_pos[data.contact_mask]
        damping_force = -model.config.ground_d * vz[data.contact_mask]
        normal_force = np.maximum(0.0, spring_force + damping_force)
        data.forces[data.contact_mask, 2] += normal_force

        v_xy = data.qvel[data.contact_mask, :2]
        speed_xy = np.linalg.norm(v_xy, axis=1)
        fric_max = model.config.friction_factor * normal_force
        fric_mag = np.minimum(fric_max, 100.0 * speed_xy)
        safe_speed = np.where(speed_xy > 1e-6, speed_xy, 1.0)
        data.forces[data.contact_mask, :2] += -v_xy * (fric_mag / safe_speed)[:, None]


def _integrate(model: VGTRModel, data: VGTRData) -> None:
    """显式欧拉积分更新锚点状态。

    对非固定锚点：
        v += (F / m) * h
        v *= damping
        x += v * h

    同时进行速度限幅（>10 m/s 时截断）与地面穿透修正。
    """
    movable = ~model.anchor_fixed
    if not np.any(movable):
        return

    safe_mass = np.where(model.anchor_mass > 1e-9, model.anchor_mass, 1.0)
    data.qvel[movable] += (data.forces[movable] / safe_mass[movable, None]) * model.config.h
    data.qvel[movable] *= model.config.damping_ratio

    speed = np.linalg.norm(data.qvel[movable], axis=1)
    fast_mask = speed > 10.0
    if np.any(fast_mask):
        limited = data.qvel[movable]
        limited_idx = np.flatnonzero(fast_mask)
        limited[limited_idx] *= (10.0 / speed[fast_mask])[:, None]
        data.qvel[movable] = limited

    data.qpos[movable] += data.qvel[movable] * model.config.h
    below_ground = data.qpos[:, 2] < 0.0
    if np.any(below_ground):
        data.qvel[below_ground, 2] = 0.0
        data.qpos[below_ground, 2] = 0.0
