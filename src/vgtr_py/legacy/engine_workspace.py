"""Legacy workspace-mutating physics engine.

保留旧的 ``Workspace`` 直写式仿真入口，仅用于兼容历史调用。
"""

from __future__ import annotations

import numpy as np

from ..workspace import Workspace


def precompute(workspace: Workspace) -> None:
    """同步拓扑与运行时数组尺寸，并刷新派生长度缓存。"""
    sync_workspace_shapes(workspace)
    refresh_derived_state(workspace)


def sync_workspace_shapes(workspace: Workspace) -> None:
    """拓扑/脚本变更后调用：同步所有状态数组尺寸。"""
    topology = workspace.topology
    physics = workspace.physics
    script = workspace.script
    config = workspace.config
    robot = workspace.robot_config

    num_anchors = topology.anchor_pos.shape[0]
    num_rod_groups = topology.rod_anchors.shape[0]
    num_control_groups = script.num_channels
    num_actions = script.num_actions

    if physics.velocities.shape != topology.anchor_pos.shape:
        new_velocities = np.zeros((num_anchors, 3), dtype=np.float64)
        shared = min(num_anchors, physics.velocities.shape[0])
        new_velocities[:shared] = physics.velocities[:shared]
        physics.velocities = new_velocities

    if physics.forces.shape != topology.anchor_pos.shape:
        physics.forces = np.zeros((num_anchors, 3), dtype=np.float64)

    topology.anchor_fixed = _resize_bool_array(topology.anchor_fixed, num_anchors, False)
    topology.anchor_mass = _resize_float_array(
        topology.anchor_mass,
        num_anchors,
        float(robot.anchor.mass),
    )
    topology.anchor_radius = _resize_float_array(
        topology.anchor_radius,
        num_anchors,
        float(robot.anchor.radius),
    )
    topology.rod_rest_length = _resize_float_array(
        topology.rod_rest_length, num_rod_groups, 2.0
    )
    topology.rod_min_length = _resize_float_array(
        topology.rod_min_length,
        num_rod_groups,
        1.0,
    )
    topology.rod_enabled = _resize_bool_array(topology.rod_enabled, num_rod_groups, True)
    topology.rod_actuated = _resize_bool_array(topology.rod_actuated, num_rod_groups, False)
    topology.rod_control_group = _resize_int_array(topology.rod_control_group, num_rod_groups, 0)
    topology.rod_group_mass = _resize_float_array(topology.rod_group_mass, num_rod_groups, 1.0)
    topology.rod_radius = _resize_float_array(
        topology.rod_radius,
        num_rod_groups,
        float(robot.rod_group.rod_radius),
    )
    topology.rod_sleeve_half = _resize_float3_array(
        topology.rod_sleeve_half,
        num_rod_groups,
        np.asarray(
            [
                float(robot.rod_group.sleeve_radius),
                float(robot.rod_group.sleeve_radius),
                float(robot.rod_group.sleeve_display_half_length_ratio),
            ],
            dtype=np.float64,
        ),
    )


    physics.control_group_target = _resize_float_array(
        physics.control_group_target,
        num_control_groups,
        0.0,
    )
    physics.control_group_value = _resize_float_array(
        physics.control_group_value,
        num_control_groups,
        0.0,
    )
    script.script = _resize_script(
        script.script,
        num_channels=num_control_groups,
        num_actions=num_actions,
    )
    script.control_group_enabled = _resize_bool_array(
        script.control_group_enabled,
        num_control_groups,
        True,
    )
    script.control_group_colors = _resize_uint8_colors(
        script.control_group_colors,
        num_control_groups,
    )

    workspace.ui.anchor_status = _resize_status_array(workspace.ui.anchor_status, num_anchors)
    workspace.ui.rod_group_status = _resize_status_array(
        workspace.ui.rod_group_status, num_rod_groups
    )


def refresh_derived_state(workspace: Workspace) -> None:
    """每帧可调用：刷新由拓扑直接派生的运行时缓存。"""
    workspace.physics.lengths = _rod_group_lengths(
        workspace.topology.anchor_pos,
        workspace.topology.rod_anchors,
    )


def run_script(workspace: Workspace) -> None:
    """根据模拟步数推进控制组脚本。"""
    physics = workspace.physics
    script_state = workspace.script
    config = workspace.config

    if script_state.script.size == 0 or script_state.num_actions <= 0:
        return

    threshold = ((physics.i_action + 1) % script_state.num_actions) * config.num_steps_action
    if physics.num_steps <= threshold:
        return

    physics.i_action = int(physics.num_steps / config.num_steps_action) % script_state.num_actions
    if workspace.ui.editing:
        physics.i_action = 0

    physics.control_group_target[:] = script_state.script[:, physics.i_action]
    np.clip(physics.control_group_target, 0.0, 1.0, out=physics.control_group_target)


def update_forces(workspace: Workspace) -> None:
    """根据杆组当前长度与目标长度差计算锚点受力。"""
    topology = workspace.topology
    physics = workspace.physics
    config = workspace.config

    num_anchors = topology.anchor_pos.shape[0]
    physics.forces = np.zeros((num_anchors, 3), dtype=np.float64)

    if topology.rod_anchors.size:
        start = topology.anchor_pos[topology.rod_anchors[:, 0]]
        end = topology.anchor_pos[topology.rod_anchors[:, 1]]
        rod_vec = end - start
        rod_lengths = np.linalg.norm(rod_vec, axis=1)
        physics.lengths = rod_lengths

        safe_lengths = np.where(rod_lengths > 1e-12, rod_lengths, 1.0)
        unit = rod_vec / safe_lengths[:, None]

        target_lengths = topology.rod_rest_length.copy()
        active_mask = topology.rod_enabled
        if np.any(active_mask) and physics.control_group_value.size:
            active_indices = np.flatnonzero(active_mask)
            active_groups = topology.rod_control_group[active_mask]
            target_lengths[active_indices] = _lerp(
                topology.rod_rest_length[active_mask],
                topology.rod_min_length[active_mask],
                physics.control_group_value[active_groups],
            )

        force_magnitudes = config.k * (rod_lengths - target_lengths)
        rod_forces = unit * force_magnitudes[:, None]

        np.add.at(physics.forces, topology.rod_anchors[:, 0], rod_forces)
        np.add.at(physics.forces, topology.rod_anchors[:, 1], -rod_forces)

    if config.gravity:
        physics.forces[:, 2] -= topology.anchor_mass * config.gravity_factor

    # --- 地面接触力 (Penalty Force + Coulomb Friction) ---
    z_pos = topology.anchor_pos[:, 2]
    v_z = physics.velocities[:, 2]
    
    # 仅处理 z < 0 的点
    contact_mask = z_pos < 0
    if np.any(contact_mask):
        # 1. 法向力 (弹簧-阻尼模型)
        # Fn = -k*z - d*vz
        # 使用 max(0, ...) 确保地面只推不吸
        f_spring = -config.ground_k * z_pos[contact_mask]
        f_damping = -config.ground_d * v_z[contact_mask]
        f_normal = np.maximum(0.0, f_spring + f_damping)
        physics.forces[contact_mask, 2] += f_normal
        
        # 2. 库仑摩擦力 (基于正压力 Fn)
        # 提取水平速度
        v_xy = physics.velocities[contact_mask, :2]
        v_speed_xy = np.linalg.norm(v_xy, axis=1)
        
        # 最大静摩擦/动摩擦力
        f_fric_max = config.friction_factor * f_normal
        
        # 为了数值稳定，使用平滑的摩擦力模型：Ff = -normalize(v) * min(f_max, viscosity * speed)
        # 这里的 100.0 是一个经验粘滞系数，用于在低速时平滑过渡
        viscosity = 100.0
        f_fric_magnitude = np.minimum(f_fric_max, viscosity * v_speed_xy)
        
        # 防止除以 0
        safe_speed = np.where(v_speed_xy > 1e-6, v_speed_xy, 1.0)
        f_fric_vec = -v_xy * (f_fric_magnitude / safe_speed)[:, None]
        
        physics.forces[contact_mask, :2] += f_fric_vec


def step(workspace: Workspace, n: int = 1, *, scripting: bool = True) -> None:
    """推进物理模拟 N 步。"""
    if n <= 0:
        return

    for _ in range(n):
        precompute(workspace)
        if not workspace.ui.simulate:
            return

        if scripting:
            run_script(workspace)

        _update_control_group_value(workspace)
        update_forces(workspace)
        _integrate(workspace)
        physics = workspace.physics
        if workspace.ui.record:
            if physics.frames is None:
                physics.frames = []
            physics.frames.append(workspace.topology.anchor_pos.tolist())
        physics.num_steps += 1


def _update_control_group_value(workspace: Workspace) -> None:
    """将控制组当前值平滑推进到目标值。"""
    value = workspace.physics.control_group_value
    target = workspace.physics.control_group_target
    rate = workspace.config.contraction_percent_rate
    if value.size == 0:
        return

    delta = np.clip(target - value, -rate, rate)
    value += delta
    np.clip(value, 0.0, 1.0, out=value)


def _update_contraction_percent(workspace: Workspace) -> None:
    """兼容旧调用点的别名。"""
    _update_control_group_value(workspace)


def _integrate(workspace: Workspace) -> None:
    """使用欧拉积分更新锚点速度与位置。"""
    topology = workspace.topology
    physics = workspace.physics
    config = workspace.config

    movable = ~topology.anchor_fixed
    if workspace.ui.moving_anchor and not workspace.ui.moving_body:
        movable &= workspace.ui.anchor_status == 2

    if not np.any(movable):
        return

    safe_mass = np.where(topology.anchor_mass > 1e-9, topology.anchor_mass, 1.0)
    physics.velocities[movable] += (
        physics.forces[movable] / safe_mass[movable, None]
    ) * config.h

    # 全局阻尼 (空气阻尼/关节阻尼)
    physics.velocities[movable] *= config.damping_ratio

    # 速度限制 (防止发散)
    speed = np.linalg.norm(physics.velocities[movable], axis=1)
    fast_mask = speed > 10.0  # 放宽限制到 10.0 m/s
    if np.any(fast_mask):
        limited = physics.velocities[movable]
        limited_indices = np.flatnonzero(fast_mask)
        limited[limited_indices] *= (10.0 / speed[fast_mask])[:, None]
        physics.velocities[movable] = limited

    # 位置更新
    topology.anchor_pos[movable] += physics.velocities[movable] * config.h

    # 地面硬限制 (防止数值漂移导致的穿透)
    below_ground = topology.anchor_pos[:, 2] < 0.0
    if np.any(below_ground):
        # 垂直速度设为 0 (能量由 ground_d 吸收)，位置强制回弹至 0
        physics.velocities[below_ground, 2] = 0.0
        topology.anchor_pos[below_ground, 2] = 0.0


def _rod_group_lengths(anchor_pos: np.ndarray, rod_anchors: np.ndarray) -> np.ndarray:
    if rod_anchors.size == 0:
        return np.zeros(0, dtype=np.float64)
    return np.linalg.norm(anchor_pos[rod_anchors[:, 1]] - anchor_pos[rod_anchors[:, 0]], axis=1)


def _resize_float_array(array: np.ndarray, size: int, fill: float) -> np.ndarray:
    resized = np.full(size, fill, dtype=np.float64)
    resized[: min(size, array.shape[0])] = array[:size]
    return resized


def _resize_int_array(array: np.ndarray, size: int, fill: int) -> np.ndarray:
    resized = np.full(size, fill, dtype=np.int32)
    resized[: min(size, array.shape[0])] = array[:size]
    return resized


def _resize_bool_array(array: np.ndarray, size: int, fill: bool) -> np.ndarray:
    resized = np.full(size, fill, dtype=np.bool_)
    resized[: min(size, array.shape[0])] = array[:size]
    return resized


def _resize_status_array(array: np.ndarray, size: int) -> np.ndarray:
    resized = np.zeros(size, dtype=np.int8)
    resized[: min(size, array.shape[0])] = array[:size]
    return resized


def _resize_float3_array(array: np.ndarray, size: int, fill: np.ndarray) -> np.ndarray:
    resized = np.tile(fill, (size, 1)).astype(np.float64)
    if array.size == 0:
        return resized
    resized[: min(size, array.shape[0])] = array[:size]
    return resized


def _resize_uint8_colors(array: np.ndarray, size: int) -> np.ndarray:
    resized = np.zeros((size, 3), dtype=np.uint8)
    if array.size == 0:
        return resized
    resized[: min(size, array.shape[0])] = array[:size]
    return resized


def _resize_script(array: np.ndarray, *, num_channels: int, num_actions: int) -> np.ndarray:
    resized = np.zeros((num_channels, num_actions), dtype=np.float64)
    if array.size == 0:
        return resized

    channel_count = min(num_channels, array.shape[0])
    action_count = min(num_actions, array.shape[1] if array.ndim > 1 else 0)
    if action_count > 0:
        resized[:channel_count, :action_count] = array[:channel_count, :action_count]
    return resized


def _lerp(start: np.ndarray, end: np.ndarray, t: np.ndarray) -> np.ndarray:
    return start + (end - start) * t
