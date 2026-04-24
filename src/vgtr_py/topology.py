"""拓扑编辑核心。"""

from __future__ import annotations

import numpy as np

from .workspace import ROD_TYPE_ACTIVE, Workspace


def selected_vertex_indices(workspace: Workspace) -> np.ndarray:
    """返回当前选中的顶点索引。

    Args:
        workspace: 当前工作区。

    Returns:
        选中顶点索引数组。
    """
    return np.flatnonzero(workspace.ui.anchor_status == 2)


def selected_edge_indices(workspace: Workspace) -> np.ndarray:
    """返回当前选中的边索引。

    Args:
        workspace: 当前工作区。

    Returns:
        选中边索引数组。
    """
    return np.flatnonzero(workspace.ui.rod_group_status == 2)


# 内部兼容性别名：vertex/edge 为底层图论语义，anchor/rod_group 为业务语义
selected_anchor_indices = selected_vertex_indices
selected_rod_group_indices = selected_edge_indices


def clear_selection(workspace: Workspace) -> None:
    """清空顶点、边和面的选择状态。

    Args:
        workspace: 当前工作区。
    """
    workspace.ui.anchor_status.fill(0)
    workspace.ui.rod_group_status.fill(0)
    workspace.ui.face_status.fill(0)


def select_vertex(workspace: Workspace, index: int, *, additive: bool = False) -> None:
    """选中指定顶点；非增量模式会先清空已有选择。

    Args:
        workspace: 当前工作区。
        index: 顶点索引。
        additive: 是否保留已有选择。
    """
    if not additive:
        clear_selection(workspace)
    workspace.ui.anchor_status[index] = 2


# 内部兼容性别名
select_anchor = select_vertex


def toggle_vertex_selection(workspace: Workspace, index: int) -> None:
    """切换指定顶点的选中状态。

    Args:
        workspace: 当前工作区。
        index: 顶点索引。
    """
    current = workspace.ui.anchor_status[index]
    workspace.ui.anchor_status[index] = 0 if current == 2 else 2


# 内部兼容性别名
toggle_anchor_selection = toggle_vertex_selection


def select_edge(workspace: Workspace, index: int, *, additive: bool = False) -> None:
    """选中指定边；非增量模式会先清空已有选择。

    Args:
        workspace: 当前工作区。
        index: 边索引。
        additive: 是否保留已有选择。
    """
    if not additive:
        clear_selection(workspace)
    workspace.ui.rod_group_status[index] = 2


# 内部兼容性别名
select_rod_group = select_edge


def toggle_edge_selection(workspace: Workspace, index: int) -> None:
    """切换指定边的选中状态。

    Args:
        workspace: 当前工作区。
        index: 边索引。
    """
    current = workspace.ui.rod_group_status[index]
    workspace.ui.rod_group_status[index] = 0 if current == 2 else 2


# 内部兼容性别名
toggle_rod_group_selection = toggle_edge_selection


def add_joint(workspace: Workspace, source_index: int | None = None) -> int:
    """新增一个 anchor，并在给定源 anchor 时自动创建连接 rod_group。

    Args:
        workspace: 当前工作区。
        source_index: 源 anchor 索引。

    Returns:
        新 anchor 索引。
    """
    topology = workspace.topology
    config = workspace.config
    robot = workspace.robot_config

    if source_index is None:
        if topology.anchor_pos.size == 0:
            source = np.zeros(3, dtype=np.float64)
        else:
            source = topology.anchor_pos.mean(axis=0)
    else:
        source = topology.anchor_pos[source_index]

    new_anchor = source.copy()
    new_anchor[0] += config.default_min_length

    topology.anchor_pos = np.vstack([topology.anchor_pos, new_anchor])
    topology.anchor_fixed = np.append(topology.anchor_fixed, False)
    topology.anchor_ids.append(f"s{topology.anchor_pos.shape[0]}")
    topology.anchor_mass = np.append(topology.anchor_mass, float(robot.anchor.mass))
    topology.anchor_radius = np.append(topology.anchor_radius, float(robot.anchor.radius))

    if source_index is not None:
        new_rod_group = np.asarray(
            [[source_index, topology.anchor_pos.shape[0] - 1]],
            dtype=np.int32,
        )
        topology.rod_anchors = (
            np.vstack([topology.rod_anchors, new_rod_group])
            if topology.rod_anchors.size
            else new_rod_group
        )
        topology.rod_group_ids.append(f"g{len(topology.rod_group_ids)}")

    precompute(workspace)
    workspace.physics.v0 = _record_v0(workspace.topology.anchor_pos)
    clear_selection(workspace)
    workspace.ui.anchor_status[topology.anchor_pos.shape[0] - 1] = 2
    return topology.anchor_pos.shape[0] - 1


def connect_anchors(workspace: Workspace, indices: np.ndarray | None = None) -> int:
    """在指定 anchor 之间补齐缺失的两两连接。

    Args:
        workspace: 当前工作区。
        indices: anchor 索引数组；为空时使用当前选中 anchors。

    Returns:
        新增 rod_group 数量。
    """
    topology = workspace.topology
    if indices is None:
        indices = selected_anchor_indices(workspace)
    indices = np.asarray(indices, dtype=np.int32)
    if indices.size < 2:
        return 0

    existing = {tuple(sorted(anchors)) for anchors in topology.rod_anchors.tolist()}
    new_rod_groups: list[list[int]] = []
    for i in range(indices.shape[0]):
        for j in range(i + 1, indices.shape[0]):
            pair = tuple(sorted((int(indices[i]), int(indices[j]))))
            if pair not in existing:
                existing.add(pair)
                new_rod_groups.append([pair[0], pair[1]])

    if not new_rod_groups:
        return 0

    new_rod_group_array = np.asarray(new_rod_groups, dtype=np.int32)
    topology.rod_anchors = (
        np.vstack([topology.rod_anchors, new_rod_group_array])
        if topology.rod_anchors.size
        else new_rod_group_array
    )
    for _ in range(new_rod_group_array.shape[0]):
        topology.rod_group_ids.append(f"g{len(topology.rod_group_ids)}")
    precompute(workspace)
    return new_rod_group_array.shape[0]


def remove_selected_edges(workspace: Workspace) -> int:
    """删除当前选中的边，不删除端点。

    Args:
        workspace: 当前工作区。

    Returns:
        删除的边数量。
    """
    indices = selected_rod_group_indices(workspace)
    if indices.size == 0:
        return 0

    keep = np.ones(workspace.topology.rod_anchors.shape[0], dtype=np.bool_)
    keep[indices] = False
    _filter_edges(workspace, keep)
    clear_selection(workspace)
    return indices.size


def remove_selected_vertices(workspace: Workspace) -> int:
    """删除当前选中的顶点，并同步清理关联边与索引映射。

    Args:
        workspace: 当前工作区。

    Returns:
        删除的顶点数量。
    """
    selected = selected_anchor_indices(workspace)
    if selected.size == 0:
        return 0

    keep_vertex = np.ones(workspace.topology.anchor_pos.shape[0], dtype=np.bool_)
    keep_vertex[selected] = False
    if not np.any(keep_vertex):
        return 0

    index_map = -np.ones(workspace.topology.anchor_pos.shape[0], dtype=np.int32)
    index_map[keep_vertex] = np.arange(np.count_nonzero(keep_vertex), dtype=np.int32)

    topology = workspace.topology
    topology.anchor_pos = topology.anchor_pos[keep_vertex]
    topology.anchor_fixed = topology.anchor_fixed[keep_vertex]
    topology.anchor_ids = [
        anchor_id
        for keep, anchor_id in zip(keep_vertex.tolist(), topology.anchor_ids, strict=False)
        if keep
    ]
    topology.anchor_mass = topology.anchor_mass[keep_vertex]
    topology.anchor_radius = topology.anchor_radius[keep_vertex]

    if topology.rod_anchors.size:
        edge_keep = (
            keep_vertex[topology.rod_anchors[:, 0]] & keep_vertex[topology.rod_anchors[:, 1]]
        )
        kept_edges = topology.rod_anchors[edge_keep]
        topology.rod_group_ids = [
            rod_group_id
            for keep, rod_group_id in zip(edge_keep.tolist(), topology.rod_group_ids, strict=False)
            if keep
        ]
        topology.rod_rest_length = (
            topology.rod_rest_length[edge_keep]
            if topology.rod_rest_length.shape[0]
            else topology.rod_rest_length[:0]
        )
        topology.rod_min_length = (
            topology.rod_min_length[edge_keep]
            if topology.rod_min_length.shape[0]
            else topology.rod_min_length[:0]
        )
        topology.rod_control_group = (
            topology.rod_control_group[edge_keep]
            if topology.rod_control_group.shape[0]
            else topology.rod_control_group[:0]
        )
        topology.rod_enabled = (
            topology.rod_enabled[edge_keep]
            if topology.rod_enabled.shape[0]
            else topology.rod_enabled[:0]
        )
        topology.rod_actuated = (
            topology.rod_actuated[edge_keep]
            if topology.rod_actuated.shape[0]
            else topology.rod_actuated[:0]
        )
        topology.rod_group_mass = (
            topology.rod_group_mass[edge_keep]
            if topology.rod_group_mass.shape[0]
            else topology.rod_group_mass[:0]
        )
        topology.rod_radius = (
            topology.rod_radius[edge_keep]
            if topology.rod_radius.shape[0]
            else topology.rod_radius[:0]
        )
        topology.rod_sleeve_half = (
            topology.rod_sleeve_half[edge_keep]
            if topology.rod_sleeve_half.shape[0]
            else topology.rod_sleeve_half[:0]
        )
        remapped_edges = index_map[kept_edges]
    else:
        topology.rod_group_ids = []
        topology.rod_rest_length = topology.rod_rest_length[:0]
        topology.rod_min_length = topology.rod_min_length[:0]
        topology.rod_control_group = topology.rod_control_group[:0]
        topology.rod_enabled = topology.rod_enabled[:0]
        topology.rod_actuated = topology.rod_actuated[:0]
        topology.rod_group_mass = topology.rod_group_mass[:0]
        topology.rod_radius = topology.rod_radius[:0]
        topology.rod_sleeve_half = topology.rod_sleeve_half[:0]
        remapped_edges = np.zeros((0, 2), dtype=np.int32)

    topology.rod_anchors = remapped_edges.astype(np.int32, copy=False)
    workspace.physics.v0 = _record_v0(topology.anchor_pos)
    precompute(workspace)
    clear_selection(workspace)
    return selected.size


def move_selected_anchors(workspace: Workspace, delta: np.ndarray) -> int:
    """将所有选中 anchor 按给定位移向量平移。

    Args:
        workspace: 当前工作区。
        delta: 位移向量。

    Returns:
        被平移的 anchor 数量。
    """
    selected = selected_anchor_indices(workspace)
    if selected.size == 0:
        return 0

    workspace.topology.anchor_pos[selected] += np.asarray(delta, dtype=np.float64)
    workspace.physics.v0 = _record_v0(workspace.topology.anchor_pos)
    precompute(workspace)
    return selected.size


def set_selected_vertices_position(workspace: Workspace, index: int, position: np.ndarray) -> None:
    """将指定顶点设置到绝对坐标，并同步更新 ``v0``。

    Args:
        workspace: 当前工作区。
        index: 顶点索引。
        position: 目标坐标。
    """
    workspace.topology.anchor_pos[index] = np.asarray(position, dtype=np.float64)
    workspace.physics.v0 = _record_v0(workspace.topology.anchor_pos)
    precompute(workspace)


def set_selected_fixed(workspace: Workspace, fixed: bool) -> int:
    """批量设置选中顶点的固定状态。

    Args:
        workspace: 当前工作区。
        fixed: 固定状态。

    Returns:
        更新的顶点数量。
    """
    selected = selected_anchor_indices(workspace)
    if selected.size == 0:
        return 0
    workspace.topology.anchor_fixed[selected] = fixed
    return selected.size


def center_model(workspace: Workspace) -> None:
    """将模型在 XY 平面居中，并将几何整体上移到地面之上。

    Args:
        workspace: 当前工作区。
    """
    anchors = workspace.topology.anchor_pos
    if anchors.size == 0:
        return
    centroid = anchors.mean(axis=0)
    z_min = anchors[:, 2].min()
    z_max = anchors[:, 2].max()
    anchors[:, 0] -= centroid[0]
    anchors[:, 1] -= centroid[1]
    anchors[:, 2] -= z_min
    anchors[:, 2] += z_max - z_min
    workspace.physics.v0 = _record_v0(anchors)
    precompute(workspace)


def _filter_edges(workspace: Workspace, keep: np.ndarray, *, preserve_edges: bool = False) -> None:
    """按掩码过滤边及其并行属性数组。

    Args:
        workspace: 当前工作区。
        keep: 边保留掩码。
        preserve_edges: 是否跳过边数组本身的过滤。
    """
    topology = workspace.topology
    if not preserve_edges:
        topology.rod_anchors = topology.rod_anchors[keep]
    topology.rod_group_ids = [
        rod_group_id
        for include, rod_group_id in zip(keep.tolist(), topology.rod_group_ids, strict=False)
        if include
    ]
    topology.rod_rest_length = (
        topology.rod_rest_length[keep]
        if topology.rod_rest_length.shape[0]
        else topology.rod_rest_length[:0]
    )
    topology.rod_min_length = (
        topology.rod_min_length[keep]
        if topology.rod_min_length.shape[0]
        else topology.rod_min_length[:0]
    )
    topology.rod_control_group = (
        topology.rod_control_group[keep]
        if topology.rod_control_group.shape[0]
        else topology.rod_control_group[:0]
    )
    topology.rod_enabled = (
        topology.rod_enabled[keep]
        if topology.rod_enabled.shape[0]
        else topology.rod_enabled[:0]
    )
    topology.rod_actuated = (
        topology.rod_actuated[keep]
        if topology.rod_actuated.shape[0]
        else topology.rod_actuated[:0]
    )
    topology.rod_group_mass = (
        topology.rod_group_mass[keep]
        if topology.rod_group_mass.shape[0]
        else topology.rod_group_mass[:0]
    )
    topology.rod_radius = (
        topology.rod_radius[keep]
        if topology.rod_radius.shape[0]
        else topology.rod_radius[:0]
    )
    topology.rod_sleeve_half = (
        topology.rod_sleeve_half[keep]
        if topology.rod_sleeve_half.shape[0]
        else topology.rod_sleeve_half[:0]
    )
    precompute(workspace)


def _record_v0(vertices: np.ndarray) -> np.ndarray:
    """记录基准位形，将模型整体抬升至地面之上。

    Args:
        vertices: 锚点位置数组，shape (N, 3)。

    Returns:
        抬升后的基准位形，保证所有锚点 z >= 0。
    """
    if vertices.size == 0:
        return vertices.copy()
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    z_offset = -bbox_min[2]
    v0 = vertices.copy()
    v0[:, 2] += z_offset
    if bbox_max[2] != bbox_min[2]:
        v0[:, 2] += bbox_max[2] - bbox_min[2]
    return v0
def precompute(workspace: Workspace) -> None:
    """同步拓扑与运行时数组尺寸，并刷新派生长度缓存。"""
    sync_workspace_shapes(workspace)


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

    topology.anchor_fixed = _resize_bool_array(topology.anchor_fixed, num_anchors, False)
    topology.anchor_projection_target = _resize_bool_array(
        topology.anchor_projection_target, num_anchors, False
    )
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
        topology.rod_rest_length, num_rod_groups, config.default_max_length
    )
    topology.rod_min_length = _resize_float_array(
        topology.rod_min_length,
        num_rod_groups,
        config.default_min_length,
    )
    topology.rod_enabled = _resize_bool_array(topology.rod_enabled, num_rod_groups, True)
    topology.rod_actuated = _resize_bool_array(topology.rod_actuated, num_rod_groups, False)
    topology.rod_control_group = _resize_int_array(topology.rod_control_group, num_rod_groups, 0)
    topology.rod_group_mass = _resize_float_array(topology.rod_group_mass, num_rod_groups, 1.0)
    topology.rod_type = _resize_int_array(topology.rod_type, num_rod_groups, ROD_TYPE_ACTIVE)
    topology.rod_length_limits = _resize_float2_array(
        topology.rod_length_limits,
        num_rod_groups,
        np.asarray(
            [float(config.default_min_length), float(config.default_max_length)],
            dtype=np.float64,
        ),
    )
    topology.rod_force_limits = _resize_float2_array(
        topology.rod_force_limits,
        num_rod_groups,
        np.asarray(
            [
                -float(config.k) * max(float(robot.rod_group.length_delta), 1.0),
                float(config.k) * max(float(robot.rod_group.length_delta), 1.0),
            ],
            dtype=np.float64,
        ),
    )
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



def _rod_group_lengths(anchor_pos: np.ndarray, rod_anchors: np.ndarray) -> np.ndarray:
    """计算所有杆组的当前长度。

    Args:
        anchor_pos: 锚点位置数组，shape (N, 3)。
        rod_anchors: 杆组两端锚点索引，shape (R, 2)。

    Returns:
        各杆组长度，shape (R,)。
    """
    if rod_anchors.size == 0:
        return np.zeros(0, dtype=np.float64)
    return np.linalg.norm(anchor_pos[rod_anchors[:, 1]] - anchor_pos[rod_anchors[:, 0]], axis=1)


def _resize_float_array(array: np.ndarray, size: int, fill: float) -> np.ndarray:
    """将一维浮点数组扩缩至目标长度，保留原有数据并以填充值补足。"""
    resized = np.full(size, fill, dtype=np.float64)
    resized[: min(size, array.shape[0])] = array[:size]
    return resized


def _resize_int_array(array: np.ndarray, size: int, fill: int) -> np.ndarray:
    """将一维整型数组扩缩至目标长度，保留原有数据并以填充值补足。"""
    resized = np.full(size, fill, dtype=np.int32)
    resized[: min(size, array.shape[0])] = array[:size]
    return resized


def _resize_bool_array(array: np.ndarray, size: int, fill: bool) -> np.ndarray:
    """将一维布尔数组扩缩至目标长度，保留原有数据并以填充值补足。"""
    resized = np.full(size, fill, dtype=np.bool_)
    resized[: min(size, array.shape[0])] = array[:size]
    return resized


def _resize_status_array(array: np.ndarray, size: int) -> np.ndarray:
    """将状态数组扩缩至目标长度，原有数据保留，新增部分置 0。"""
    resized = np.zeros(size, dtype=np.int8)
    resized[: min(size, array.shape[0])] = array[:size]
    return resized


def _resize_float2_array(array: np.ndarray, size: int, fill: np.ndarray) -> np.ndarray:
    """将二维浮点数组扩缩至目标长度，保留原有数据并以填充向量补足。

    Args:
        array: 原始数组，形状可为 (N, 2) 或空。
        size: 目标长度。
        fill: 长度为 2 的填充向量。

    Returns:
        调整后的 np.float64 数组，形状 (size, 2)。
    """
    resized = np.tile(fill, (size, 1)).astype(np.float64)
    if array.size == 0:
        return resized
    resized[: min(size, array.shape[0])] = array[:size]
    return resized


def _resize_float3_array(array: np.ndarray, size: int, fill: np.ndarray) -> np.ndarray:
    """将三维浮点数组（每行 3 列）扩缩至目标长度。"""
    resized = np.tile(fill, (size, 1)).astype(np.float64)
    if array.size == 0:
        return resized
    resized[: min(size, array.shape[0])] = array[:size]
    return resized


def _resize_uint8_colors(array: np.ndarray, size: int) -> np.ndarray:
    """将 RGB 颜色数组扩缩至目标长度，新增部分置黑色。"""
    resized = np.zeros((size, 3), dtype=np.uint8)
    if array.size == 0:
        return resized
    resized[: min(size, array.shape[0])] = array[:size]
    return resized


def _resize_script(array: np.ndarray, *, num_channels: int, num_actions: int) -> np.ndarray:
    """将脚本矩阵扩缩至目标通道数与动作数，保留左上角原有数据。

    Args:
        array: 原始脚本矩阵，shape (C, A)。
        num_channels: 目标通道数。
        num_actions: 目标动作数。

    Returns:
        调整后的脚本矩阵，shape (num_channels, num_actions)。
    """
    resized = np.zeros((num_channels, num_actions), dtype=np.float64)
    if array.size == 0:
        return resized

    channel_count = min(num_channels, array.shape[0])
    action_count = min(num_actions, array.shape[1] if array.ndim > 1 else 0)
    if action_count > 0:
        resized[:channel_count, :action_count] = array[:channel_count, :action_count]
    return resized


