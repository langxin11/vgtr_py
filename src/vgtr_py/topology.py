"""拓扑编辑核心。"""

from __future__ import annotations

import numpy as np

from .engine import precompute
from .workspace import Workspace


def selected_anchor_indices(workspace: Workspace) -> np.ndarray:
    """返回当前选中的 anchor 索引。

    Args:
        workspace: 当前工作区。

    Returns:
        选中 anchor 索引数组。
    """
    return np.flatnonzero(workspace.ui.anchor_status == 2)


def selected_rod_group_indices(workspace: Workspace) -> np.ndarray:
    """返回当前选中的 rod_group 索引。

    Args:
        workspace: 当前工作区。

    Returns:
        选中 rod_group 索引数组。
    """
    return np.flatnonzero(workspace.ui.rod_group_status == 2)


def clear_selection(workspace: Workspace) -> None:
    """清空 anchor、rod_group 和面的选择状态。

    Args:
        workspace: 当前工作区。
    """
    workspace.ui.anchor_status.fill(0)
    workspace.ui.rod_group_status.fill(0)
    workspace.ui.face_status.fill(0)


def select_anchor(workspace: Workspace, index: int, *, additive: bool = False) -> None:
    """选中指定 anchor；非增量模式会先清空已有选择。

    Args:
        workspace: 当前工作区。
        index: anchor 索引。
        additive: 是否保留已有选择。
    """
    if not additive:
        clear_selection(workspace)
    workspace.ui.anchor_status[index] = 2


def toggle_anchor_selection(workspace: Workspace, index: int) -> None:
    """切换指定 anchor 的选中状态。

    Args:
        workspace: 当前工作区。
        index: anchor 索引。
    """
    current = workspace.ui.anchor_status[index]
    workspace.ui.anchor_status[index] = 0 if current == 2 else 2


def select_rod_group(workspace: Workspace, index: int, *, additive: bool = False) -> None:
    """选中指定 rod_group；非增量模式会先清空已有选择。

    Args:
        workspace: 当前工作区。
        index: rod_group 索引。
        additive: 是否保留已有选择。
    """
    if not additive:
        clear_selection(workspace)
    workspace.ui.rod_group_status[index] = 2


def toggle_rod_group_selection(workspace: Workspace, index: int) -> None:
    """切换指定 rod_group 的选中状态。

    Args:
        workspace: 当前工作区。
        index: rod_group 索引。
    """
    current = workspace.ui.rod_group_status[index]
    workspace.ui.rod_group_status[index] = 0 if current == 2 else 2


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
