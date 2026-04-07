"""网格拓扑操作系统核心。

集成了对 PneuMesh 无向图模型的各类直接操作，支持顶点的提取、修改及图结构边界与连通成分的诊断功能。
"""

from __future__ import annotations

import numpy as np

from .engine import precompute
from .workspace import Workspace


def selected_vertex_indices(workspace: Workspace) -> np.ndarray:
    """返回当前选中的顶点索引。

    Args:
        workspace: 当前工作区。

    Returns:
        选中顶点索引数组。
    """
    return np.flatnonzero(workspace.ui.vertex_status == 2)


def selected_edge_indices(workspace: Workspace) -> np.ndarray:
    """返回当前选中的边索引。

    Args:
        workspace: 当前工作区。

    Returns:
        选中边索引数组。
    """
    return np.flatnonzero(workspace.ui.edge_status == 2)


def clear_selection(workspace: Workspace) -> None:
    """清空顶点、边和面的选择状态。

    Args:
        workspace: 当前工作区。
    """
    workspace.ui.vertex_status.fill(0)
    workspace.ui.edge_status.fill(0)
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
    workspace.ui.vertex_status[index] = 2


def toggle_vertex_selection(workspace: Workspace, index: int) -> None:
    """切换指定顶点的选中状态。

    Args:
        workspace: 当前工作区。
        index: 顶点索引。
    """
    current = workspace.ui.vertex_status[index]
    workspace.ui.vertex_status[index] = 0 if current == 2 else 2


def select_edge(workspace: Workspace, index: int, *, additive: bool = False) -> None:
    """选中指定边；非增量模式会先清空已有选择。

    Args:
        workspace: 当前工作区。
        index: 边索引。
        additive: 是否保留已有选择。
    """
    if not additive:
        clear_selection(workspace)
    workspace.ui.edge_status[index] = 2


def toggle_edge_selection(workspace: Workspace, index: int) -> None:
    """切换指定边的选中状态。

    Args:
        workspace: 当前工作区。
        index: 边索引。
    """
    current = workspace.ui.edge_status[index]
    workspace.ui.edge_status[index] = 0 if current == 2 else 2


def add_joint(workspace: Workspace, source_index: int | None = None) -> int:
    """新增一个顶点，并在给定源顶点时自动创建连接边。

    Args:
        workspace: 当前工作区。
        source_index: 源顶点索引。

    Returns:
        新顶点索引。
    """
    topology = workspace.topology
    config = workspace.config

    if source_index is None:
        if topology.vertices.size == 0:
            source = np.zeros(3, dtype=np.float64)
        else:
            source = topology.vertices.mean(axis=0)
    else:
        source = topology.vertices[source_index]

    new_vertex = source.copy()
    new_vertex[0] += config.default_min_length

    topology.vertices = np.vstack([topology.vertices, new_vertex])
    topology.fixed_vs = np.append(topology.fixed_vs, False)
    topology.anchor_ids.append(f"s{topology.vertices.shape[0]}")
    topology.anchor_mass = np.append(topology.anchor_mass, 1.0)
    topology.anchor_radius = np.append(topology.anchor_radius, 0.06)

    if source_index is not None:
        new_edge = np.asarray([[source_index, topology.vertices.shape[0] - 1]], dtype=np.int32)
        topology.edges = np.vstack([topology.edges, new_edge]) if topology.edges.size else new_edge
        topology.rod_group_ids.append(f"g{len(topology.rod_group_ids)}")

    precompute(workspace)
    workspace.physics.v0 = _record_v0(workspace.topology.vertices)
    clear_selection(workspace)
    workspace.ui.vertex_status[topology.vertices.shape[0] - 1] = 2
    return topology.vertices.shape[0] - 1


def connect_vertices(workspace: Workspace, indices: np.ndarray | None = None) -> int:
    """在指定顶点之间补齐缺失的两两连接。

    Args:
        workspace: 当前工作区。
        indices: 顶点索引数组；为空时使用当前选中顶点。

    Returns:
        新增边数量。
    """
    topology = workspace.topology
    if indices is None:
        indices = selected_vertex_indices(workspace)
    indices = np.asarray(indices, dtype=np.int32)
    if indices.size < 2:
        return 0

    existing = {tuple(sorted(edge)) for edge in topology.edges.tolist()}
    new_edges: list[list[int]] = []
    for i in range(indices.shape[0]):
        for j in range(i + 1, indices.shape[0]):
            pair = tuple(sorted((int(indices[i]), int(indices[j]))))
            if pair not in existing:
                existing.add(pair)
                new_edges.append([pair[0], pair[1]])

    if not new_edges:
        return 0

    new_edge_array = np.asarray(new_edges, dtype=np.int32)
    topology.edges = (
        np.vstack([topology.edges, new_edge_array]) if topology.edges.size else new_edge_array
    )
    for _ in range(new_edge_array.shape[0]):
        topology.rod_group_ids.append(f"g{len(topology.rod_group_ids)}")
    precompute(workspace)
    return new_edge_array.shape[0]


def remove_selected_edges(workspace: Workspace) -> int:
    """删除当前选中的边，不删除端点。

    Args:
        workspace: 当前工作区。

    Returns:
        删除的边数量。
    """
    indices = selected_edge_indices(workspace)
    if indices.size == 0:
        return 0

    keep = np.ones(workspace.topology.edges.shape[0], dtype=np.bool_)
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
    selected = selected_vertex_indices(workspace)
    if selected.size == 0:
        return 0

    keep_vertex = np.ones(workspace.topology.vertices.shape[0], dtype=np.bool_)
    keep_vertex[selected] = False
    if not np.any(keep_vertex):
        return 0

    index_map = -np.ones(workspace.topology.vertices.shape[0], dtype=np.int32)
    index_map[keep_vertex] = np.arange(np.count_nonzero(keep_vertex), dtype=np.int32)

    topology = workspace.topology
    topology.vertices = topology.vertices[keep_vertex]
    topology.fixed_vs = topology.fixed_vs[keep_vertex]
    topology.anchor_ids = [
        anchor_id for keep, anchor_id in zip(keep_vertex.tolist(), topology.anchor_ids, strict=False) if keep
    ]
    topology.anchor_mass = topology.anchor_mass[keep_vertex]
    topology.anchor_radius = topology.anchor_radius[keep_vertex]

    if topology.edges.size:
        edge_keep = keep_vertex[topology.edges[:, 0]] & keep_vertex[topology.edges[:, 1]]
        kept_edges = topology.edges[edge_keep]
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

    topology.edges = remapped_edges.astype(np.int32, copy=False)
    workspace.physics.v0 = _record_v0(topology.vertices)
    precompute(workspace)
    clear_selection(workspace)
    return selected.size


def move_selected_vertices(workspace: Workspace, delta: np.ndarray) -> int:
    """将所有选中顶点按给定位移向量平移。

    Args:
        workspace: 当前工作区。
        delta: 位移向量。

    Returns:
        被平移的顶点数量。
    """
    selected = selected_vertex_indices(workspace)
    if selected.size == 0:
        return 0

    workspace.topology.vertices[selected] += np.asarray(delta, dtype=np.float64)
    workspace.physics.v0 = _record_v0(workspace.topology.vertices)
    precompute(workspace)
    return selected.size


def set_selected_vertices_position(workspace: Workspace, index: int, position: np.ndarray) -> None:
    """将指定顶点设置到绝对坐标，并同步更新 ``v0``。

    Args:
        workspace: 当前工作区。
        index: 顶点索引。
        position: 目标坐标。
    """
    workspace.topology.vertices[index] = np.asarray(position, dtype=np.float64)
    workspace.physics.v0 = _record_v0(workspace.topology.vertices)
    precompute(workspace)


def set_selected_fixed(workspace: Workspace, fixed: bool) -> int:
    """批量设置选中顶点的固定状态。

    Args:
        workspace: 当前工作区。
        fixed: 固定状态。

    Returns:
        更新的顶点数量。
    """
    selected = selected_vertex_indices(workspace)
    if selected.size == 0:
        return 0
    workspace.topology.fixed_vs[selected] = fixed
    return selected.size


def center_model(workspace: Workspace) -> None:
    """将模型在 XY 平面居中，并将几何整体上移到地面之上。

    Args:
        workspace: 当前工作区。
    """
    vertices = workspace.topology.vertices
    if vertices.size == 0:
        return
    centroid = vertices.mean(axis=0)
    z_min = vertices[:, 2].min()
    z_max = vertices[:, 2].max()
    vertices[:, 0] -= centroid[0]
    vertices[:, 1] -= centroid[1]
    vertices[:, 2] -= z_min
    vertices[:, 2] += z_max - z_min
    workspace.physics.v0 = _record_v0(vertices)
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
        topology.edges = topology.edges[keep]
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
