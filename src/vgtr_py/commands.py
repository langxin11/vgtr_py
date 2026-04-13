"""UI 命令封装层。

为界面操作提供统一的命令入口，负责两件事：
1. 调用 topology.py 执行实际的拓扑/状态修改；
2. 通过 WorkspaceHistory 自动管理撤销/重做快照。

所有会改变工作区状态的 UI 操作（增删锚点、连接杆组、固定/拖拽等）都应经过这里，
以保证历史记录的一致性和可恢复性。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from .config import default_config, load_config
from .history import WorkspaceHistory, snapshots_equal
from .schema import dump_workspace_file, load_workspace_file
from .topology import (
    add_joint,
    center_model,
    clear_selection,
    connect_anchors,
    remove_selected_edges,
    remove_selected_vertices,
    select_anchor,
    select_rod_group,
    set_selected_fixed,
    set_selected_vertices_position,
    toggle_anchor_selection,
    toggle_rod_group_selection,
)
from .workspace import Workspace, WorkspaceSnapshot


def load_workspace_from_paths(*, config_path: Path | None, example_path: Path) -> Workspace:
    """从给定路径加载一个新的工作区。

    Args:
        config_path: 仿真配置文件路径；若为 None 则使用默认配置。
        example_path: 工作区 JSON 文件路径。
    Returns:
        新构建的 Workspace 实例。
    """
    config = load_config(config_path) if config_path is not None else default_config()
    workspace_file = load_workspace_file(example_path)
    return Workspace.from_file_data(workspace_file, config)


def save_workspace_to_path(workspace: Workspace, path: Path) -> None:
    """将当前工作区导出到指定 JSON 路径。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    dump_workspace_file(path, workspace.to_workspace_file())


def apply_edit(
    workspace: Workspace,
    history: WorkspaceHistory,
    action: Callable[[], object],
) -> bool:
    """执行一次会改动工作区的命令，并在状态确实发生变化时写入历史。

    这是绝大多数 UI 按钮操作的统一入口。
    """
    before = workspace.snapshot()
    action()
    if not snapshots_equal(before, workspace.snapshot()):
        history.push(before)
        return True
    return False


def complete_drag_edit(
    workspace: Workspace,
    history: WorkspaceHistory,
    before: WorkspaceSnapshot | None,
) -> bool:
    """在拖拽结束时，比较起始快照与当前状态，如有变化则写入历史。

    与 apply_edit 的区别：action 已经在拖拽过程中被调用，这里只负责补录快照。
    """
    if before is None:
        return False
    if not snapshots_equal(before, workspace.snapshot()):
        history.push(before)
        return True
    return False


def undo(workspace: Workspace, history: WorkspaceHistory) -> bool:
    """撤销上一次编辑。"""
    return history.undo(workspace)


def redo(workspace: Workspace, history: WorkspaceHistory) -> bool:
    """重做上一次撤销的编辑。"""
    return history.redo(workspace)


def clear_workspace_history(history: WorkspaceHistory) -> None:
    """清空工作区相关历史。"""
    history.clear()


def clear_workspace_selection(workspace: Workspace) -> None:
    """清空当前选择。"""
    clear_selection(workspace)


def fix_selected(workspace: Workspace, history: WorkspaceHistory, fixed: bool) -> bool:
    """批量固定或解固定当前选中 anchor。"""
    return apply_edit(workspace, history, lambda: set_selected_fixed(workspace, fixed))


def add_joint_from_selection(workspace: Workspace, history: WorkspaceHistory) -> bool:
    """基于当前选择状态添加一个新 anchor。"""
    selected = np.flatnonzero(workspace.ui.anchor_status == 2)
    source = int(selected[0]) if selected.size == 1 else None
    return apply_edit(workspace, history, lambda: add_joint(workspace, source_index=source))


def add_joint_from_anchor(workspace: Workspace, history: WorkspaceHistory, index: int) -> bool:
    """从指定 anchor 延伸添加一个新 anchor。"""
    return apply_edit(workspace, history, lambda: add_joint(workspace, source_index=index))


def connect_selected_anchors(workspace: Workspace, history: WorkspaceHistory) -> bool:
    """连接当前选中的 anchors。"""
    return apply_edit(workspace, history, lambda: connect_anchors(workspace))


def remove_selected_anchors_command(workspace: Workspace, history: WorkspaceHistory) -> bool:
    """删除当前选中 anchors。"""
    return apply_edit(workspace, history, lambda: remove_selected_vertices(workspace))


def remove_selected_edges_command(workspace: Workspace, history: WorkspaceHistory) -> bool:
    """删除当前选中的 rod groups（边）。"""
    return apply_edit(workspace, history, lambda: remove_selected_edges(workspace))


def center_model_command(workspace: Workspace, history: WorkspaceHistory) -> bool:
    """对模型执行居中。"""
    return apply_edit(workspace, history, lambda: center_model(workspace))


def set_selected_anchor_position(workspace: Workspace, index: int, position: np.ndarray) -> None:
    """更新选中 anchor 位置。

    注意：此函数不经过 apply_edit，适用于拖拽等需要连续更新位置的场景；
    拖拽结束时应调用 complete_drag_edit 将起始快照补入历史。
    """
    set_selected_vertices_position(workspace, index, position)


def assign_selected_rod_groups_control_group(
    workspace: Workspace,
    history: WorkspaceHistory,
    *,
    control_group_index: int,
) -> bool:
    """将当前选中的 rod_group 分配到指定 control_group。"""
    if control_group_index < 0 or control_group_index >= workspace.script.num_channels:
        raise ValueError(f"control_group_index out of range: {control_group_index}")

    def _apply() -> None:
        selected = np.flatnonzero(workspace.ui.rod_group_status == 2)
        if selected.size == 0:
            return
        workspace.topology.rod_control_group[selected] = control_group_index

    return apply_edit(workspace, history, _apply)


def select_anchor_by_mode(workspace: Workspace, *, index: int, mode: str) -> bool:
    """根据交互模式处理 anchor 点击。"""
    if mode == "replace":
        select_anchor(workspace, index, additive=False)
        return False
    if mode == "toggle":
        toggle_anchor_selection(workspace, index)
        return False
    raise ValueError(f"unsupported anchor click mode: {mode}")


def select_edge_by_mode(workspace: Workspace, *, index: int, mode: str) -> bool:
    """根据交互模式处理 rod_group 点击。"""
    if mode == "toggle":
        toggle_rod_group_selection(workspace, index)
        return False
    select_rod_group(workspace, index, additive=False)
    return False
