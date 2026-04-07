"""命令行接口模块。

定义了应用级的 CLI 命令和操作集，通过 Typer 暴露命令或封装可执行的高级命令，如加载、保存和测试工作区等功能。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from .config import load_config
from .history import WorkspaceHistory, snapshots_equal
from .schema import dump_workspace_file, load_workspace_file
from .topology import (
    add_joint,
    center_model,
    clear_selection,
    connect_vertices,
    remove_selected_edges,
    remove_selected_vertices,
    select_edge,
    select_vertex,
    set_selected_fixed,
    set_selected_vertices_position,
    toggle_edge_selection,
    toggle_vertex_selection,
)
from .workspace import Workspace, WorkspaceSnapshot


def load_workspace_from_paths(*, config_path: Path, example_path: Path) -> Workspace:
    """从给定路径加载一个新的工作区。"""
    config = load_config(config_path)
    workspace_file = load_workspace_file(example_path)
    return Workspace.from_file_data(workspace_file, config)


def save_workspace_to_path(workspace: Workspace, path: Path) -> None:
    """将当前工作区导出到指定 JSON 路径。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if workspace.storage_format == "vgtr":
        dump_workspace_file(path, workspace.to_workspace_file())
        return
    dump_workspace_file(path, workspace.to_legacy_file())


def apply_edit(
    workspace: Workspace,
    history: WorkspaceHistory,
    action: Callable[[], object],
) -> bool:
    """执行一次会改动工作区的命令，并在必要时写入历史。"""
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
    """在拖拽结束时，将起始快照写入历史。"""
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
    """批量固定或解固定当前选中顶点。"""
    return apply_edit(workspace, history, lambda: set_selected_fixed(workspace, fixed))


def add_joint_from_selection(workspace: Workspace, history: WorkspaceHistory) -> bool:
    """基于当前选择状态添加一个新顶点。"""
    selected = np.flatnonzero(workspace.ui.vertex_status == 2)
    source = int(selected[0]) if selected.size == 1 else None
    return apply_edit(workspace, history, lambda: add_joint(workspace, source_index=source))


def add_joint_from_vertex(workspace: Workspace, history: WorkspaceHistory, index: int) -> bool:
    """从指定顶点延伸添加一个新顶点。"""
    return apply_edit(workspace, history, lambda: add_joint(workspace, source_index=index))


def connect_selected_vertices(workspace: Workspace, history: WorkspaceHistory) -> bool:
    """连接当前选中的顶点。"""
    return apply_edit(workspace, history, lambda: connect_vertices(workspace))


def remove_selected_vertices_command(workspace: Workspace, history: WorkspaceHistory) -> bool:
    """删除当前选中顶点。"""
    return apply_edit(workspace, history, lambda: remove_selected_vertices(workspace))


def remove_selected_edges_command(workspace: Workspace, history: WorkspaceHistory) -> bool:
    """删除当前选中边。"""
    return apply_edit(workspace, history, lambda: remove_selected_edges(workspace))


def center_model_command(workspace: Workspace, history: WorkspaceHistory) -> bool:
    """对模型执行居中。"""
    return apply_edit(workspace, history, lambda: center_model(workspace))


def set_selected_vertex_position(workspace: Workspace, index: int, position: np.ndarray) -> None:
    """更新选中顶点位置。"""
    set_selected_vertices_position(workspace, index, position)


def select_vertex_by_mode(workspace: Workspace, *, index: int, mode: str) -> bool:
    """根据交互模式处理顶点点击。"""
    if mode == "replace":
        select_vertex(workspace, index, additive=False)
        return False
    if mode == "toggle":
        toggle_vertex_selection(workspace, index)
        return False
    raise ValueError(f"unsupported vertex click mode: {mode}")


def select_edge_by_mode(workspace: Workspace, *, index: int, mode: str) -> bool:
    """根据交互模式处理边点击。"""
    if mode == "toggle":
        toggle_edge_selection(workspace, index)
        return False
    select_edge(workspace, index, additive=False)
    return False
